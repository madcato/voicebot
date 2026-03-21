use anyhow::{Context, Result};
use block2::RcBlock;
use core::ptr::NonNull;
use objc2_avf_audio::{
    AVAudioBuffer, AVAudioPCMBuffer, AVSpeechSynthesizer, AVSpeechSynthesisVoice,
    AVSpeechUtterance,
};
use objc2_foundation::NSString;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// CoreFoundation FFI for running the thread's run loop so AVSpeechSynthesizer
// callbacks are dispatched on spawn_blocking threads.
unsafe extern "C" {
    fn CFRunLoopRunInMode(mode: *const std::ffi::c_void, seconds: f64, return_after: u8) -> i32;
    static kCFRunLoopDefaultMode: *const std::ffi::c_void;
}

/// Spin the current thread's CFRunLoop until `done` is true or `timeout` elapses.
/// This is required because AVSpeechSynthesizer dispatches buffer callbacks via
/// the run loop — a plain Condvar::wait blocks the thread and prevents delivery.
fn run_loop_until(done: &Arc<Mutex<bool>>, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if *done.lock().unwrap() {
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        unsafe {
            // Process pending run-loop sources for up to 10ms, then re-check done flag.
            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.01, 1);
        }
    }
}

/// macOS AVSpeechSynthesizer TTS backend.
///
/// Uses the native macOS `AVSpeechSynthesizer` API via `objc2` bindings, writing
/// synthesized PCM audio directly into a buffer instead of routing to the
/// audio hardware. This avoids a subprocess and runs fully in-process.
///
/// Voice is configured via `AVSPEECH_VOICE` (default: "Jorge (Enhanced)").
/// Rate is a normalized float [0.0, 1.0] via `AVSPEECH_RATE` (default: 0.55 ≈ 215 wpm).
/// List available voices with: `say -v ?`
///
/// Requires: `--features avspeech` at build time.
/// macOS 10.15+ required for `writeUtterance:toBufferCallback:`.
pub struct AvSpeechTts {
    /// System voice identifier, e.g. "com.apple.voice.enhanced.es-MX.Jorge".
    voice_identifier: String,
    /// Normalized speech rate [0.0, 1.0]. AVSpeechUtteranceDefaultSpeechRate = 0.5 ≈ 180 wpm.
    rate: f32,
    /// Sample rate reported by the first synthesized buffer (22050 Hz for most voices).
    sample_rate: u32,
}

impl AvSpeechTts {
    /// Find a voice by its display name (e.g. "Jorge (Enhanced)") and construct the TTS backend.
    ///
    /// `rate` — normalized speech rate [0.0, 1.0]. 0.5 is the system default (~180 wpm).
    /// Use ~0.55 for ~215 wpm.
    pub fn new(voice_name: &str, rate: f32) -> Result<Self> {
        let identifier = find_voice_identifier(voice_name)
            .with_context(|| format!(
                "Voice '{}' not found — run `say -v ?` to list available voices",
                voice_name
            ))?;

        // Probe the sample rate by synthesizing a minimal utterance.
        let sample_rate = probe_sample_rate(&identifier, rate)
            .unwrap_or(22050);

        tracing::info!(
            target: "tts",
            "AvSpeechTts ready: voice={:?} id={} rate={:.2} sample_rate={}Hz",
            voice_name, identifier, rate, sample_rate
        );

        Ok(Self { voice_identifier: identifier, rate, sample_rate })
    }

    /// Synthesize `text` into mono f32 PCM samples at `self.sample_rate` Hz.
    ///
    /// CPU-bound — call from `tokio::task::spawn_blocking`.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        let samples = synth_text(text, &self.voice_identifier, self.rate)?;

        // Prepend 30 ms silence to absorb CoreAudio stream-init latency.
        let silence_len = (self.sample_rate as usize * 30) / 1000;
        let mut with_silence = vec![0.0f32; silence_len];
        with_silence.extend_from_slice(&samples);
        Ok(with_silence)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Iterate installed voices and return the system identifier for the one whose
/// display name matches `voice_name` exactly.
fn find_voice_identifier(voice_name: &str) -> Option<String> {
    unsafe {
        let voices = AVSpeechSynthesisVoice::speechVoices();
        for voice in voices.iter() {
            let name = voice.name().to_string();
            if name == voice_name {
                return Some(voice.identifier().to_string());
            }
        }
        None
    }
}

/// Drive a minimal synthesis call to discover the sample rate AVSpeechSynthesizer
/// will actually use for the chosen voice.
fn probe_sample_rate(identifier: &str, rate: f32) -> Option<u32> {
    // "." is the shortest possible utterance.
    let result = synth_text(".", identifier, rate).ok()?;
    // The sample_rate isn't read from result length; we read it inside synth_text via the
    // buffer format. But synth_text returns Vec<f32> and we've already stored the rate in
    // the AvSpeechTts constructor. Here we just confirm synthesis worked; the real rate
    // comes from synth_sample_rate().
    drop(result);
    synth_sample_rate(identifier, rate)
}

/// Returns the sample rate of the first non-empty buffer produced for the voice.
fn synth_sample_rate(identifier: &str, rate: f32) -> Option<u32> {
    let rate_cell: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));
    let done: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));

    let rate_cb = Arc::clone(&rate_cell);
    let done_cb = Arc::clone(&done);

    let block = RcBlock::new(move |buf: NonNull<AVAudioBuffer>| {
        unsafe {
            let pcm = buf.cast::<AVAudioPCMBuffer>().as_ref();
            let frame_length = pcm.frameLength();

            if frame_length == 0 {
                *done_cb.lock().unwrap() = true;
                return;
            }

            if rate_cb.lock().unwrap().is_none() {
                let sr = pcm.format().sampleRate() as u32;
                *rate_cb.lock().unwrap() = Some(sr);
            }
        }
    });

    let block_ptr = RcBlock::as_ptr(&block)
        as *mut block2::DynBlock<dyn Fn(NonNull<AVAudioBuffer>)>;

    unsafe {
        let synth = AVSpeechSynthesizer::new();
        let ns_text = NSString::from_str(".");
        let utterance = AVSpeechUtterance::speechUtteranceWithString(&ns_text);
        let ns_id = NSString::from_str(identifier);
        if let Some(voice) = AVSpeechSynthesisVoice::voiceWithIdentifier(&ns_id) {
            utterance.setVoice(Some(&voice));
        }
        utterance.setRate(rate);
        synth.writeUtterance_toBufferCallback(&utterance, block_ptr);

        run_loop_until(&done, Duration::from_secs(5));

        drop(synth);
    }
    drop(block);

    rate_cell.lock().unwrap().take()
}

/// Core synthesis: returns raw f32 mono PCM samples for `text` spoken with `identifier`
/// at normalized `rate`.
fn synth_text(text: &str, identifier: &str, rate: f32) -> Result<Vec<f32>> {
    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let done: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));

    let samples_cb = Arc::clone(&samples);
    let done_cb = Arc::clone(&done);

    let block = RcBlock::new(move |buf: NonNull<AVAudioBuffer>| {
        unsafe {
            let pcm = buf.cast::<AVAudioPCMBuffer>().as_ref();
            let frame_length = pcm.frameLength() as usize;

            if frame_length == 0 {
                *done_cb.lock().unwrap() = true;
                return;
            }

            let channel_ptrs = pcm.floatChannelData();
            if channel_ptrs.is_null() {
                return;
            }
            let ch0: *const f32 = (*channel_ptrs).as_ptr();
            let slice = std::slice::from_raw_parts(ch0, frame_length);
            samples_cb.lock().unwrap().extend_from_slice(slice);
        }
    });

    let block_ptr = RcBlock::as_ptr(&block)
        as *mut block2::DynBlock<dyn Fn(NonNull<AVAudioBuffer>)>;

    unsafe {
        let synth = AVSpeechSynthesizer::new();

        let ns_text = NSString::from_str(text);
        let utterance = AVSpeechUtterance::speechUtteranceWithString(&ns_text);

        let ns_id = NSString::from_str(identifier);
        if let Some(voice) = AVSpeechSynthesisVoice::voiceWithIdentifier(&ns_id) {
            utterance.setVoice(Some(&voice));
        }
        utterance.setRate(rate);

        synth.writeUtterance_toBufferCallback(&utterance, block_ptr);

        // Spin the CFRunLoop so callbacks are delivered on this thread.
        run_loop_until(&done, Duration::from_secs(30));

        if !*done.lock().unwrap() {
            tracing::warn!(target: "tts", "AvSpeechTts: synthesis timed out for {:?}", text);
        }

        drop(synth);
    }
    drop(block);

    let result = std::mem::take(&mut *samples.lock().unwrap());
    Ok(result)
}
