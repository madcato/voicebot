use anyhow::{Context, Result};
use block2::RcBlock;
use core::ptr::NonNull;
use objc2::rc::autoreleasepool;
use objc2_avf_audio::{
    AVAudioBuffer, AVAudioPCMBuffer, AVSpeechSynthesisVoice, AVSpeechSynthesizer, AVSpeechUtterance,
};
use objc2_foundation::NSString;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

// GCD FFI — dispatch synthesis onto the main queue where
// AVSpeechSynthesizer's run-loop sources live.
unsafe extern "C" {
    #[link_name = "_dispatch_main_q"]
    static DISPATCH_MAIN_Q: std::ffi::c_void;
    fn dispatch_async_f(
        queue: *const std::ffi::c_void,
        context: *mut std::ffi::c_void,
        work: unsafe extern "C" fn(*mut std::ffi::c_void),
    );
}

/// macOS AVSpeechSynthesizer TTS backend.
///
/// Uses the native macOS `AVSpeechSynthesizer` API via `objc2` bindings, writing
/// synthesized PCM audio directly into a buffer instead of routing to the
/// audio hardware. This avoids a subprocess and runs fully in-process.
///
/// **Threading**: `writeUtterance:toBufferCallback:` delivers buffer callbacks
/// via the main thread's CFRunLoop.  The application must keep the main thread
/// running CFRunLoop (see `main()` when `feature = "avspeech"`).  Synthesis is
/// dispatched to the main queue via GCD; the calling thread blocks on a Condvar
/// until the callbacks complete.
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
        let identifier = find_voice_identifier(voice_name).with_context(|| {
            format!(
                "Voice '{}' not found — run `say -v ?` to list available voices",
                voice_name
            )
        })?;

        // Probe the sample rate by synthesizing a minimal utterance.
        let sample_rate = probe_sample_rate(&identifier, rate).unwrap_or(22050);

        tracing::info!(
            target: "tts",
            "AvSpeechTts ready: voice={:?} id={} rate={:.2} sample_rate={}Hz",
            voice_name, identifier, rate, sample_rate
        );

        Ok(Self {
            voice_identifier: identifier,
            rate,
            sample_rate,
        })
    }

    /// Synthesize `text` into mono f32 PCM samples at `self.sample_rate` Hz.
    ///
    /// Safe to call from any thread — dispatches to the main queue internally.
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

    /// Print all available AVSpeechSynthesizer voices to stdout.
    pub fn list_voices() {
        autoreleasepool(|_pool| unsafe {
            let voices = AVSpeechSynthesisVoice::speechVoices();
            let mut entries: Vec<(String, String, String, String, String)> = Vec::new();
            for voice in voices.iter() {
                let name = voice.name().to_string();
                let lang = voice.language().to_string();
                let identifier = voice.identifier().to_string();
                let quality = match voice.quality() {
                    q if q == objc2_avf_audio::AVSpeechSynthesisVoiceQuality::Enhanced => {
                        "Enhanced"
                    }
                    q if q == objc2_avf_audio::AVSpeechSynthesisVoiceQuality::Premium => "Premium",
                    _ => "Default",
                };
                let gender = match voice.gender() {
                    g if g == objc2_avf_audio::AVSpeechSynthesisVoiceGender::Male => "Male",
                    g if g == objc2_avf_audio::AVSpeechSynthesisVoiceGender::Female => "Female",
                    _ => "Unspecified",
                };
                entries.push((
                    name,
                    lang,
                    quality.to_string(),
                    gender.to_string(),
                    identifier,
                ));
            }
            entries.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

            println!("Available voices for TTS provider: avspeech");
            println!(
                "{:<30} {:<10} {:<10} {:<12} {}",
                "Name", "Language", "Quality", "Gender", "Identifier"
            );
            println!("{}", "-".repeat(100));
            for (name, lang, quality, gender, identifier) in &entries {
                println!(
                    "{:<30} {:<10} {:<10} {:<12} {}",
                    name, lang, quality, gender, identifier
                );
            }
            println!("\nTotal: {} voices", entries.len());
        });
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Iterate installed voices and return the system identifier for the one whose
/// display name matches `voice_name` exactly.
fn find_voice_identifier(voice_name: &str) -> Option<String> {
    autoreleasepool(|_pool| unsafe {
        let voices = AVSpeechSynthesisVoice::speechVoices();
        for voice in voices.iter() {
            let name = voice.name().to_string();
            if name == voice_name {
                return Some(voice.identifier().to_string());
            }
        }
        None
    })
}

/// Drive a minimal synthesis call to discover the sample rate AVSpeechSynthesizer
/// will actually use for the chosen voice.
fn probe_sample_rate(identifier: &str, rate: f32) -> Option<u32> {
    let result = synth_text(".", identifier, rate).ok()?;
    drop(result);
    synth_sample_rate(identifier, rate)
}

/// Returns the sample rate of the first non-empty buffer produced for the voice.
fn synth_sample_rate(identifier: &str, rate: f32) -> Option<u32> {
    let rate_cell: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));
    let pair: Arc<(Mutex<bool>, Condvar)> = Arc::new((Mutex::new(false), Condvar::new()));

    let rate_cb = Arc::clone(&rate_cell);
    let pair_cb = Arc::clone(&pair);

    let block = RcBlock::new(move |buf: NonNull<AVAudioBuffer>| unsafe {
        let pcm = buf.cast::<AVAudioPCMBuffer>().as_ref();
        let frame_length = pcm.frameLength();

        if frame_length == 0 {
            let (lock, cvar) = &*pair_cb;
            *lock.lock().unwrap() = true;
            cvar.notify_one();
            return;
        }

        if rate_cb.lock().unwrap().is_none() {
            let sr = pcm.format().sampleRate() as u32;
            *rate_cb.lock().unwrap() = Some(sr);
        }
    });

    let block_ptr =
        RcBlock::as_ptr(&block) as *mut block2::DynBlock<dyn Fn(NonNull<AVAudioBuffer>)>;

    // Dispatch synthesis onto the main queue and wait for completion.
    let synth_handle = dispatch_synth_on_main(".", identifier, rate, block_ptr);

    let (lock, cvar) = &*pair;
    let _result =
        cvar.wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(5), |done| !*done);

    // Drop synthesizer and block now that callbacks are done.
    drop(synth_handle);
    drop(block);
    rate_cell.lock().unwrap().take()
}

/// Core synthesis: returns raw f32 mono PCM samples for `text` spoken with
/// `identifier` at normalized `rate`.
fn synth_text(text: &str, identifier: &str, rate: f32) -> Result<Vec<f32>> {
    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let pair: Arc<(Mutex<bool>, Condvar)> = Arc::new((Mutex::new(false), Condvar::new()));

    let samples_cb = Arc::clone(&samples);
    let pair_cb = Arc::clone(&pair);

    let block = RcBlock::new(move |buf: NonNull<AVAudioBuffer>| unsafe {
        let pcm = buf.cast::<AVAudioPCMBuffer>().as_ref();
        let frame_length = pcm.frameLength() as usize;

        if frame_length == 0 {
            let (lock, cvar) = &*pair_cb;
            *lock.lock().unwrap() = true;
            cvar.notify_one();
            return;
        }

        let channel_ptrs = pcm.floatChannelData();
        if channel_ptrs.is_null() {
            return;
        }
        let ch0: *const f32 = (*channel_ptrs).as_ptr();
        let slice = std::slice::from_raw_parts(ch0, frame_length);
        samples_cb.lock().unwrap().extend_from_slice(slice);
    });

    let block_ptr =
        RcBlock::as_ptr(&block) as *mut block2::DynBlock<dyn Fn(NonNull<AVAudioBuffer>)>;

    // Dispatch synthesis onto the main queue and wait for completion.
    let synth_handle = dispatch_synth_on_main(text, identifier, rate, block_ptr);

    let (lock, cvar) = &*pair;
    let result = cvar
        .wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(30), |done| !*done)
        .unwrap();

    if result.1.timed_out() {
        tracing::warn!(target: "tts", "AvSpeechTts: synthesis timed out for {:?}", text);
    }

    // Drop synthesizer and block now that callbacks are done.
    drop(synth_handle);
    drop(block);
    let result = std::mem::take(&mut *samples.lock().unwrap());
    Ok(result)
}

// ── GCD dispatch helper ─────────────────────────────────────────────────────

/// Opaque wrapper so we can send the synthesizer pointer across threads.
/// The synthesizer is created on the main thread and dropped by the caller
/// after all callbacks have fired.
#[allow(dead_code)]
struct SynthHandle(*mut std::ffi::c_void);
unsafe impl Send for SynthHandle {}

/// Context passed through `dispatch_async_f` to the main queue.
struct SynthContext {
    text: String,
    identifier: String,
    rate: f32,
    block_ptr: *mut block2::DynBlock<dyn Fn(NonNull<AVAudioBuffer>)>,
    /// Caller-provided slot: the trampoline writes the synthesizer pointer here
    /// so the caller can drop it after callbacks complete.
    synth_out: Arc<Mutex<Option<SynthHandle>>>,
}

// SAFETY: The block_ptr points to an RcBlock that outlives the dispatch call
// (caller holds the RcBlock until after Condvar wait returns).
unsafe impl Send for SynthContext {}

/// Dispatch `writeUtterance:toBufferCallback:` onto the main GCD queue.
/// Returns an `Arc` that will hold the synthesizer once the trampoline runs.
/// The caller must keep it alive until callbacks are done, then drop it.
fn dispatch_synth_on_main(
    text: &str,
    identifier: &str,
    rate: f32,
    block_ptr: *mut block2::DynBlock<dyn Fn(NonNull<AVAudioBuffer>)>,
) -> Arc<Mutex<Option<SynthHandle>>> {
    let synth_out: Arc<Mutex<Option<SynthHandle>>> = Arc::new(Mutex::new(None));
    let ctx = Box::new(SynthContext {
        text: text.to_string(),
        identifier: identifier.to_string(),
        rate,
        block_ptr,
        synth_out: Arc::clone(&synth_out),
    });
    let ctx_ptr = Box::into_raw(ctx) as *mut std::ffi::c_void;

    unsafe {
        dispatch_async_f(
            std::ptr::addr_of!(DISPATCH_MAIN_Q),
            ctx_ptr,
            synth_trampoline,
        );
    }
    synth_out
}

/// C-callable trampoline invoked by GCD on the main thread.
unsafe extern "C" fn synth_trampoline(ctx_ptr: *mut std::ffi::c_void) {
    let ctx = unsafe { Box::from_raw(ctx_ptr as *mut SynthContext) };

    autoreleasepool(|_pool| unsafe {
        let synth = AVSpeechSynthesizer::new();

        let ns_text = NSString::from_str(&ctx.text);
        let utterance = AVSpeechUtterance::speechUtteranceWithString(&ns_text);

        let ns_id = NSString::from_str(&ctx.identifier);
        if let Some(voice) = AVSpeechSynthesisVoice::voiceWithIdentifier(&ns_id) {
            utterance.setVoice(Some(&voice));
        }
        utterance.setRate(ctx.rate);

        synth.writeUtterance_toBufferCallback(&utterance, ctx.block_ptr);

        // Transfer ownership to the caller so the synthesizer stays alive
        // while callbacks fire during subsequent run-loop iterations.
        let raw = objc2::rc::Retained::into_raw(synth) as *mut std::ffi::c_void;
        *ctx.synth_out.lock().unwrap() = Some(SynthHandle(raw));
    });
}
