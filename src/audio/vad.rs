use anyhow::Result;
use tracing::trace;
use voice_activity_detector::VoiceActivityDetector as SileroDetector;

/// Silero VAD requires 16kHz audio with 512-sample frames (32ms per frame).
const VAD_SAMPLE_RATE: u32 = 16_000;
const VAD_FRAME_SIZE: usize = 512;

/// Consecutive 32ms frames of speech required before signalling SpeechStart (1 frame = 32ms).
const SPEECH_FRAMES_NEEDED: u32 = 1;
/// Probability threshold above which a frame is considered speech.
const SPEECH_THRESHOLD: f32 = 0.5;

pub struct VoiceActivityDetector {
    detector: SileroDetector,
    source_rate: u32,
    /// Accumulates 16kHz samples until a full 512-sample frame is ready.
    frame_buffer: Vec<f32>,
    is_active: bool,
    speech_counter: u32,
    silence_counter: u32,
    /// Consecutive 32ms frames of silence needed before signalling SpeechEnd.
    silence_frames_needed: u32,
}

impl VoiceActivityDetector {
    /// Create a new VAD. `silence_ms` controls how long silence must persist
    /// before a `SpeechEnd` is fired. Lower values feel more responsive but
    /// risk cutting off speakers who pause mid-sentence.
    /// Typical range: 600–1500ms. Default recommended: 800ms.
    pub fn new(source_rate: u32, silence_ms: u32) -> Result<Self> {
        let detector = SileroDetector::builder()
            .sample_rate(VAD_SAMPLE_RATE)
            .chunk_size(VAD_FRAME_SIZE)
            .build()?;

        // Each Silero frame is VAD_FRAME_SIZE samples at 16kHz = 32ms.
        let silence_frames_needed = ((silence_ms as f32 / 32.0).round() as u32).max(1);
        tracing::debug!(target: "audio", "VAD silence threshold: {}ms ({} frames)", silence_ms, silence_frames_needed);

        Ok(Self {
            detector,
            source_rate,
            frame_buffer: Vec::new(),
            is_active: false,
            speech_counter: 0,
            silence_counter: 0,
            silence_frames_needed,
        })
    }

    /// Process a chunk of mono audio samples and return the current VAD state.
    ///
    /// Internally resamples to 16kHz and buffers samples until a full 512-sample
    /// Silero frame is ready. State transitions require several consecutive frames
    /// above/below the threshold before firing SpeechStart / SpeechEnd, avoiding
    /// false triggers on short transients.
    pub fn process(&mut self, samples: &[f32]) -> VadResult {
        let resampled = resample_nearest(samples, self.source_rate, VAD_SAMPLE_RATE);
        self.frame_buffer.extend(resampled);

        let mut result = None;

        while self.frame_buffer.len() >= VAD_FRAME_SIZE {
            let frame: Vec<f32> = self.frame_buffer.drain(..VAD_FRAME_SIZE).collect();
            let prob = self.detector.predict(frame);

            trace!(target: "audio", "VAD prob: {:.3}", prob);

            if prob >= SPEECH_THRESHOLD {
                self.speech_counter += 1;
                self.silence_counter = 0;

                if !self.is_active && self.speech_counter >= SPEECH_FRAMES_NEEDED {
                    self.is_active = true;
                    result = Some(VadResult::SpeechStart);
                }
            } else {
                self.silence_counter += 1;
                self.speech_counter = 0;

                if self.is_active && self.silence_counter >= self.silence_frames_needed {
                    self.is_active = false;
                    result = Some(VadResult::SpeechEnd);
                }
            }
        }

        result.unwrap_or(if self.is_active {
            VadResult::Speech
        } else {
            VadResult::Silence
        })
    }

    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.is_active = false;
        self.speech_counter = 0;
        self.silence_counter = 0;
    }
}

/// Nearest-neighbor resampling — sufficient quality for voice activity detection.
fn resample_nearest(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (samples.len() as f64 / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let src = (i as f64 * ratio) as usize;
            samples[src.min(samples.len() - 1)]
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadResult {
    Speech,
    Silence,
    SpeechStart,
    SpeechEnd,
}
