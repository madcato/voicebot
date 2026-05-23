use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info};

use super::{ContextEntry, ContextLens};
use crate::audio::speaker::{SpeakerVerdict, SpeakerVerifier};

/// How long an identity entry stays fresh in the ContextLens.
const IDENTITY_TTL: Duration = Duration::from_secs(120);

/// Wraps `SpeakerVerifier` and writes speaker-identity context into the
/// shared `ContextLens` after each verification call.
///
/// The `verify` method is synchronous (mirrors the underlying model call) and
/// is intended to be called from the audio loop on each `SpeechEnd` event.
pub struct IdentityAnalyzer {
    verifier: SpeakerVerifier,
    lens: Arc<Mutex<ContextLens>>,
}

/// Result of a single speaker verification call.
pub struct IdentityResult {
    /// Whether the audio came from the primary enrolled user (profile index 0).
    pub is_main_speaker: bool,
    /// Human-readable label for the detected speaker.
    pub speaker_label: String,
}

impl IdentityAnalyzer {
    pub fn new(verifier: SpeakerVerifier, lens: Arc<Mutex<ContextLens>>) -> Self {
        Self { verifier, lens }
    }

    /// Verify the speaker for the given audio frame, update the ContextLens,
    /// and return a lightweight result the audio loop can act on immediately.
    pub fn verify(&mut self, sample_rate: u32, audio: &[f32]) -> IdentityResult {
        let verdict = self.verifier.verify(sample_rate, audio);

        let (is_main_speaker, speaker_label, confidence, entry_value) = match &verdict {
            SpeakerVerdict::Enrolled { id, label } => {
                let main = *id == 0;
                let value = if main {
                    format!("{label} (enrolled main user)")
                } else {
                    format!("{label} (enrolled secondary speaker)")
                };
                (main, label.clone(), 1.0_f32, value)
            }
            SpeakerVerdict::Known {
                id,
                label,
                similarity,
            } => {
                let main = *id == 0;
                let value = format!("{label} (similarity={similarity:.2})");
                (main, label.clone(), *similarity, value)
            }
            SpeakerVerdict::Unknown { similarity } => {
                let value = format!("unknown speaker (similarity={similarity:.2})");
                (false, "Ambiente".to_string(), *similarity, value)
            }
        };

        if is_main_speaker {
            debug!(target: "speaker", "Main speaker verified: {}", speaker_label);
        } else {
            info!(target: "speaker", "Non-main speaker: {} (main=false)", speaker_label);
        }

        let entry = ContextEntry {
            key: "speaker_identity",
            value: entry_value,
            confidence,
            valid_until: Instant::now() + IDENTITY_TTL,
            source: "identity_analyzer",
        };
        self.lens.lock().unwrap().upsert(entry);

        IdentityResult {
            is_main_speaker,
            speaker_label,
        }
    }
}
