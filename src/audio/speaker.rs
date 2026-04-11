use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
#[cfg(feature = "speaker")]
use tracing::{info, warn};

#[cfg(feature = "speaker")]
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};

/// Result of a speaker verification check.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum SpeakerVerdict {
    /// Matched an enrolled profile. `id=0` is always the main user.
    Known {
        id: u8,
        label: String,
        similarity: f32,
    },
    /// No profile matched above the threshold.
    Unknown { similarity: f32 },
    /// First utterance from a new speaker — auto-enrolled as a new profile.
    Enrolled { id: u8, label: String },
}

/// A single enrolled speaker profile.
#[allow(dead_code)]
pub struct SpeakerProfile {
    pub id: u8,
    pub label: String,
    embedding: Vec<f32>,
}

/// Verifies and tracks multiple speaker identities.
/// The first enrolled speaker (id=0) is always the "main user".
/// Additional speakers are auto-enrolled up to `max_profiles`.
#[allow(dead_code)]
pub struct SpeakerVerifier {
    #[cfg(feature = "speaker")]
    extractor: EmbeddingExtractor,
    profiles: Vec<SpeakerProfile>,
    profiles_dir: PathBuf,
    threshold: f32,
    max_profiles: u8,
}

impl SpeakerVerifier {
    /// `enrollment_path` is used to derive the profiles directory:
    /// profiles are stored as `{parent}/speaker_0.emb`, `speaker_1.emb`, etc.
    /// For backward compatibility, an existing `enrollment_path` file is
    /// loaded as profile 0 on first run.
    pub fn new(
        model_path: &str,
        enrollment_path: &Path,
        threshold: f32,
        max_profiles: u8,
    ) -> Result<Self> {
        let profiles_dir = enrollment_path
            .parent()
            .unwrap_or(Path::new("data"))
            .to_path_buf();

        #[cfg(feature = "speaker")]
        {
            let config = ExtractorConfig {
                model: model_path.to_string(),
                debug: false,
                num_threads: Some(1),
                provider: None,
            };
            let extractor = EmbeddingExtractor::new(config)
                .map_err(|e| anyhow::anyhow!("Failed to load speaker embedding model: {e}"))?;

            let profiles = Self::load_profiles(&profiles_dir, enrollment_path, max_profiles);
            if profiles.is_empty() {
                info!(target: "speaker", "No speaker profiles found — will auto-enroll first speaker");
            } else {
                info!(target: "speaker", "Loaded {} speaker profile(s) from {:?}", profiles.len(), profiles_dir);
            }

            Ok(Self {
                extractor,
                profiles,
                profiles_dir,
                threshold,
                max_profiles,
            })
        }
        #[cfg(not(feature = "speaker"))]
        {
            let _ = (model_path, threshold, max_profiles);
            Ok(Self {
                profiles: Vec::new(),
                profiles_dir,
                threshold,
                max_profiles,
            })
        }
    }

    /// Verify mono f32 samples. Returns `Known`, `Unknown`, or `Enrolled`.
    pub fn verify(&mut self, sample_rate: u32, samples: &[f32]) -> SpeakerVerdict {
        #[cfg(feature = "speaker")]
        {
            let embedding = match self
                .extractor
                .compute_speaker_embedding(samples.to_vec(), sample_rate)
            {
                Ok(e) => e,
                Err(e) => {
                    warn!(target: "speaker", "Speaker embedding error: {e} — treating as main user");
                    return SpeakerVerdict::Known {
                        id: 0,
                        label: "Usuario".to_string(),
                        similarity: 1.0,
                    };
                }
            };

            // Find the best matching profile.
            let best = self
                .profiles
                .iter()
                .map(|p| {
                    (
                        p.id,
                        p.label.clone(),
                        cosine_similarity(&embedding, &p.embedding),
                    )
                })
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            match best {
                Some((id, label, sim)) if sim >= self.threshold => SpeakerVerdict::Known {
                    id,
                    label,
                    similarity: sim,
                },
                Some((_, _, sim)) if self.profiles.len() >= self.max_profiles as usize => {
                    // No match and registry is full.
                    SpeakerVerdict::Unknown { similarity: sim }
                }
                _ => {
                    // No match (or no profiles yet) — auto-enroll as new profile.
                    let id = self.profiles.len() as u8;
                    let label = if id == 0 {
                        "Usuario".to_string()
                    } else {
                        format!("Speaker_{id}")
                    };
                    let path = self.profile_path(id);
                    if let Err(e) = Self::save_embedding(&path, &embedding) {
                        warn!(target: "speaker", "Failed to save profile {id}: {e}");
                    } else {
                        info!(target: "speaker", "Speaker {} enrolled → {:?}", label, path);
                    }
                    self.profiles.push(SpeakerProfile {
                        id,
                        label: label.clone(),
                        embedding,
                    });
                    SpeakerVerdict::Enrolled { id, label }
                }
            }
        }
        #[cfg(not(feature = "speaker"))]
        {
            let _ = (sample_rate, samples);
            SpeakerVerdict::Known {
                id: 0,
                label: "Usuario".to_string(),
                similarity: 1.0,
            }
        }
    }

    #[allow(dead_code)]
    fn profile_path(&self, id: u8) -> PathBuf {
        self.profiles_dir.join(format!("speaker_{id}.emb"))
    }

    /// Load all persisted profiles from `profiles_dir`.
    /// Handles backward compatibility: if `speaker_0.emb` is missing but the
    /// legacy `enrollment_path` file exists, it is treated as profile 0.
    #[cfg(feature = "speaker")]
    fn load_profiles(
        profiles_dir: &Path,
        legacy_path: &Path,
        max_profiles: u8,
    ) -> Vec<SpeakerProfile> {
        let mut profiles = Vec::new();
        for id in 0..max_profiles {
            let path = profiles_dir.join(format!("speaker_{id}.emb"));
            // Backward compat: try legacy single-file path for id=0.
            let effective_path = if id == 0 && !path.exists() && legacy_path.exists() {
                legacy_path.to_path_buf()
            } else {
                path
            };
            if let Some(embedding) = Self::load_embedding(&effective_path) {
                let label = if id == 0 {
                    "Usuario".to_string()
                } else {
                    format!("Speaker_{id}")
                };
                profiles.push(SpeakerProfile {
                    id,
                    label,
                    embedding,
                });
            } else {
                break; // IDs are contiguous — stop at first gap.
            }
        }
        profiles
    }

    #[allow(dead_code)]
    fn load_embedding(path: &Path) -> Option<Vec<f32>> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() % 4 != 0 {
            return None;
        }
        Some(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        )
    }

    #[allow(dead_code)]
    fn save_embedding(path: &Path, embedding: &[f32]) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(path, &bytes).with_context(|| format!("Failed to write profile to {path:?}"))
    }
}

#[allow(dead_code)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn embedding_round_trip_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("speaker_0.emb");
        let original: Vec<f32> = vec![0.1, 0.2, -0.3, 0.4, 0.5];
        SpeakerVerifier::save_embedding(&path, &original).unwrap();
        let loaded = SpeakerVerifier::load_embedding(&path).unwrap();
        assert_eq!(original.len(), loaded.len());
        for (a, b) in original.iter().zip(&loaded) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn cosine_same_vector_is_one() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "expected 1.0, got {sim}");
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "expected 0.0, got {sim}");
    }

    #[cfg(not(feature = "speaker"))]
    #[test]
    fn stub_always_passes_as_main_user() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("speaker.emb");
        let mut sv = SpeakerVerifier::new("unused", &path, 0.5, 5).unwrap();
        let samples: Vec<f32> = vec![0.0; 16000];
        let verdict = sv.verify(16000, &samples);
        assert_eq!(
            verdict,
            SpeakerVerdict::Known {
                id: 0,
                label: "Usuario".to_string(),
                similarity: 1.0
            }
        );
    }
}
