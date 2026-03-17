use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
#[cfg(feature = "speaker")]
use tracing::{info, warn};

#[cfg(feature = "speaker")]
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum SpeakerVerdict {
    IsMainSpeaker { similarity: f32 },
    OtherSpeaker { similarity: f32 },
    Enrolled,
}

#[allow(dead_code)]
pub struct SpeakerVerifier {
    #[cfg(feature = "speaker")]
    extractor: EmbeddingExtractor,
    enrollment: Option<Vec<f32>>,
    enrollment_path: PathBuf,
    threshold: f32,
}

impl SpeakerVerifier {
    pub fn new(model_path: &str, enrollment_path: &Path, threshold: f32) -> Result<Self> {
        #[cfg(feature = "speaker")]
        {
            let config = ExtractorConfig {
                model: model_path.to_string(),
                debug: false,
                num_threads: Some(1),
                provider: None, // uses get_default_provider() internally
            };
            let extractor = EmbeddingExtractor::new(config)
                .map_err(|e| anyhow::anyhow!("Failed to load speaker embedding model: {e}"))?;
            let enrollment = Self::load_embedding(enrollment_path);
            if enrollment.is_some() {
                info!(target: "speaker", "Speaker enrollment loaded from {:?}", enrollment_path);
            } else {
                info!(target: "speaker", "No enrollment found — will auto-enroll first speaker");
            }
            Ok(Self {
                extractor,
                enrollment,
                enrollment_path: enrollment_path.to_path_buf(),
                threshold,
            })
        }
        #[cfg(not(feature = "speaker"))]
        {
            let _ = (model_path, threshold);
            Ok(Self {
                enrollment: None,
                enrollment_path: enrollment_path.to_path_buf(),
                threshold,
            })
        }
    }

    /// Verify mono f32 samples. sample_rate is the actual rate (usually 16000).
    pub fn verify(&mut self, sample_rate: u32, samples: &[f32]) -> SpeakerVerdict {
        #[cfg(feature = "speaker")]
        {
            let embedding = match self
                .extractor
                .compute_speaker_embedding(samples.to_vec(), sample_rate)
            {
                Ok(e) => e,
                Err(e) => {
                    warn!(target: "speaker", "Speaker embedding error: {e} — passing through");
                    return SpeakerVerdict::IsMainSpeaker { similarity: 1.0 };
                }
            };
            match &self.enrollment {
                None => {
                    if let Err(e) = Self::save_embedding(&self.enrollment_path, &embedding) {
                        warn!(target: "speaker", "Failed to save enrollment: {e}");
                    } else {
                        info!(target: "speaker", "Speaker enrolled → {:?}", self.enrollment_path);
                    }
                    self.enrollment = Some(embedding);
                    SpeakerVerdict::Enrolled
                }
                Some(enrolled) => {
                    let sim = cosine_similarity(&embedding, enrolled);
                    if sim >= self.threshold {
                        SpeakerVerdict::IsMainSpeaker { similarity: sim }
                    } else {
                        SpeakerVerdict::OtherSpeaker { similarity: sim }
                    }
                }
            }
        }
        #[cfg(not(feature = "speaker"))]
        {
            let _ = (sample_rate, samples);
            SpeakerVerdict::IsMainSpeaker { similarity: 1.0 }
        }
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
        std::fs::write(path, &bytes)
            .with_context(|| format!("Failed to write enrollment to {path:?}"))
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
        let path = dir.path().join("speaker.emb");
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
    fn stub_always_passes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("speaker.emb");
        let mut sv = SpeakerVerifier::new("unused", &path, 0.5).unwrap();
        let samples: Vec<f32> = vec![0.0; 16000];
        let verdict = sv.verify(16000, &samples);
        assert_eq!(verdict, SpeakerVerdict::IsMainSpeaker { similarity: 1.0 });
    }
}
