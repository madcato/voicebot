use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Piper TTS wrapper (subprocess-based).
///
/// Calls the `piper` binary per sentence with `--output_raw`, reading raw
/// signed 16-bit PCM from stdout and converting to f32.
///
/// The piper binary can be downloaded from:
///   https://github.com/rhasspy/piper/releases
/// and placed in PATH or pointed to via the `PIPER_BIN` env var.
///
/// Model files (.onnx + .onnx.json) are the same format used by the butler
/// Python project (es_ES-sharvard-medium, en_US-lessac-medium, etc.).
///
/// NOTE: piper-rs 0.1.9 is currently broken (ort 2.0.0-rc.9 API mismatch).
/// This subprocess approach is equivalent and works reliably.
#[allow(dead_code)]
pub struct PiperTts {
    /// Path to the .onnx model file
    model_onnx: PathBuf,
    sample_rate: u32,
    piper_bin: String,
}

#[allow(dead_code)]
impl PiperTts {
    /// Create a new PiperTts from a `.onnx.json` config path.
    /// The `.onnx` file must be in the same directory with the same base name.
    pub fn new(config_path: &str) -> Result<Self> {
        let config = Path::new(config_path);
        let sample_rate = Self::read_sample_rate(config)?;

        // The .onnx file is the config path with the .json stripped
        let model_onnx = config.with_extension("").to_path_buf(); // removes .json → .onnx
        if !model_onnx.exists() {
            anyhow::bail!(
                "Piper model file not found: {}  (expected alongside {})",
                model_onnx.display(),
                config_path
            );
        }

        let piper_bin = std::env::var("PIPER_BIN").unwrap_or_else(|_| "piper".to_string());

        // Validate piper binary is available
        let check = Command::new(&piper_bin).arg("--version").output();
        if check.is_err() {
            anyhow::bail!(
                "Piper binary '{}' not found. Install piper and ensure it's in PATH, \
                 or set PIPER_BIN=/path/to/piper. \
                 Download: https://github.com/rhasspy/piper/releases",
                piper_bin
            );
        }

        tracing::info!(
            "Piper TTS ready: model={} ({}Hz), bin={}",
            model_onnx.display(),
            sample_rate,
            piper_bin
        );

        Ok(Self {
            model_onnx,
            sample_rate,
            piper_bin,
        })
    }

    fn read_sample_rate(config_path: &Path) -> Result<u32> {
        #[derive(serde::Deserialize)]
        struct AudioCfg {
            sample_rate: u32,
        }
        #[derive(serde::Deserialize)]
        struct ModelCfg {
            audio: AudioCfg,
        }
        let content = std::fs::read_to_string(config_path)
            .with_context(|| format!("Cannot read Piper config: {}", config_path.display()))?;
        let cfg: ModelCfg =
            serde_json::from_str(&content).context("Failed to parse Piper model config JSON")?;
        Ok(cfg.audio.sample_rate)
    }

    /// Synthesize text to mono f32 PCM samples at `self.sample_rate()`.
    /// CPU-bound — run via `tokio::task::spawn_blocking`.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        let mut child = Command::new(&self.piper_bin)
            .args([
                "--model",
                self.model_onnx.to_str().unwrap_or_default(),
                "--output_raw",
                "--quiet",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to spawn piper process")?;

        // Write text to stdin then close it so piper knows input is done
        if let Some(stdin) = child.stdin.take() {
            let mut stdin = stdin;
            stdin
                .write_all(text.as_bytes())
                .context("Failed to write to piper stdin")?;
        }

        let output = child.wait_with_output().context("Piper process failed")?;

        if !output.status.success() {
            anyhow::bail!("Piper exited with status {}", output.status);
        }

        // Output is raw signed 16-bit PCM little-endian mono
        let samples: Vec<f32> = output
            .stdout
            .chunks_exact(2)
            .map(|b| {
                let s = i16::from_le_bytes([b[0], b[1]]);
                s as f32 / 32768.0
            })
            .collect();

        Ok(samples)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
