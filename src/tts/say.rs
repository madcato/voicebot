use anyhow::{Context, Result};
use std::io::Write;
use std::process::{Command, Stdio};

/// macOS `say` TTS wrapper (subprocess-based).
///
/// Uses the built-in macOS `say` command with `--data-format=LEI16@22050`
/// to produce raw signed 16-bit PCM at 22050 Hz on stdout.
///
/// Voice is configured via the `SAY_VOICE` env var (default: "Marisol (Enhanced)").
/// List available voices with: `say -v ?`
///
/// Planned replacement: Kokoro TTS via onnxruntime (better quality, offline model).
pub struct SayTts {
    voice: String,
    rate: u32,
    sample_rate: u32,
}

impl SayTts {
    pub fn new(voice: &str, rate: u32) -> Result<Self> {
        // Validate the `say` binary is available
        Command::new("say")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .context("'say' command not found — this TTS backend requires macOS")?;

        tracing::info!(target: "tts", "SayTts ready: voice={:?} rate={}wpm (22050Hz)", voice, rate);

        Ok(Self {
            voice: voice.to_string(),
            rate,
            sample_rate: 22050,
        })
    }

    /// Synthesize text to mono f32 PCM samples at 22050 Hz.
    /// CPU-bound — call from `tokio::task::spawn_blocking`.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        // `say -o /dev/stdout` fails on macOS (error -54), so we write to a temp file.
        let tmp_path = std::env::temp_dir()
            .join(format!("voicebot_say_{}.raw", std::process::id()));

        let mut child = Command::new("say")
            .args([
                "-v",
                &self.voice,
                "--file-format=WAVE",
                "--data-format=LEI16",
                "-r", &self.rate.to_string(),
                "-o",
                tmp_path.to_str().unwrap(),
            ])
            .stdin(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to spawn 'say' process")?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(text.as_bytes())
                .context("Failed to write to say stdin")?;
        }

        let exit = child.wait().context("say process failed")?;
        if !exit.success() {
            anyhow::bail!("say exited with status {}", exit);
        }

        let bytes = std::fs::read(&tmp_path)
            .context("Failed to read say output file")?;
        let _ = std::fs::remove_file(&tmp_path);

        let samples = wav_to_f32(&bytes)?;

        // Prepend a short silence so CoreAudio's stream initialisation latency
        // does not clip the first word of each sentence.
        let silence_len = (self.sample_rate as usize * 30) / 1000; // 30 ms
        let mut with_silence = vec![0.0f32; silence_len];
        with_silence.extend_from_slice(&samples);
        Ok(with_silence)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Parse a WAV file and return f32 samples.
/// Reads sample_rate from the fmt chunk; finds the data chunk dynamically.
fn wav_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    // fmt chunk: bytes 24-27 = sample rate (little-endian u32)
    if bytes.len() < 44 {
        anyhow::bail!("WAV file too short ({} bytes)", bytes.len());
    }
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    tracing::debug!(target: "tts", "WAV sample_rate from header: {}", sample_rate);

    // Find "data" chunk to locate raw PCM
    let data_pos = bytes
        .windows(4)
        .position(|w| w == b"data")
        .context("No 'data' chunk in WAV output")?;
    let pcm = &bytes[data_pos + 8..]; // skip "data" (4) + chunk size (4)

    let samples = pcm
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    Ok(samples)
}
