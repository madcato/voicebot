//! Pipeline benchmark: measures STT → LLM → TTS latency using WAV fixtures.
//!
//! Runs 3 Spanish WAV files through the real pipeline and prints per-stage
//! timings plus a summary table.
//!
//! ```sh
//! RUST_LOG=performance=debug cargo run --bin bench_pipeline
//! ```

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing_subscriber::EnvFilter;

use voicebot::config::Config;
use voicebot::llm::{LlamaClient, LlmSession, StreamToken};
use voicebot::stt::WhisperStt;
use voicebot::tts::{SayTts, SentenceSplitter, TtsEngine};
use whisper_rs::install_logging_hooks;
#[cfg(feature = "kokoro")]
use voicebot::tts::KokoroTts;
#[cfg(feature = "avspeech")]
use voicebot::tts::AvSpeechTts;

const WAV_FILES: &[&str] = &[
    "tests/fixtures/es_short_greeting.wav",
    "tests/fixtures/es_short_numbers.wav",
    "tests/fixtures/es_long_intro.wav",
];

/// Perceived latency breakdown: user stops speaking → first audio plays.
/// Total = VAD + LLM TTFT + TTS 1st sentence.
/// STT is excluded because it runs in parallel with VAD (streaming Whisper).
struct BenchResult {
    file: String,
    vad_ms: u32,
    #[allow(dead_code)]
    stt_ms: u128, // informational only, not in perceived total
    llm_ttft_ms: u128,
    tts_first_ms: u128,
}

impl BenchResult {
    /// Total perceived latency from user silence to first audio output.
    fn total_ms(&self) -> u128 {
        self.vad_ms as u128 + self.llm_ttft_ms + self.tts_first_ms
    }
}

fn load_wav_as_f32(path: &str) -> Result<Vec<f32>> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path).with_context(|| format!("opening {path}"))?;
    let mut reader =
        hound::WavReader::new(BufReader::new(file)).with_context(|| format!("parsing WAV {path}"))?;

    let spec = reader.spec();
    anyhow::ensure!(spec.sample_rate == 16000, "WAV must be 16kHz, got {}Hz", spec.sample_rate);
    anyhow::ensure!(spec.channels == 1, "WAV must be mono, got {} channels", spec.channels);

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    Ok(samples)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Disable whisper prints into console
    install_logging_hooks();
    
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();
    dotenvy::dotenv().ok();

    let config = Config::from_env()?;

    // ── STT init ─────────────────────────────────────────────────────────────
    let whisper_model = config.whisper_model.clone();
    let whisper_lang = config.language.clone();
    println!("Loading Whisper model: {whisper_model}");
    let stt = Arc::new(
        tokio::task::spawn_blocking(move || WhisperStt::new(&whisper_model, &whisper_lang)).await??,
    );

    // ── LLM init ─────────────────────────────────────────────────────────────
    let llm_client = LlamaClient::new(
        &config.llm_url,
        &config.llm_model,
        config.llm_max_tokens,
        config.llm_temperature,
        config.llm_slot_id,
        config.llm_background_slot_id,
    )
    .with_provider(&config.llm_provider)
    .with_api_key(&config.llm_api_key);
    println!("LLM endpoint: {} (provider: {})", config.llm_url, config.llm_provider);

    // ── TTS init ─────────────────────────────────────────────────────────────
    let tts: TtsEngine = match config.tts_provider.as_str() {
        #[cfg(feature = "avspeech")]
        "avspeech" => {
            let voice = config.avspeech_voice.clone();
            let rate = config.avspeech_rate;
            TtsEngine::AvSpeech(tokio::task::spawn_blocking(move || AvSpeechTts::new(&voice, rate)).await??)
        }
        #[cfg(not(feature = "avspeech"))]
        "avspeech" => anyhow::bail!("TTS_PROVIDER=avspeech requires the 'avspeech' feature"),
        #[cfg(feature = "kokoro")]
        "kokoro" => {
            TtsEngine::Kokoro(KokoroTts::new(
                &config.kokoro_model,
                &config.kokoro_voices,
                &config.kokoro_voice,
                &config.kokoro_language,
            ).await?)
        }
        #[cfg(not(feature = "kokoro"))]
        "kokoro" => anyhow::bail!("TTS_PROVIDER=kokoro requires the 'kokoro' feature"),
        _ => {
            let voice = config.say_voice.clone();
            let rate = config.say_rate;
            println!("TTS provider: say (voice={voice}, rate={rate}wpm)");
            TtsEngine::Say(tokio::task::spawn_blocking(move || SayTts::new(&voice, rate)).await??)
        }
    };
    let tts = Arc::new(tts);

    let vad_ms = config.vad_silence_ms;

    println!("\n{}", "=".repeat(60));
    println!("  Pipeline Benchmark — Perceived Latency");
    println!("  (user stops speaking → first audio plays)");
    println!("  VAD silence: {}ms (from config)", vad_ms);
    println!("{}\n", "=".repeat(60));

    let mut results: Vec<BenchResult> = Vec::new();
    let mut bench_sentence: Option<String> = None; // saved for TTS comparison

    for (i, wav_path) in WAV_FILES.iter().enumerate() {
        println!("── [{}/{}] {} ──", i + 1, WAV_FILES.len(), wav_path);

        // Load WAV
        let audio = load_wav_as_f32(wav_path)?;
        let audio_duration_ms = (audio.len() as f64 / 16000.0 * 1000.0) as u128;
        println!("  Audio: {}ms ({} samples)", audio_duration_ms, audio.len());

        // ── STT ──────────────────────────────────────────────────────────────
        let stt_c = Arc::clone(&stt);
        let t = Instant::now();
        let transcript = tokio::task::spawn_blocking(move || stt_c.transcribe(&audio)).await??;
        let stt_ms = t.elapsed().as_millis();
        println!("  STT:          {:>6}ms  → {:?}", stt_ms, transcript);

        if transcript.trim().is_empty() {
            println!("  SKIP: empty transcript\n");
            continue;
        }

        // ── LLM ──────────────────────────────────────────────────────────────
        let mut session = LlmSession::new(&config.llm_system_prompt, config.llm_slot_id);
        session.add_user_turn(&transcript);
        let messages = session.all_messages_api();

        let t = Instant::now();
        let mut rx = llm_client.stream(&messages, &[]).await?;
        let mut llm_ttft_ms: Option<u128> = None;
        let mut full_response = String::new();

        while let Some(token) = rx.recv().await {
            match token {
                StreamToken::Content(s) => {
                    if llm_ttft_ms.is_none() && !s.is_empty() {
                        llm_ttft_ms = Some(t.elapsed().as_millis());
                    }
                    full_response.push_str(&s);
                }
                StreamToken::ToolCall { .. } => {}
            }
        }
        let llm_total_ms = t.elapsed().as_millis();
        let llm_ttft_ms = llm_ttft_ms.unwrap_or(llm_total_ms);
        println!("  LLM TTFT:     {:>6}ms  (total {}ms)", llm_ttft_ms, llm_total_ms);
        println!("  LLM response: {:?}", truncate(&full_response, 80));

        // ── TTS (first sentence only matters for perceived latency) ──────────
        let mut splitter = SentenceSplitter::new();
        let mut first_sentence: Option<String> = None;
        for word in full_response.split_whitespace() {
            if let Some(s) = splitter.push(&format!("{word} ")) {
                first_sentence = Some(s);
                break;
            }
        }
        if first_sentence.is_none() {
            first_sentence = splitter.flush();
        }

        let tts_first_ms = if let Some(sentence) = &first_sentence {
            let tts_c = Arc::clone(&tts);
            let sentence = sentence.clone();
            let t = Instant::now();
            let _samples = tokio::task::spawn_blocking(move || tts_c.synthesize(&sentence)).await??;
            t.elapsed().as_millis()
        } else {
            0
        };
        println!("  TTS 1st sent: {:>6}ms  → {:?}", tts_first_ms, first_sentence.as_deref().unwrap_or(""));

        // Save the first sentence for TTS provider comparison
        if bench_sentence.is_none() {
            bench_sentence = first_sentence.clone();
        }

        let perceived = vad_ms as u128 + llm_ttft_ms + tts_first_ms;
        println!("  ─────────────────────");
        println!("  PERCEIVED:    {:>6}ms  (VAD {} + TTFT {} + TTS {})",
                 perceived, vad_ms, llm_ttft_ms, tts_first_ms);
        println!("  (STT {}ms runs parallel with VAD — not counted)\n", stt_ms);

        results.push(BenchResult {
            file: wav_path.rsplit('/').next().unwrap_or(wav_path).to_string(),
            vad_ms,
            stt_ms,
            llm_ttft_ms,
            tts_first_ms,
        });
    }

    // ── Summary table ────────────────────────────────────────────────────────
    if results.is_empty() {
        println!("No results to summarize.");
        return Ok(());
    }

    println!("╔{:═<26}╦{:═<8}╦{:═<10}╦{:═<10}╦{:═<12}╗",
             "", "", "", "", "");
    println!("║ {:<24} ║ {:>6} ║ {:>8} ║ {:>8} ║ {:>10} ║",
             "File", "VAD", "LLM TTFT", "TTS 1st", "PERCEIVED");
    println!("╠{:═<26}╬{:═<8}╬{:═<10}╬{:═<10}╬{:═<12}╣",
             "", "", "", "", "");

    for r in &results {
        println!("║ {:<24} ║ {:>5}ms║ {:>6}ms ║ {:>6}ms ║ {:>8}ms ║",
                 r.file, r.vad_ms, r.llm_ttft_ms,
                 r.tts_first_ms, r.total_ms());
    }

    let n = results.len() as u128;
    let avg_ttft = results.iter().map(|r| r.llm_ttft_ms).sum::<u128>() / n;
    let avg_tts1 = results.iter().map(|r| r.tts_first_ms).sum::<u128>() / n;
    let avg_tot = results.iter().map(|r| r.total_ms()).sum::<u128>() / n;

    println!("╠{:═<26}╬{:═<8}╬{:═<10}╬{:═<10}╬{:═<12}╣",
             "", "", "", "", "");
    println!("║ {:<24} ║ {:>5}ms║ {:>6}ms ║ {:>6}ms ║ {:>8}ms ║",
             "Average", vad_ms, avg_ttft, avg_tts1, avg_tot);
    println!("╚{:═<26}╩{:═<8}╩{:═<10}╩{:═<10}╩{:═<12}╝",
             "", "", "", "", "");

    // ── TTS Provider Comparison ────────────────────────────────────────────
    if let Some(ref sentence) = bench_sentence {
        println!("\n{}", "=".repeat(60));
        println!("  TTS Provider Comparison");
        println!("  Sentence: {:?}", truncate(sentence, 60));
        println!("{}\n", "=".repeat(60));

        const TTS_RUNS: usize = 3;
        let active_provider = config.tts_provider.as_str();

        // Collect (name, engine) pairs.
        // Reuse the main pipeline's TTS instance (already proven working) and
        // only create new instances for OTHER providers.
        let mut providers: Vec<(&str, Arc<TtsEngine>)> = Vec::new();

        // Active provider — reuse existing instance (avoids second-instance issues)
        providers.push((active_provider, Arc::clone(&tts)));

        // say — only create if not already the active provider
        if active_provider != "say" {
            let voice = config.say_voice.clone();
            let rate = config.say_rate;
            match tokio::task::spawn_blocking(move || SayTts::new(&voice, rate)).await? {
                Ok(s) => providers.push(("say", Arc::new(TtsEngine::Say(s)))),
                Err(e) => println!("  say: init failed — {e}"),
            }
        }

        // avspeech — only create if not already the active provider
        #[cfg(feature = "avspeech")]
        if active_provider != "avspeech" {
            let voice = config.avspeech_voice.clone();
            let rate = config.avspeech_rate;
            match tokio::task::spawn_blocking(move || AvSpeechTts::new(&voice, rate)).await? {
                Ok(a) => providers.push(("avspeech", Arc::new(TtsEngine::AvSpeech(a)))),
                Err(e) => println!("  avspeech: init failed — {e}"),
            }
        }

        // kokoro — only create if not already the active provider
        #[cfg(feature = "kokoro")]
        if active_provider != "kokoro" {
            match KokoroTts::new(
                &config.kokoro_model,
                &config.kokoro_voices,
                &config.kokoro_voice,
                &config.kokoro_language,
            ).await {
                Ok(k) => providers.push(("kokoro", Arc::new(TtsEngine::Kokoro(k)))),
                Err(e) => println!("  kokoro: init failed — {e}"),
            }
        }

        if !providers.is_empty() {
            println!("╔{:═<18}╦{:═<10}╦{:═<10}╦{:═<10}╦{:═<10}╗",
                     "", "", "", "", "");
            println!("║ {:<16} ║ {:>8} ║ {:>8} ║ {:>8} ║ {:>8} ║",
                     "Provider", "Run 1", "Run 2", "Run 3", "Average");
            println!("╠{:═<18}╬{:═<10}╬{:═<10}╬{:═<10}╬{:═<10}╣",
                     "", "", "", "", "");

            for (name, engine) in &providers {
                let mut times = Vec::new();
                for _ in 0..TTS_RUNS {
                    let eng = Arc::clone(engine);
                    let s = sentence.clone();
                    let t = Instant::now();
                    let r = tokio::task::spawn_blocking(move || eng.synthesize(&s)).await?;
                    if let Err(e) = r {
                        println!("║ {:<16} ║ ERROR: {} ║", name, e);
                        break;
                    }
                    times.push(t.elapsed().as_millis());
                }
                if times.len() == TTS_RUNS {
                    let avg = times.iter().sum::<u128>() / TTS_RUNS as u128;
                    println!("║ {:<16} ║ {:>6}ms ║ {:>6}ms ║ {:>6}ms ║ {:>6}ms ║",
                             name, times[0], times[1], times[2], avg);
                }
            }

            println!("╚{:═<18}╩{:═<10}╩{:═<10}╩{:═<10}╩{:═<10}╝",
                     "", "", "", "", "");
        }
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}
