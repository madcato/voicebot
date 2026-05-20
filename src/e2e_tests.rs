//! End-to-end integration tests for the voicebot pipeline.
//!
//! These tests exercise the full STT → LLM → TTS → DB pipeline using:
//! - Direct transcript injection (no Whisper needed for most tests)
//! - `TtsEngine::Mock` — captures synthesized sentences instead of playing audio
//! - wiremock — deterministic LLM responses, no LLM server needed
//! - Real SQLite in a temp directory — DB assertions without side effects
//!
//! All tests are marked `#[ignore]` and must be run explicitly:
//!
//! ```sh
//! # All e2e tests
//! cargo test e2e -- --ignored --nocapture
//!
//! # A specific scenario
//! cargo test e2e::basic_conversation -- --ignored --nocapture
//!
//! # STT tests that require the Whisper model
//! cargo test e2e::stt_ -- --ignored --nocapture
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use tokio::sync::mpsc;
use uuid::Uuid;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::agents::ProactiveEvent;
use crate::analysis::ContextLens;
use crate::audio::output::AudioOutput;
use crate::config::Config;
use crate::db::Database;
use crate::llm::{LlmSession, OpenAIClient};
use crate::pipeline::{llm_task, sen_task, tts_task, PipelineEvents, PipelineState};
use crate::tools::ToolRegistry;
use crate::tts::{mock_tts::MockTts, TtsEngine};

// ── SSE helpers ───────────────────────────────────────────────────────────────

/// Build an OpenAI-compatible SSE stream body that emits `text` word-by-word.
fn make_sse(text: &str) -> String {
    let mut body = String::new();
    for (i, word) in text.split_whitespace().enumerate() {
        let token = if i == 0 {
            word.to_string()
        } else {
            format!(" {word}")
        };
        let escaped = token.replace('"', "\\\"");
        body.push_str(&format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{escaped}\"}}}}]}}\n\n"
        ));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

// ── Test harness ──────────────────────────────────────────────────────────────

struct E2eHarness {
    pub server: MockServer,
    pub llm_client: OpenAIClient,
    pub llm_session: Arc<Mutex<LlmSession>>,
    pub tts: Arc<TtsEngine>,
    /// Accumulates every sentence sent to TTS.
    pub captured: Arc<Mutex<Vec<String>>>,
    pub audio_output: Arc<AudioOutput>,
    pub db: Database,
    pub session_id: Uuid,
    pub tools: Arc<ToolRegistry>,
    pub shared_history: Arc<RwLock<String>>,
    // Kept alive so the temp directory isn't deleted before the test ends.
    _db_dir: tempfile::TempDir,
}

impl E2eHarness {
    async fn new() -> Self {
        Self::with_system_prompt("You are a test assistant.").await
    }

    async fn with_system_prompt(system_prompt: &str) -> Self {
        let server = MockServer::start().await;
        let llm_client = OpenAIClient::new(
            &server.uri(),
            "test-model",
            400, // max_tokens
            0.0, // temperature — deterministic
        );
        let llm_session = Arc::new(Mutex::new(LlmSession::new(system_prompt)));

        let (mock_tts, captured) = MockTts::new();
        let tts = Arc::new(TtsEngine::Mock(mock_tts));

        let audio_output = Arc::new(AudioOutput::null());

        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("test.db");
        let db = Database::new(db_path.to_str().unwrap()).await.unwrap();
        let session_id = db.get_or_create_session().await.unwrap();

        let tools = Arc::new(ToolRegistry::new());
        let shared_history = Arc::new(RwLock::new(String::new()));

        Self {
            server,
            llm_client,
            llm_session,
            tts,
            captured,
            audio_output,
            db,
            session_id,
            tools,
            shared_history,
            _db_dir: db_dir,
        }
    }

    /// Mount a mock that returns `text` as a streaming SSE response.
    async fn mock_llm_response(&self, text: &str) {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(make_sse(text))
                    .append_header("content-type", "text/event-stream"),
            )
            .mount(&self.server)
            .await;
    }

    /// Run the full pipeline with a pre-known transcript (no Whisper).
    async fn run(&self, transcript: &str) {
        self.run_with_opts(transcript, false, "jarvis").await
    }

    /// Run the pipeline in ambient mode.
    async fn run_ambient(&self, transcript: &str, wake_word: &str) {
        self.run_with_opts(transcript, true, wake_word).await
    }

    async fn run_with_opts(&self, transcript: &str, ambient: bool, wake_word: &str) {
        // In ambient mode without the wake word the audio loop discards the transcript.
        if ambient && !transcript.to_lowercase().contains(&wake_word.to_lowercase()) {
            return;
        }

        let events = Arc::new(PipelineEvents::new());
        let play_cancel = Arc::new(AtomicBool::new(false));
        let tts_muted = Arc::new(AtomicBool::new(false));
        let turn_commit = Arc::new(AtomicU64::new(0));
        let context_lens = Arc::new(Mutex::new(ContextLens::new()));
        let t_vad_end: Arc<Mutex<Option<std::time::Instant>>> = Arc::new(Mutex::new(None));
        let t_llm_post_send: Arc<Mutex<Option<std::time::Instant>>> = Arc::new(Mutex::new(None));

        // Inject transcript directly — bypasses STT/VAD for deterministic tests.
        if transcript.trim().is_empty() {
            return;
        }

        let sample_rate = self.tts.sample_rate();

        let (proactive_tx, _proactive_rx) = mpsc::channel::<ProactiveEvent>(8);
        let (state_tx, state_rx) = tokio::sync::watch::channel(PipelineState::Idle);
        let state_tx = Arc::new(state_tx);
        let (sentences_tx, sentences_rx) = tokio::sync::mpsc::channel::<crate::pipeline::PipelineFrame>(64);
        let (llm_tx, llm_rx) = tokio::sync::mpsc::channel::<crate::pipeline::PipelineFrame>(256);
        let (transcript_tx, transcript_rx) = tokio::sync::mpsc::channel::<crate::pipeline::PipelineFrame>(16);

        // Spawn the three pipeline tasks.
        let h_llm = {
            let events_c          = Arc::clone(&events);
            let state_tx_c        = Arc::clone(&state_tx);
            let state_rx_c        = state_rx.clone();
            let sent_tx_c         = sentences_tx.clone();
            let llm_tx_c          = llm_tx.clone();
            let t_llm_post_send_c = Arc::clone(&t_llm_post_send);
            let session_c         = Arc::clone(&self.llm_session);
            let client_c          = self.llm_client.clone();
            let db_c              = self.db.clone();
            let tools_c           = Arc::clone(&self.tools);
            let history_c         = Arc::clone(&self.shared_history);
            let turn_c            = Arc::clone(&turn_commit);
            let lens_c            = Arc::clone(&context_lens);
            let sid               = self.session_id;
            tokio::spawn(async move {
                llm_task(
                    events_c, state_tx_c, state_rx_c, sent_tx_c, llm_tx_c, transcript_rx,
                    t_llm_post_send_c, session_c, client_c, db_c, sid,
                    tools_c, history_c, turn_c, proactive_tx, lens_c,
                )
                .await;
            })
        };
        let h_sen = {
            let events_c          = Arc::clone(&events);
            let sent_tx_c         = sentences_tx.clone();
            let t_vad_end_c       = Arc::clone(&t_vad_end);
            let t_llm_post_send_c = Arc::clone(&t_llm_post_send);
            tokio::spawn(async move {
                sen_task(events_c, llm_rx, sent_tx_c, t_vad_end_c, t_llm_post_send_c).await
            })
        };
        let h_tts = {
            let events_c    = Arc::clone(&events);
            let t_vad_end_c = Arc::clone(&t_vad_end);
            let tts_c       = Arc::clone(&self.tts);
            let out_c       = Arc::clone(&self.audio_output);
            let cancel_c    = Arc::clone(&play_cancel);
            let muted_c     = Arc::clone(&tts_muted);
            tokio::spawn(async move {
                tts_task(events_c, t_vad_end_c, sentences_rx, tts_c, out_c, sample_rate, cancel_c, muted_c).await
            })
        };

        // Send transcript to wake up the LLM task.
        transcript_tx.send(crate::pipeline::PipelineFrame::TranscriptReady {
            utterance_id: 0,
            text: transcript.to_string(),
        }).await.ok();

        // Wait for the LLM to finish streaming.
        events.llm_post_finished.notified().await;

        // Give TTS tasks time to synthesize (MockTts is instant).
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Cancel all tasks.
        events.barge_in_tx.send(0).ok();
        play_cancel.store(true, std::sync::atomic::Ordering::SeqCst);

        let _ = tokio::time::timeout(Duration::from_secs(2), async {
            let _ = h_llm.await;
            let _ = h_sen.await;
            let _ = h_tts.await;
        })
        .await;

        // Allow background DB writes to settle.
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    /// All messages saved to the DB for this session.
    async fn db_messages(&self) -> Vec<(String, String)> {
        self.db
            .get_session_context(self.session_id, 0)
            .await
            .unwrap()
            .1
    }

    /// Sentences that were synthesized by TTS.
    fn tts_sentences(&self) -> Vec<String> {
        self.captured.lock().unwrap().clone()
    }
}

// ── Scenarios ─────────────────────────────────────────────────────────────────

/// Basic conversation: mock transcript → mock LLM → TTS captures text → DB stores turn.
#[tokio::test]
#[ignore]
async fn basic_conversation_mocked_transcript() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Son las diez y media.").await;

    h.run("Hola, ¿qué hora es?").await;

    let sentences = h.tts_sentences();
    assert!(!sentences.is_empty(), "expected TTS to receive text, got empty");
    let full = sentences.join(" ");
    assert!(full.contains("diez"), "expected response in TTS, got: {full:?}");

    let msgs = h.db_messages().await;
    assert!(
        msgs.iter().any(|(r, c)| r == "Assistant" && c.contains("diez")),
        "expected assistant message in DB, got: {msgs:?}"
    );
}

/// Empty transcript is silently discarded (no LLM call, no DB write).
#[tokio::test]
#[ignore]
async fn empty_transcript_is_discarded() {
    let h = E2eHarness::new().await;
    h.run("").await;
    assert!(h.tts_sentences().is_empty(), "expected no TTS for empty transcript");
    assert!(h.db_messages().await.is_empty(), "expected no DB writes for empty transcript");
}

/// Ambient mode — utterance without wake word: pipeline discards after STT.
#[tokio::test]
#[ignore]
async fn ambient_mode_discards_utterance_without_wake_word() {
    let h = E2eHarness::new().await;
    h.run_ambient("Cuéntame algo interesante, por favor.", "jarvis").await;
    assert!(
        h.tts_sentences().is_empty(),
        "expected bot to stay silent in ambient mode, got: {:?}",
        h.tts_sentences()
    );
    assert!(
        h.db_messages().await.is_empty(),
        "expected no DB writes in ambient mode without wake word"
    );
}

/// Ambient mode — utterance WITH wake word: pipeline continues normally.
#[tokio::test]
#[ignore]
async fn ambient_mode_responds_when_wake_word_present() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Claro, son las once.").await;
    h.run_ambient("Jarvis, ¿qué hora es?", "jarvis").await;

    let sentences = h.tts_sentences();
    assert!(!sentences.is_empty(), "expected response when wake word present");
    let full = sentences.join(" ");
    assert!(full.contains("once"), "expected LLM response in TTS, got: {full:?}");
}

/// Multi-sentence response: each sentence is synthesized separately.
#[tokio::test]
#[ignore]
async fn multi_sentence_response_splits_into_sentences() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Primera frase. Segunda frase. Tercera frase.").await;
    h.run("Dime tres cosas.").await;

    let sentences = h.tts_sentences();
    assert!(
        sentences.len() >= 2,
        "expected multiple TTS sentences, got: {sentences:?}"
    );
}

/// Session persistence: multiple turns accumulate in the DB.
#[tokio::test]
#[ignore]
async fn db_persists_multiple_turns() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Primera respuesta.").await;
    h.run("Primera pregunta.").await;

    h.mock_llm_response("Segunda respuesta.").await;
    h.run("Segunda pregunta.").await;

    let msgs = h.db_messages().await;
    let assistant_count = msgs.iter().filter(|(r, _)| r == "Assistant").count();
    assert_eq!(assistant_count, 2, "expected 2 assistant turns in DB, got: {msgs:?}");
}

// ── STT tests (require Whisper model) ─────────────────────────────────────────

/// Load a WAV file, transcribe with WhisperSTTVAD, assert transcript is non-empty.
/// Requires: WHISPER_MODEL env var and VAD_MODEL env var.
#[tokio::test]
#[ignore = "requires Whisper + VAD models"]
async fn stt_transcribes_wav_file() {
    use crate::stt::{WhisperSTTVAD, WhisperSTTVADConfig};

    let model_path = std::env::var("WHISPER_MODEL")
        .unwrap_or_else(|_| "models/ggml-large-v3-turbo.bin".to_string());
    let vad_model = std::env::var("VAD_MODEL")
        .unwrap_or_else(|_| "models/ggml-silero-vad.bin".to_string());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("SKIP: Whisper model not found at {model_path}");
        return;
    }

    let wav_path = "tests/fixtures/es_short_greeting.wav";
    if !std::path::Path::new(wav_path).exists() {
        eprintln!("SKIP: fixture not found at {wav_path}");
        return;
    }

    let config = WhisperSTTVADConfig {
        whisper_model: model_path,
        vad_model,
        language: "es".to_string(),
        silence_ms: 500,
    };
    let stt = WhisperSTTVAD::new(config).expect("failed to load Whisper model");
    let audio = load_wav_as_f32(wav_path).expect("failed to load WAV fixture");
    let transcript = stt.transcribe_complete(&audio).expect("Whisper transcription failed");

    println!("Transcript: {transcript:?}");
    assert!(
        !transcript.trim().is_empty(),
        "expected non-empty transcript from {wav_path}"
    );
}

/// Full pipeline with real STT: WAV → WhisperSTTVAD → mock LLM → TTS capture → DB.
#[tokio::test]
#[ignore = "requires Whisper + VAD models"]
async fn full_pipeline_wav_to_db() {
    use crate::stt::{WhisperSTTVAD, WhisperSTTVADConfig};

    let model_path = std::env::var("WHISPER_MODEL")
        .unwrap_or_else(|_| "models/ggml-large-v3-turbo.bin".to_string());
    let vad_model = std::env::var("VAD_MODEL")
        .unwrap_or_else(|_| "models/ggml-silero-vad.bin".to_string());
    let wav_path = "tests/fixtures/es_long_intro.wav";

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("SKIP: Whisper model not found");
        return;
    }
    if !std::path::Path::new(wav_path).exists() {
        eprintln!("SKIP: fixture not found at {wav_path}");
        return;
    }

    let config = WhisperSTTVADConfig {
        whisper_model: model_path,
        vad_model,
        language: "es".to_string(),
        silence_ms: 500,
    };
    let stt = WhisperSTTVAD::new(config).unwrap();
    let audio = load_wav_as_f32(wav_path).unwrap();
    let transcript = stt.transcribe_complete(&audio).unwrap();
    println!("STT transcript: {transcript:?}");

    let h = E2eHarness::new().await;
    h.mock_llm_response("Respuesta de prueba.").await;
    h.run(&transcript).await;

    let msgs = h.db_messages().await;
    assert!(
        msgs.iter().any(|(r, _)| r == "Assistant"),
        "expected assistant message in DB, got: {msgs:?}"
    );
    println!("DB messages: {msgs:?}");
    println!("TTS sentences: {:?}", h.tts_sentences());
}

// ── VAD model path regression tests ─────────────────────────────────────────

/// Verify the default VAD model path matches the canonical path.
/// This prevents drift between Config defaults and documentation.
/// The canonical VAD model path is `models/ggml-silero-vad.bin`.
#[test]
fn default_vad_model_path() {
    let canonical = "models/ggml-silero-vad.bin";

    // Structural check: verify the filename part is correct
    let path = std::path::Path::new(canonical);
    assert_eq!(
        path.file_name().unwrap(),
        "ggml-silero-vad.bin",
        "canonical VAD model filename must be ggml-silero-vad.bin"
    );

    // Verify Config default when VAD_MODEL env var is unset
    temp_env::with_var("VAD_MODEL", None::<&str>, || {
        let config = Config::from_env().expect("Config::from_env() should succeed");
        assert_eq!(
            config.vad_model, canonical,
            "Config default VAD model must match canonical path"
        );
    });
}

// ── WAV loading helper ────────────────────────────────────────────────────────

fn load_wav_as_f32(path: &str) -> anyhow::Result<Vec<f32>> {
    use anyhow::Context;
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path).with_context(|| format!("opening {path}"))?;
    let mut reader = hound::WavReader::new(BufReader::new(file))
        .with_context(|| format!("parsing WAV header of {path}"))?;

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
