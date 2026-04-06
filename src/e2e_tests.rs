//! End-to-end integration tests for the voicebot pipeline.
//!
//! These tests exercise the full STT → LLM → TTS → DB pipeline using:
//! - `SttStream::mock()` — injects a known transcript, no Whisper needed
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
use std::time::Instant;

use uuid::Uuid;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::audio::output::AudioOutput;
use crate::db::Database;
use crate::llm::{OpenAIClient, LlmSession};
use crate::stt::SttStream;
use crate::tools::{ToolRegistry, ConversationMode};
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

/// SSE body for a native function-call tool invocation.
fn make_sse_tool_call(tool_name: &str, args: &str) -> String {
    let escaped_args = args.replace('"', "\\\"");
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"call_0\",\
         \"type\":\"function\",\"function\":{{\"name\":\"{tool_name}\",\"arguments\":\"\"}}}}]}},\
         \"finish_reason\":null}}]}}\n\n\
         data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"function\":\
         {{\"arguments\":\"{escaped_args}\"}}}}]}},\"finish_reason\":null}}]}}\n\n\
         data: [DONE]\n\n"
    )
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
            400,  // max_tokens
            0.0,  // temperature — deterministic
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

    /// Mount a mock that returns `text` as a streaming SSE response for every
    /// POST to /v1/chat/completions (covers the main stream + speculative prefill).
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
        let stt_stream = SttStream::mock(transcript.to_string());
        let cancel = Arc::new(AtomicBool::new(false));
        let sample_rate = self.tts.sample_rate();
        let conv_mode = Arc::new(Mutex::new(ConversationMode::Active));

        super::run_pipeline(
            1, // min_stt_gen — matches SttStream::mock seq
            stt_stream,
            cancel,
            Arc::clone(&self.tts),
            Arc::clone(&self.audio_output),
            Arc::clone(&self.llm_session),
            self.llm_client.clone(),
            self.db.clone(),
            self.session_id,
            sample_rate,
            Arc::clone(&self.tools),
            Arc::clone(&self.shared_history),
            4096,  // context_tokens — high enough that summarization never triggers
            6,     // summary_keep_turns
            Instant::now(),
            ambient,
            conv_mode,
            wake_word.to_string(),
            Arc::new(AtomicU64::new(0)),
        )
        .await;

        // Allow background DB writes (user message save) to settle.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
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
async fn basic_conversation_mocked_transcript() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Son las diez y media.").await;

    h.run("Hola, ¿qué hora es?").await;

    // TTS received the response
    let sentences = h.tts_sentences();
    assert!(!sentences.is_empty(), "expected TTS to receive text, got empty");
    let full = sentences.join(" ");
    assert!(
        full.contains("diez"),
        "expected response in TTS, got: {full:?}"
    );

    // DB has the assistant response (saved synchronously inside run_pipeline)
    let msgs = h.db_messages().await;
    assert!(
        msgs.iter().any(|(r, c)| r == "Assistant" && c.contains("diez")),
        "expected assistant message in DB, got: {msgs:?}"
    );
}

/// Barge-in: empty transcript is silently discarded (no LLM call, no DB write).
#[tokio::test]
async fn empty_transcript_is_discarded() {
    let h = E2eHarness::new().await;
    // No mock mounted — any HTTP call would cause a connection error, making the
    // test fail if the pipeline incorrectly calls the LLM.

    h.run("").await;

    assert!(h.tts_sentences().is_empty(), "expected no TTS for empty transcript");
    assert!(h.db_messages().await.is_empty(), "expected no DB writes for empty transcript");
}

/// Ambient mode — utterance without wake word: pipeline discards after STT.
#[tokio::test]
async fn ambient_mode_discards_utterance_without_wake_word() {
    let h = E2eHarness::new().await;
    // No mock needed — the pipeline should return before reaching the LLM.

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
async fn ambient_mode_responds_when_wake_word_present() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Claro, son las once.").await;

    h.run_ambient("Jarvis, ¿qué hora es?", "jarvis").await;

    let sentences = h.tts_sentences();
    assert!(!sentences.is_empty(), "expected response when wake word present");
    let full = sentences.join(" ");
    assert!(
        full.contains("once"),
        "expected LLM response in TTS, got: {full:?}"
    );
}

/// Multi-sentence response: each sentence is synthesized separately.
#[tokio::test]
async fn multi_sentence_response_splits_into_sentences() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Primera frase. Segunda frase. Tercera frase.").await;

    h.run("Dime tres cosas.").await;

    let sentences = h.tts_sentences();
    // SentenceSplitter should emit at least 2 sentences (punctuation-delimited)
    assert!(
        sentences.len() >= 2,
        "expected multiple TTS sentences, got: {sentences:?}"
    );
}

/// Session persistence: multiple turns accumulate in the DB.
#[tokio::test]
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

// ── STT tests (require Whisper model) ────────────────────────────────────────
//
// These tests load a WAV file from tests/fixtures/ and run real Whisper.
// See e2e.md for how to record fixture files.

/// Load a WAV file, transcribe with Whisper, assert transcript is non-empty.
/// Requires: WHISPER_MODEL env var pointing to a GGML model file.
/// In CI this is set to models/ggml-base.bin (downloaded by the CI step).
#[tokio::test]
#[ignore = "requires Whisper model (set WHISPER_MODEL env var)"]
async fn stt_transcribes_wav_file() {
    let model_path = std::env::var("WHISPER_MODEL")
        .unwrap_or_else(|_| "models/ggml-large-v3-turbo.bin".to_string());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("SKIP: Whisper model not found at {model_path}");
        return;
    }

    let wav_path = "tests/fixtures/es_short_greeting.wav";
    if !std::path::Path::new(wav_path).exists() {
        eprintln!("SKIP: fixture not found at {wav_path}");
        return;
    }

    let stt = crate::stt::WhisperStt::new(&model_path, "es", 0)
        .expect("failed to load Whisper model");

    let audio = load_wav_as_f32(wav_path).expect("failed to load WAV fixture");
    let transcript = tokio::task::spawn_blocking(move || stt.transcribe(&audio))
        .await
        .unwrap()
        .expect("Whisper transcription failed");

    println!("Transcript: {transcript:?}");
    assert!(
        !transcript.trim().is_empty(),
        "expected non-empty transcript from {wav_path}"
    );
}

/// Full pipeline with real STT: WAV → Whisper → mock LLM → TTS capture → DB.
/// Requires: WHISPER_MODEL env var pointing to a GGML model file.
/// In CI this is set to models/ggml-base.bin (downloaded by the CI step).
#[tokio::test]
// #[ignore = "requires Whisper model (set WHISPER_MODEL env var)"]
async fn full_pipeline_wav_to_db() {
    let model_path = std::env::var("WHISPER_MODEL")
        .unwrap_or_else(|_| "models/ggml-large-v3-turbo.bin".to_string());
    let wav_path = "tests/fixtures/es_long_intro.wav";

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("SKIP: Whisper model not found");
        return;
    }
    if !std::path::Path::new(wav_path).exists() {
        eprintln!("SKIP: fixture not found at {wav_path}");
        return;
    }

    // Run real STT first to get the transcript
    let stt = crate::stt::WhisperStt::new(&model_path, "es", 0).unwrap();
    let audio = load_wav_as_f32(wav_path).unwrap();
    let transcript = tokio::task::spawn_blocking(move || stt.transcribe(&audio))
        .await
        .unwrap()
        .unwrap();
    println!("STT transcript: {transcript:?}");

    // Now feed the transcript into the pipeline with a mock LLM
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

// ── WAV loading helper ────────────────────────────────────────────────────────

/// Load a 16kHz mono WAV file as f32 samples in [-1.0, 1.0].
/// Whisper requires exactly 16kHz mono f32 input.
fn load_wav_as_f32(path: &str) -> anyhow::Result<Vec<f32>> {
    use anyhow::Context;
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path).with_context(|| format!("opening {path}"))?;
    let mut reader = hound::WavReader::new(BufReader::new(file))
        .with_context(|| format!("parsing WAV header of {path}"))?;

    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_rate == 16000,
        "WAV must be 16kHz, got {}Hz — see e2e.md for conversion command",
        spec.sample_rate
    );
    anyhow::ensure!(
        spec.channels == 1,
        "WAV must be mono, got {} channels — see e2e.md for conversion command",
        spec.channels
    );

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().map(|s| s.unwrap()).collect()
        }
        hound::SampleFormat::Int => {
            let max = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max)
                .collect()
        }
    };

    Ok(samples)
}
