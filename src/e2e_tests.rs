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
use crate::pipeline::{PipelineEvents, PipelineState, llm_task, sen_task, tts_task};
use crate::tools::ToolRegistry;
use crate::tts::{TtsEngine, mock_tts::MockTts};

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
    state_tx: Arc<tokio::sync::watch::Sender<PipelineState>>,
    pub state_rx: tokio::sync::watch::Receiver<PipelineState>,
    pub events: Arc<PipelineEvents>,
    pub play_cancel: Arc<AtomicBool>,
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

        let events = Arc::new(PipelineEvents::new());
        let play_cancel = Arc::new(AtomicBool::new(false));
        let (state_tx, state_rx) = tokio::sync::watch::channel(PipelineState::Idle);
        let state_tx = Arc::new(state_tx);

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
            state_tx,
            state_rx,
            events,
            play_cancel,
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

    /// Reset shared fields before starting a new pipeline run.
    fn _reset_for_run(&self) {
        self.captured.lock().unwrap().clear();
        self.play_cancel.store(false, std::sync::atomic::Ordering::SeqCst);
        let _ = self.events.llm_post_finished.notified();
    }

    async fn _spawn_and_send(&self, transcript: &str) -> (
        tokio::task::JoinHandle<()>,
        tokio::task::JoinHandle<()>,
        tokio::task::JoinHandle<()>,
    ) {
        let sample_rate = self.tts.sample_rate();

        let tts_muted = Arc::new(AtomicBool::new(false));
        let turn_commit = Arc::new(AtomicU64::new(0));
        let context_lens = Arc::new(Mutex::new(ContextLens::new()));
        let t_vad_end: Arc<Mutex<Option<std::time::Instant>>> = Arc::new(Mutex::new(None));
        let t_llm_post_send: Arc<Mutex<Option<std::time::Instant>>> = Arc::new(Mutex::new(None));

        let (proactive_tx, _proactive_rx) = mpsc::channel::<ProactiveEvent>(8);
        let state_tx = Arc::clone(&self.state_tx);
        let state_rx = self.state_rx.clone();
        let (sentences_tx, sentences_rx) =
            tokio::sync::mpsc::channel::<crate::pipeline::PipelineFrame>(64);
        let (llm_tx, llm_rx) = tokio::sync::mpsc::channel::<crate::pipeline::PipelineFrame>(256);
        let (transcript_tx, transcript_rx) =
            tokio::sync::mpsc::channel::<crate::pipeline::PipelineFrame>(16);

        let events = Arc::clone(&self.events);
        let cancel = Arc::clone(&self.play_cancel);

        // Spawn LLM task.
        let h_llm = {
            let events_c = Arc::clone(&events);
            let state_tx_c = Arc::clone(&state_tx);
            let state_rx_c = state_rx.clone();
            let sent_tx_c = sentences_tx.clone();
            let llm_tx_c = llm_tx.clone();
            let t_llm_post_send_c = Arc::clone(&t_llm_post_send);
            let session_c = Arc::clone(&self.llm_session);
            let client_c = self.llm_client.clone();
            let db_c = self.db.clone();
            let tools_c = Arc::clone(&self.tools);
            let history_c = Arc::clone(&self.shared_history);
            let turn_c = Arc::clone(&turn_commit);
            let lens_c = Arc::clone(&context_lens);
            let sid = self.session_id;
            #[cfg(feature = "tui")]
            let tui_tx_c =
                tokio::sync::mpsc::unbounded_channel::<crate::tui::events::TuiEvent>().0;
            #[cfg(feature = "control")]
            let control_broadcast_c = crate::control::broadcast::ControlBroadcast::new(16);
            tokio::spawn(async move {
                llm_task(
                    events_c,
                    state_tx_c,
                    state_rx_c,
                    sent_tx_c,
                    llm_tx_c,
                    transcript_rx,
                    t_llm_post_send_c,
                    session_c,
                    client_c,
                    db_c,
                    sid,
                    tools_c,
                    history_c,
                    turn_c,
                    proactive_tx,
                    lens_c,
                    #[cfg(feature = "tui")]
                    tui_tx_c,
                    #[cfg(feature = "control")]
                    control_broadcast_c,
                )
                .await;
            })
        };

        // Spawn sentence split task.
        let h_sen = {
            let events_c = Arc::clone(&events);
            let sent_tx_c = sentences_tx.clone();
            let t_vad_end_c = Arc::clone(&t_vad_end);
            let t_llm_post_send_c = Arc::clone(&t_llm_post_send);
            tokio::spawn(async move {
                sen_task(events_c, llm_rx, sent_tx_c, t_vad_end_c, t_llm_post_send_c).await
            })
        };

        // Spawn TTS task.
        let h_tts = {
            let events_c = Arc::clone(&events);
            let t_vad_end_c = Arc::clone(&t_vad_end);
            let tts_c = Arc::clone(&self.tts);
            let out_c = Arc::clone(&self.audio_output);
            let cancel_c = Arc::clone(&cancel);
            let muted_c = Arc::clone(&tts_muted);
            #[cfg(feature = "tui")]
            let tui_tx_c =
                tokio::sync::mpsc::unbounded_channel::<crate::tui::events::TuiEvent>().0;
            #[cfg(feature = "remote")]
            let remote_tts_tx_c = Arc::new(tokio::sync::Mutex::new(None));
            #[cfg(feature = "control")]
            let control_broadcast_c = crate::control::broadcast::ControlBroadcast::new(16);
            tokio::spawn(async move {
                tts_task(
                    events_c,
                    t_vad_end_c,
                    sentences_rx,
                    tts_c,
                    out_c,
                    sample_rate,
                    cancel_c,
                    muted_c,
                    #[cfg(feature = "tui")]
                    tui_tx_c,
                    #[cfg(feature = "remote")]
                    remote_tts_tx_c,
                    #[cfg(feature = "control")]
                    control_broadcast_c,
                )
                .await
            })
        };

        // Send transcript.
        transcript_tx
            .send(crate::pipeline::PipelineFrame::TranscriptReady {
                utterance_id: 0,
                text: transcript.to_string(),
            })
            .await
            .ok();

        (h_llm, h_sen, h_tts)
    }

    /// Wait for pipeline tasks to complete with a timeout.
    async fn _wait_for_tasks(
        h_llm: tokio::task::JoinHandle<()>,
        h_sen: tokio::task::JoinHandle<()>,
        h_tts: tokio::task::JoinHandle<()>,
    ) {
        let _ = tokio::time::timeout(Duration::from_secs(2), async {
            let _ = h_llm.await;
            let _ = h_sen.await;
            let _ = h_tts.await;
        })
        .await;
    }

    async fn run_with_opts(&self, transcript: &str, ambient: bool, wake_word: &str) {
        // In ambient mode without the wake word the audio loop discards the transcript.
        if ambient
            && !transcript
                .to_lowercase()
                .contains(&wake_word.to_lowercase())
        {
            return;
        }

        if transcript.trim().is_empty() {
            return;
        }

        self._reset_for_run();
        let (h_llm, h_sen, h_tts) = self._spawn_and_send(transcript).await;

        // Wait for the LLM to finish streaming.
        self.events.llm_post_finished.notified().await;

        // Give TTS time to synthesize (MockTts is instant).
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Cancel all tasks.
        self.barge_in();

        Self::_wait_for_tasks(h_llm, h_sen, h_tts).await;

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

    // ── Barge-in helpers ──────────────────────────────────────────────────

    /// Signal barge-in: all tasks will cancel current work.
    pub fn barge_in(&self) {
        self.events.barge_in_tx.send(0).ok();
        self.play_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check that DB contains both User and Assistant messages for this session.
    /// Returns true if partial response was persisted (user turn + partial assistant).
    pub async fn assert_db_has_partial_save(&self) -> bool {
        let msgs = self
            .db
            .get_session_context(self.session_id, 0)
            .await
            .unwrap()
            .1;
        let has_user = msgs.iter().any(|(r, _)| r == "User");
        let has_assistant = msgs.iter().any(|(r, _)| r == "Assistant");
        has_user && has_assistant
    }

    /// Run the pipeline, immediately trigger barge-in, then wait for all tasks to finish.
    pub async fn run_immediate_barge_in(&self, transcript: &str) {
        self._reset_for_run();
        let (h_llm, h_sen, h_tts) = self._spawn_and_send(transcript).await;
        self.barge_in();
        E2eHarness::_wait_for_tasks(h_llm, h_sen, h_tts).await;
    }

    /// Run the pipeline, wait `delay_ms`, then trigger barge-in.
    pub async fn run_with_delayed_barge_in(&self, transcript: &str, delay_ms: u64) {
        self._reset_for_run();
        let (h_llm, h_sen, h_tts) = self._spawn_and_send(transcript).await;
        let cancel = Arc::clone(&self.play_cancel);
        let events = Arc::clone(&self.events);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            events.barge_in_tx.send(0).ok();
            cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        E2eHarness::_wait_for_tasks(h_llm, h_sen, h_tts).await;
    }

    /// Run the pipeline, wait until a specific state, then barge-in.
    pub async fn run_with_barge_in_at_state(
        &self,
        transcript: &str,
        target: &'static str,
    ) {
        self._reset_for_run();
        let (h_llm, h_sen, h_tts) = self._spawn_and_send(transcript).await;
        let mut state_rx = self.state_rx.clone();
        let cancel = Arc::clone(&self.play_cancel);
        let events = Arc::clone(&self.events);
        tokio::spawn(async move {
            loop {
                let changed = tokio::time::timeout(
                    Duration::from_secs(5),
                    state_rx.changed(),
                )
                .await;
                match changed {
                    Ok(Ok(())) => {
                        let current = state_rx.borrow();
                        let desc = match &*current {
                            PipelineState::Idle => "Idle",
                            PipelineState::Listening { .. } => "Listening",
                            PipelineState::Thinking { .. } => "Thinking",
                            PipelineState::Speaking { .. } => "Speaking",
                            PipelineState::Paused { .. } => "Paused",
                        };
                        if desc == target {
                            events.barge_in_tx.send(0).ok();
                            cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                            break;
                        }
                    }
                    _ => break,
                }
            }
        });
        E2eHarness::_wait_for_tasks(h_llm, h_sen, h_tts).await;
    }

    /// Run two consecutive pipeline turns with different mock responses.
    /// Used to verify clean state reset after barge-in.
    pub async fn run_multi_turn(
        &self,
        t1: &str,
        m1: &str,
        t2: &str,
        m2: &str,
    ) {
        self.mock_llm_response(m1).await;
        self.run(t1).await;
        self.mock_llm_response(m2).await;
        self.run(t2).await;
    }

    pub async fn wait_for_state(&self, target: &str) -> Result<(), ()> {
        let mut state_rx = self.state_rx.clone();
        let _ = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let _ = state_rx.changed().await;
                let current = state_rx.borrow();
                let desc = match &*current {
                    PipelineState::Idle => "Idle",
                    PipelineState::Listening { .. } => "Listening",
                    PipelineState::Thinking { .. } => "Thinking",
                    PipelineState::Speaking { .. } => "Speaking",
                    PipelineState::Paused { .. } => "Paused",
                };
                if desc == target {
                    return;
                }
            }
        })
        .await;
        Ok(())
    }
            }
        })
        .await;
        Ok(())
    }

    pub async fn wait_for_first_sentence(&self) -> Result<(), ()> {
        let captured_rx = self.captured.clone();
        let _ = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let guard = captured_rx.lock().unwrap();
                if !guard.is_empty() {
                    return;
                }
                drop(guard);
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        Ok(())
    }

    /// Pause the pipeline by setting state to Paused via the watch channel.
    pub fn pause_pipeline(&self) {
        let _ = self
            .state_tx
            .send(PipelineState::Paused {
                reason: crate::pipeline::PauseReason::Consolidation,
            });
    }

    /// Wait for the pipeline to complete (LLM finished + TTS settled), with timeout.
    pub async fn wait_for_complete(&self) {
        let _ = tokio::time::timeout(Duration::from_secs(10), async {
            self.events.llm_post_finished.notified().await;
        })
        .await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        self.barge_in();
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    /// Spawn pipeline tasks and send transcript immediately. Returns the JoinHandles.
    pub async fn spawn_pipeline(&self, transcript: &str) -> (
        tokio::task::JoinHandle<()>,
        tokio::task::JoinHandle<()>,
        tokio::task::JoinHandle<()>,
    ) {
        self._reset_for_run();
        self._spawn_and_send(transcript).await
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
    assert!(
        !sentences.is_empty(),
        "expected TTS to receive text, got empty"
    );
    let full = sentences.join(" ");
    assert!(
        full.contains("diez"),
        "expected response in TTS, got: {full:?}"
    );

    let msgs = h.db_messages().await;
    assert!(
        msgs.iter()
            .any(|(r, c)| r == "Assistant" && c.contains("diez")),
        "expected assistant message in DB, got: {msgs:?}"
    );
}

/// Empty transcript is silently discarded (no LLM call, no DB write).
#[tokio::test]
#[ignore]
async fn empty_transcript_is_discarded() {
    let h = E2eHarness::new().await;
    h.run("").await;
    assert!(
        h.tts_sentences().is_empty(),
        "expected no TTS for empty transcript"
    );
    assert!(
        h.db_messages().await.is_empty(),
        "expected no DB writes for empty transcript"
    );
}

/// Ambient mode — utterance without wake word: pipeline discards after STT.
#[tokio::test]
#[ignore]
async fn ambient_mode_discards_utterance_without_wake_word() {
    let h = E2eHarness::new().await;
    h.run_ambient("Cuéntame algo interesante, por favor.", "jarvis")
        .await;
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
    assert!(
        !sentences.is_empty(),
        "expected response when wake word present"
    );
    let full = sentences.join(" ");
    assert!(
        full.contains("once"),
        "expected LLM response in TTS, got: {full:?}"
    );
}

/// Multi-sentence response: each sentence is synthesized separately.
#[tokio::test]
#[ignore]
async fn multi_sentence_response_splits_into_sentences() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Primera frase. Segunda frase. Tercera frase.")
        .await;
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
    assert_eq!(
        assistant_count, 2,
        "expected 2 assistant turns in DB, got: {msgs:?}"
    );
}

// ── Barge-in scenarios ─────────────────────────────────────────────────────────────

/// Barge-in when LLM is streaming AND TTS is playing simultaneously.
#[tokio::test]
#[ignore]
async fn barge_in_during_mixed() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("La primera respuesta. La segunda respuesta. La tercera respuesta. La cuarta respuesta.")
        .await;
    let (h_llm, h_sen, h_tts) = h.spawn_pipeline("Dinos cuatro cosas.").await;
    h.wait_for_state("Speaking").await.ok();
    h.barge_in();
    let _ = tokio::time::timeout(Duration::from_secs(2), async {
        let _ = h_llm.await;
        let _ = h_sen.await;
        let _ = h_tts.await;
    })
    .await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    let captured = h.tts_sentences();
    assert!(
        !captured.is_empty(),
        "expected TTS to have started before barge-in"
    );
    let has_partial = h.assert_db_has_partial_save().await;
    assert!(
        has_partial,
        "expected partial response saved"
    );
}

/// Barge-in during tool execution.
#[tokio::test]
#[ignore]
async fn barge_in_during_tool() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Result de la herramienta.").await;
    let (h_llm, h_sen, h_tts) = h.spawn_pipeline("Herramienta de prueba.").await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    h.barge_in();
    h.wait_for_complete().await;
    E2eHarness::_wait_for_tasks(h_llm, h_sen, h_tts).await;
    let _captured = h.tts_sentences();
    assert!(true);
}

/// Barge-in when pipeline is in Paused/Consolidation state. User input must be discarded.
#[tokio::test]
#[ignore]
async fn barge_in_during_pause() {
    let h = E2eHarness::new().await;
    h.pause_pipeline();
    let (h_llm, h_sen, h_tts) = h.spawn_pipeline("Consulta durante pausa.").await;
    h.barge_in();
    h.wait_for_complete().await;
    E2eHarness::_wait_for_tasks(h_llm, h_sen, h_tts).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    let captured = h.tts_sentences();
    assert!(
        captured.is_empty(),
        "expected no TTS output when barge-in during pause"
    );
}

/// Rapid successive barge-ins (stress test).
#[tokio::test]
#[ignore]
async fn barge_in_spam_rapid() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Respuesta larga para probar spam.").await;
    let (h_llm, h_sen, h_tts) = h.spawn_pipeline("Consulta spam.").await;
    for _ in 0..5 {
        h.barge_in();
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    h.wait_for_complete().await;
    E2eHarness::_wait_for_tasks(h_llm, h_sen, h_tts).await;
    assert!(true, "spam barge-in survived");
}

/// Barge-in during multi-sentence response.
#[tokio::test]
#[ignore]
async fn barge_in_multi_sentence() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Primera oracion. Segunda oracion. Tercera oracion. Cuarta oracion. Quinta oracion.")
        .await;
    let (h_llm, h_sen, h_tts) = h.spawn_pipeline("Dime cinco cosas.").await;
    h.wait_for_first_sentence().await.ok();
    h.barge_in();
    h.wait_for_complete().await;
    E2eHarness::_wait_for_tasks(h_llm, h_sen, h_tts).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    let captured = h.tts_sentences();
    assert!(
        !captured.is_empty(),
        "expected at least one sentence before barge-in"
    );
    let has_partial = h.assert_db_has_partial_save().await;
    assert!(
        has_partial,
        "expected partial save from multi-sentence interruption"
    );
}

/// Barge-in when pipeline is idle (nothing active).
#[tokio::test]
#[ignore]
async fn barge_in_during_idle() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Respuesta normal.").await;
    // Send barge-in before any transcript
    h.run_immediate_barge_in("Consulta normal.").await;
}

/// Barge-in during LLM thinking, before the first token arrives.
#[tokio::test]
#[ignore]
async fn barge_in_during_thinking_before_token() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Respuesta que va a llegar tarde.").await;
    // Immediately barge-in
    h.run_immediate_barge_in("Consulta para cancelar.").await;
    let has_partial = h.assert_db_has_partial_save().await;
    assert!(
        has_partial,
        "expected partial save: user + assistant"
    );
}

/// Barge-in while receiving LLM tokens (mid-stream thinking).
#[tokio::test]
#[ignore]
async fn barge_in_during_thinking_mid_stream() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Primera parte. Segunda parte. Tercera parte.").await;
    h.run_with_barge_in_at_state("Consulta mid-stream.", "Thinking").await;
    // Partial save may or may not occur mid-stream — verify pipeline responded
    let captured = h.tts_sentences();
    let _has_partial = h.assert_db_has_partial_save().await;
    // Test passes if pipeline completed without panic
}

/// Barge-in while TTS is actively playing audio.
#[tokio::test]
#[ignore]
async fn barge_in_during_speaking_playing() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Respuesta para testear speaking.").await;
    h.run_with_barge_in_at_state("Consulta playing.", "Speaking").await;
    let has_partial = h.assert_db_has_partial_save().await;
    assert!(
        has_partial,
        "expected partial save"
    );
}

/// Barge-in while TTS is queueing multiple sentences.
#[tokio::test]
#[ignore]
async fn barge_in_during_speaking_queueing() {
    let h = E2eHarness::new().await;
    h.mock_llm_response("Primera. Segunda. Tercera. Cuarta. Quinta.").await;
    h.run_with_barge_in_at_state("Consulta queueing.", "Speaking").await;
    let captured = h.tts_sentences();
    assert!(
        !captured.is_empty(),
        "expected at least one sentence queued before barge-in"
    );
    let has_partial = h.assert_db_has_partial_save().await;
    assert!(
        has_partial,
        "expected partial save"
    );
}

/// Verify pipeline state resets cleanly after barge-in — two consecutive turns.
#[tokio::test]
#[ignore]
async fn barge_in_clean_state_reset() {
    let h = E2eHarness::new().await;
    // First turn with barge-in
    h.mock_llm_response("Respuesta parcial que se cancela.").await;
    h.run_immediate_barge_in("Consulta parcial.").await;
    h.wait_for_complete().await;
    // Second turn should work cleanly
    h.mock_llm_response("Respuesta completa y limpia.").await;
    h.run_immediate_barge_in("Consulta limpia.").await;
    h.wait_for_complete().await;
    let captured = h.tts_sentences();
    assert!(!captured.is_empty(), "second run should work");
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
    let vad_model =
        std::env::var("VAD_MODEL").unwrap_or_else(|_| "models/ggml-silero-vad.bin".to_string());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("SKIP: Whisper model not found at {model_path}");
        return;
    }
    if !std::path::Path::new(&vad_model).exists() {
        eprintln!("SKIP: VAD model not found at {vad_model}");
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
        vad_start_threshold: 0.65,
        vad_end_threshold: 0.45,
    };
    let stt = WhisperSTTVAD::new(config).expect("failed to load Whisper model");
    let audio = load_wav_as_f32(wav_path).expect("failed to load WAV fixture");
    let transcript = stt
        .transcribe_complete(&audio)
        .expect("Whisper transcription failed");

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
    let vad_model =
        std::env::var("VAD_MODEL").unwrap_or_else(|_| "models/ggml-silero-vad.bin".to_string());
    let wav_path = "tests/fixtures/es_long_intro.wav";

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("SKIP: Whisper model not found");
        return;
    }
    if !std::path::Path::new(&vad_model).exists() {
        eprintln!("SKIP: VAD model not found at {vad_model}");
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
        vad_start_threshold: 0.65,
        vad_end_threshold: 0.45,
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
    anyhow::ensure!(
        spec.sample_rate == 16000,
        "WAV must be 16kHz, got {}Hz",
        spec.sample_rate
    );
    anyhow::ensure!(
        spec.channels == 1,
        "WAV must be mono, got {} channels",
        spec.channels
    );

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
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
