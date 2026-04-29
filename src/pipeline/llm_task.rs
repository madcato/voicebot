use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use tokio::sync::{mpsc, watch};
use tracing::{error, info, warn};

use crate::agents::ProactiveEvent;
use crate::analysis::ContextLens;
use crate::db::Database;
use crate::llm::{LlmSession, OpenAIClient, StreamToken};
use crate::tools::{format_history, ToolRegistry};
use super::frames::PipelineFrame;
use super::fsm::PipelineState;
use super::state::PipelineEvents;

/// Monotonically increasing counter for tagging each pipeline run with a unique ID.
static PIPELINE_RUN_ID: AtomicU64 = AtomicU64::new(0);

/// Maximum number of sequential tool calls allowed per user turn.
pub const MAX_TOOL_ITERATIONS: usize = 5;

/// LLM task: receives transcript frames, runs the LLM+tools pipeline, fires events.
#[allow(clippy::too_many_arguments)]
pub async fn llm_task(
    events: Arc<PipelineEvents>,
    pipeline_state_tx: Arc<watch::Sender<PipelineState>>,
    mut pipeline_state_rx: watch::Receiver<PipelineState>,
    sentences_tx: mpsc::Sender<PipelineFrame>,
    llm_tx: mpsc::Sender<PipelineFrame>,
    mut transcript_rx: mpsc::Receiver<PipelineFrame>,
    t_llm_post_send: Arc<Mutex<Option<Instant>>>,
    llm_session: Arc<Mutex<LlmSession>>,
    llm_client: OpenAIClient,
    db: Database,
    session_id: uuid::Uuid,
    tools: Arc<ToolRegistry>,
    shared_history: Arc<RwLock<String>>,
    turn_commit_counter: Arc<AtomicU64>,
    proactive_tx: mpsc::Sender<ProactiveEvent>,
    context_lens: Arc<Mutex<ContextLens>>,
    #[cfg(feature = "tui")] tui_tx: crate::tui::events::TuiEventTx,
) {
    let pipeline_id = PIPELINE_RUN_ID.fetch_add(1, Ordering::SeqCst);
    let mut cancel_rx = events.barge_in_tx.subscribe();

    loop {
        // Block until a transcript frame arrives; ignore cancels while idle.
        let frame = loop {
            tokio::select! {
                frame = transcript_rx.recv() => {
                    match frame {
                        Some(f) => break f,
                        None => return, // channel closed — exit
                    }
                }
                _ = cancel_rx.recv() => {}
            }
        };

        // Decode the incoming frame into (text, tool_continuation, is_text_input).
        let (text, tool_continuation, is_text_input) = match frame {
            PipelineFrame::TranscriptReady { text, .. } => (text, false, false),
            PipelineFrame::TextInput { text }           => (text, false, true),
            PipelineFrame::SystemNotification { text }  => (text, false, false),
            PipelineFrame::AgentResult { tool_call_id: Some(_), .. } => (String::new(), true, false),
            _ => continue, // unexpected frame type — wait for next
        };

        // Wait for consolidation to finish before starting a new turn.
        loop {
            if !matches!(*pipeline_state_rx.borrow(), PipelineState::Paused { .. }) { break; }
            pipeline_state_rx.changed().await.ok();
        }

        let _ = pipeline_state_tx.send(PipelineState::Thinking { utterance_id: pipeline_id });

        if tool_continuation {
            info!(target: "pipeline", "[pipe={}] Tool result delivered — continuing turn", pipeline_id);
        } else {
            info!(target: "pipeline", "[pipe={}] User: {}", pipeline_id, text);
        }

        #[cfg(feature = "tui")]
        if !tool_continuation {
            let source = if is_text_input {
                crate::tui::events::InputSource::Text
            } else {
                crate::tui::events::InputSource::Voice
            };
            tui_tx
                .send(crate::tui::events::TuiEvent::UserMessage { text: text.clone(), source })
                .ok();
            tui_tx
                .send(crate::tui::events::TuiEvent::StateChange(
                    crate::tui::events::PipelineState::Thinking,
                ))
                .ok();
        }

        let messages_snapshot = llm_session.lock().unwrap().messages.clone();

        if !tool_continuation {
            {
                let mut s = llm_session.lock().unwrap();
                s.add_user_turn(&text);
                turn_commit_counter.fetch_add(1, Ordering::SeqCst);
                if let Ok(mut h) = shared_history.write() {
                    *h = format_history(&s.messages);
                }
            }
            {
                let db_c = db.clone();
                let text_c = text.clone();
                tokio::spawn(async move {
                    if let Err(e) = db_c.save_message(session_id, "User", &text_c).await {
                        warn!(target: "db", "Failed to save User message: {}", e);
                    }
                });
            }
        } else {
            if let Ok(mut h) = shared_history.write() {
                let s = llm_session.lock().unwrap();
                *h = format_history(&s.messages);
            }
            turn_commit_counter.fetch_add(1, Ordering::SeqCst);
        }

        // (assistant_text / llm_post_finished flags removed; channel carries this now)

        let tool_defs = tools.tool_definitions();
        info!(
            target: "pipeline",
            "LLM request: {} tool(s) available: {:?}",
            tool_defs.len(),
            tool_defs
                .iter()
                .filter_map(|t| t["function"]["name"].as_str())
                .collect::<Vec<_>>()
        );
        let mut messages = llm_session.lock().unwrap().all_messages_api();
        // Inject fresh analysis context (speaker identity, emotion, video scene) into the
        // system message for this call only — never persisted to the session or DB.
        if let Some(ctx) = context_lens.lock().unwrap().format_for_llm() {
            if let Some(sys_msg) = messages.first_mut() {
                if let Some(content) = sys_msg["content"].as_str() {
                    let enriched = format!("{}{}", content, ctx);
                    sys_msg["content"] = serde_json::Value::String(enriched);
                }
            }
        }
        let base_msg_len = messages.len();
        let mut final_response = String::new();
        let mut committed = false;
        let mut cancelled = false;
        let mut first_token_logged = false;

        'pipeline: {
            'tool_loop: for iter in 0..MAX_TOOL_ITERATIONS {
                info!(target: "performance", "LLM request [pipe={}]", pipeline_id);
                let (token_rx, stream_handle) =
                    match llm_client.stream(&messages, &tool_defs).await {
                        Ok(r) => r,
                        Err(e) => {
                            error!(target: "llm", "LLM error: {}", e);
                            #[cfg(feature = "tui")]
                            tui_tx
                                .send(crate::tui::events::TuiEvent::Error(format!(
                                    "LLM error: {e}"
                                )))
                                .ok();
                            let _ = sentences_tx.send(super::frames::PipelineFrame::SentenceReady {
                                utterance_id: pipeline_id,
                                sentence: "Lo siento, no pude conectar con el modelo de lenguaje.".to_string(),
                            }).await;
                            break 'pipeline;
                        }
                    };

                *t_llm_post_send.lock().unwrap() = Some(Instant::now());

                let mut token_rx = token_rx;
                let mut llm_text = String::new();
                let mut tool_call: Option<(String, String)> = None;

                loop {
                    tokio::select! {
                        token = token_rx.recv() => {
                            match token {
                                Some(StreamToken::Content(t)) => {
                                    let t = if llm_text.is_empty() {
                                        t.trim_start_matches('\n').to_string()
                                    } else {
                                        t
                                    };
                                    if t.is_empty() { continue; }
                                    if !first_token_logged {
                                        first_token_logged = true;
                                        if let Some(t0) = t_llm_post_send.lock().unwrap().as_ref() {
                                            info!(target: "performance", "[+{}ms] LLM first token (TTFT)", t0.elapsed().as_millis());
                                        }
                                    }
                                    llm_text.push_str(&t);
                                    let _ = llm_tx.send(super::frames::PipelineFrame::LLMToken {
                                        utterance_id: pipeline_id,
                                        token: t.clone(),
                                    }).await;
                                    #[cfg(feature = "tui")]
                                    tui_tx.send(crate::tui::events::TuiEvent::AssistantToken(t)).ok();
                                }
                                Some(StreamToken::ToolCall { name, args }) => {
                                    info!(target: "pipeline", "ToolCall received: name={} args={}", name, args);
                                    tool_call = Some((name, args));
                                    break;
                                }
                                None => {
                                    let _ = llm_tx.send(super::frames::PipelineFrame::LLMResponseDone {
                                        utterance_id: pipeline_id,
                                        full_text: llm_text.clone(),
                                    }).await;
                                    events.llm_post_finished.notify_one();
                                    #[cfg(feature = "tui")]
                                    tui_tx.send(crate::tui::events::TuiEvent::AssistantDone).ok();
                                    break;
                                }
                            }
                        }
                        _ = cancel_rx.recv() => {
                            cancelled = true;
                            drop(token_rx);
                            stream_handle.abort();
                            break;
                        }
                    }
                }

                if cancelled {
                    break 'pipeline;
                }

                match tool_call {
                    Some((name, args)) => {
                        if tools.is_background(&name) {
                            let ack = match name.as_str() {
                                "web_search" => "Buscando.",
                                "run_shell" => "Ejecutando.",
                                _ => "Procesando en segundo plano, le aviso al terminar.",
                            };
                            let _ = llm_tx.send(super::frames::PipelineFrame::LLMToken {
                                utterance_id: pipeline_id,
                                token: ack.to_string(),
                            }).await;
                            let _ = llm_tx.send(super::frames::PipelineFrame::LLMResponseDone {
                                utterance_id: pipeline_id,
                                full_text: ack.to_string(),
                            }).await;
                            events.llm_post_finished.notify_one();

                            let tc_id = format!("bg_{}_{}_{}", pipeline_id, iter, name);
                            let tool_call_msg = serde_json::json!({
                                "role": "assistant",
                                "content": serde_json::Value::Null,
                                "tool_calls": [{
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {"name": &name, "arguments": &args}
                                }]
                            });
                            messages.push(tool_call_msg);

                            {
                                let tool_exchanges = messages[base_msg_len..].to_vec();
                                {
                                    let mut s = llm_session.lock().unwrap();
                                    s.add_tool_exchange(tool_exchanges.clone());
                                    if let Ok(mut h) = shared_history.write() {
                                        *h = format_history(&s.messages);
                                    }
                                }
                                let db_c = db.clone();
                                tokio::spawn(async move {
                                    if let Err(e) =
                                        db_c.save_tool_exchanges(session_id, &tool_exchanges).await
                                    {
                                        warn!(target: "db", "Failed to save tool_call exchange: {}", e);
                                    }
                                });
                            }

                            let tools_c = Arc::clone(&tools);
                            let name_c = name.clone();
                            let args_c = args.clone();
                            let proactive_c = proactive_tx.clone();
                            let tc_id_c = tc_id.clone();
                            tokio::spawn(async move {
                                info!(target: "pipeline", "Background tool `{}` started", name_c);
                                let result = tools_c.execute(&name_c, &args_c).await;
                                info!(
                                    target: "pipeline",
                                    "Background tool `{}` finished ({} chars): {:?}",
                                    name_c, result.len(), result
                                );
                                proactive_c
                                    .send(ProactiveEvent::AgentResult {
                                        task: name_c,
                                        result,
                                        tool_call_id: Some(tc_id_c),
                                    })
                                    .await
                                    .ok();
                            });

                            committed = true;
                            break 'pipeline;
                        }

                        let result = tools.execute(&name, &args).await;
                        info!(target: "pipeline", "Tool[{}] `{}` → {}", iter, name, result);
                        #[cfg(feature = "tui")]
                        tui_tx
                            .send(crate::tui::events::TuiEvent::ToolCall {
                                name: name.clone(),
                                result: result.clone(),
                            })
                            .ok();

                        let tool_call_id = format!("call_{}_{}", name, iter);
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": serde_json::Value::Null,
                            "tool_calls": [{
                                "id": tool_call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": args}
                            }]
                        }));
                        messages.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result
                        }));

                        if cancel_rx.try_recv().is_ok() {
                            cancelled = true;
                            break 'pipeline;
                        }
                    }
                    None => {
                        final_response = llm_text;
                        break 'tool_loop;
                    }
                }
            }

            if final_response.is_empty() || cancelled {
                break 'pipeline;
            }

            info!(
                target: "pipeline",
                "[pipe={}] Assistant: {}",
                pipeline_id, final_response
            );

            {
                let db_c = db.clone();
                let resp_c = final_response.clone();
                let tool_exchanges_c = messages[base_msg_len..].to_vec();
                tokio::spawn(async move {
                    if !tool_exchanges_c.is_empty() {
                        if let Err(e) =
                            db_c.save_tool_exchanges(session_id, &tool_exchanges_c).await
                        {
                            warn!(target: "db", "Failed to save tool exchanges: {}", e);
                        }
                    }
                    if let Err(e) = db_c.save_message(session_id, "Assistant", &resp_c).await {
                        warn!(target: "db", "Failed to save assistant message: {}", e);
                    }
                });
            }
            {
                let mut s = llm_session.lock().unwrap();
                let tool_exchanges = messages[base_msg_len..].to_vec();
                if !tool_exchanges.is_empty() {
                    s.add_tool_exchange(tool_exchanges);
                }
                s.add_assistant_turn(&final_response);
            }
            committed = true;
        }

        if !committed && cancelled {
            llm_session.lock().unwrap().messages = messages_snapshot;
            info!(
                target: "pipeline",
                "[pipe={}] Cancelled — session rolled back",
                pipeline_id
            );
        }

        let _ = pipeline_state_tx.send(PipelineState::Idle);
        #[cfg(feature = "tui")]
        tui_tx
            .send(crate::tui::events::TuiEvent::StateChange(
                crate::tui::events::PipelineState::Idle,
            ))
            .ok();

        while cancel_rx.try_recv().is_ok() {}
    }
}
