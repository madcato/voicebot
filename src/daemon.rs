/// Inference daemon — Jarvis's background "is there anything worth saying?" loop.
///
/// Every `interval_secs` this daemon asks the LLM if there is something
/// genuinely worth telling the user, and — if the answer is not `NOTHING` —
/// pushes a `ProactiveEvent::InferenceDaemon` to the proactive channel so
/// `run_proactive_pipeline` can vocalize it.
///
/// The LLM call uses `complete_short()` (no slot, no KV-cache eviction).
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::agents::ProactiveEvent;
use crate::llm::{LlmSession, Message, OpenAIClient};

/// Sentinel the LLM must return when it decides there is nothing to say.
const NOTHING: &str = "NOTHING";

pub struct InferenceDaemon {
    pub interval_secs: u64,
    pub llm_client: OpenAIClient,
    pub llm_session: std::sync::Arc<std::sync::Mutex<LlmSession>>,
    pub proactive_tx: mpsc::Sender<ProactiveEvent>,
}

impl InferenceDaemon {
    /// Spawns the daemon as a background tokio task. Returns immediately.
    pub fn spawn(self) {
        tokio::spawn(async move {
            self.run().await;
        });
    }

    async fn run(self) {
        info!(
            target: "daemon",
            "Inference daemon started (interval={}s)",
            self.interval_secs
        );

        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(self.interval_secs)).await;

            // Don't bother if the proactive channel is already full — the user
            // is probably busy processing a previous event.
            if self.proactive_tx.capacity() == 0 {
                debug!(target: "daemon", "Inference daemon: proactive channel full, skipping tick");
                continue;
            }

            debug!(target: "daemon", "Inference daemon tick");

            let system_prompt = {
                let s = self.llm_session.lock().unwrap();
                s.system_prompt().to_string()
            };

            let messages = vec![
                Message {
                    role: "system".to_string(),
                    content: build_daemon_system_prompt(&system_prompt),
                },
                Message {
                    role: "user".to_string(),
                    content: "check".to_string(),
                },
            ];

            match self.llm_client.complete_short(&messages).await {
                Ok(response) => {
                    let trimmed = response.trim();
                    if trimmed.is_empty() || trimmed.to_uppercase().starts_with(NOTHING) {
                        debug!(target: "daemon", "Inference daemon: nothing to say");
                    } else {
                        info!(target: "daemon", "Inference daemon: proactive message → {:?}", trimmed);
                        let event = ProactiveEvent::InferenceDaemon {
                            message: trimmed.to_string(),
                        };
                        if let Err(e) = self.proactive_tx.try_send(event) {
                            warn!(target: "daemon", "Inference daemon: failed to send proactive event: {}", e);
                        }
                    }
                }
                Err(e) => {
                    warn!(target: "daemon", "Inference daemon LLM call failed: {}", e);
                }
            }
        }
    }
}

/// Builds the system prompt sent to the LLM for the daemon check.
///
/// Deliberately high threshold — the daemon should only interrupt when
/// something is genuinely worth the user's attention, not on routine state.
fn build_daemon_system_prompt(assistant_system_prompt: &str) -> String {
    format!(
        "{assistant_system_prompt}\n\n\
         ---\n\
         MODO: demonio de inferencia proactiva.\n\
         Tu trabajo es decidir si hay algo genuinamente importante que \
         comunicar al usuario ahora mismo, basándote en el contexto de la \
         conversación.\n\n\
         REGLAS ESTRICTAS:\n\
         - Si no hay nada importante, responde exactamente: NOTHING\n\
         - Solo interviene si tienes algo urgente o claramente útil que decir\n\
         - NO interrumpas sin motivo claro\n\
         - Si decides intervenir, escribe solo el mensaje a pronunciar (1-2 frases \
           naturales, sin saludos, sin markdown)."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_sentinel_is_recognized_case_insensitive() {
        for input in &["NOTHING", "Nothing", "nothing", "NOTHING.", " NOTHING "] {
            assert!(
                input.trim().to_uppercase().starts_with(NOTHING),
                "should recognize {input:?} as NOTHING"
            );
        }
    }

    #[test]
    fn real_message_is_not_nothing() {
        let msg = "La batería está al 8% y no está cargando.";
        assert!(!msg.trim().to_uppercase().starts_with(NOTHING));
    }

    #[test]
    fn build_daemon_system_prompt_includes_base_prompt() {
        let base = "Eres Jarvis, el asistente.";
        let result = build_daemon_system_prompt(base);
        assert!(result.contains(base));
        assert!(result.contains("NOTHING"));
        assert!(result.contains("demonio de inferencia"));
    }
}
