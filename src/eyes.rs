/// EYES — periodic visual awareness loop.
///
/// Every `interval_secs` (configured via `EYES_INTERVAL_SECS`) this module
/// takes a silent screenshot, sends it to the secondary vision LLM, and asks
/// it to decide whether anything on screen warrants notifying the user.
///
/// The secondary LLM must respond with a structured two-field format:
///   warn_user: true|false
///   message: <optional natural-language sentence>
///
/// When `warn_user` is true, an `AgentResult` proactive event is pushed so
/// the main assistant LLM reformulates the message in Jarvis's voice before
/// speaking it to the user.
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::agents::ProactiveEvent;
use crate::llm::OpenAIClient;

const EYES_PROMPT: &str = "\
You are a background visual monitor for a voice assistant called Jarvis. \
Your task is to inspect the screenshot and decide if there is anything the user \
should be told about RIGHT NOW — such as an error dialog, a warning, an \
important notification, a deadline, a critical status, or something genuinely \
useful the user might have missed.\n\n\
Rules:\n\
- Be conservative. Only set warn_user to true for things that are clearly \
  important or time-sensitive.\n\
- Do NOT warn about normal application state, idle screens, or routine content.\n\
- If you decide to warn the user, write the message in the same language visible \
  on screen (Spanish or English). Keep it to 1-2 short sentences.\n\n\
Respond ONLY in this exact format (no extra text):\n\
warn_user: true\n\
message: <your message here>\n\n\
OR:\n\
warn_user: false";

pub struct EyesDaemon {
    pub interval_secs: u64,
    pub vision_client: OpenAIClient,
    pub proactive_tx: mpsc::Sender<ProactiveEvent>,
}

impl EyesDaemon {
    /// Spawns the EYES daemon as a background tokio task.
    pub fn spawn(self) {
        tokio::spawn(async move {
            self.run().await;
        });
    }

    async fn run(self) {
        info!(
            target: "eyes",
            "EYES daemon started (interval={}s)",
            self.interval_secs
        );

        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(self.interval_secs)).await;

            if self.proactive_tx.capacity() == 0 {
                debug!(target: "eyes", "Proactive channel full, skipping tick");
                continue;
            }

            debug!(target: "eyes", "EYES tick — capturing screen");

            let png_bytes = match capture_screen().await {
                Ok(b) => b,
                Err(e) => {
                    warn!(target: "eyes", "Screenshot failed: {}", e);
                    continue;
                }
            };

            let b64 = B64.encode(&png_bytes);
            let data_url = format!("data:image/png;base64,{b64}");

            let raw = match self.vision_client.complete_multimodal(&data_url, EYES_PROMPT).await {
                Ok(r) => r,
                Err(e) => {
                    warn!(target: "eyes", "Vision LLM call failed: {}", e);
                    continue;
                }
            };

            match parse_eyes_response(&raw) {
                Some(message) => {
                    info!(target: "eyes", "EYES: warning user → {:?}", message);
                    let event = ProactiveEvent::AgentResult {
                        task: "EYES — observación visual".to_string(),
                        result: message,
                    };
                    if let Err(e) = self.proactive_tx.try_send(event) {
                        warn!(target: "eyes", "Failed to send proactive event: {}", e);
                    }
                }
                None => {
                    debug!(target: "eyes", "EYES: nothing to report");
                }
            }
        }
    }
}

/// Capture the screen silently to a temp file and return the PNG bytes.
async fn capture_screen() -> Result<Vec<u8>, String> {
    let path = "/tmp/voicebot_eyes.png";
    let output = tokio::process::Command::new("screencapture")
        .args(["-x", "-t", "png", path])
        .output()
        .await
        .map_err(|e| format!("screencapture failed to launch: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("screencapture error: {stderr}"));
    }

    tokio::fs::read(path)
        .await
        .map_err(|e| format!("failed to read screenshot file: {e}"))
}

/// Parse the structured EYES response.
///
/// Returns `Some(message)` when `warn_user: true` and a message is present,
/// `None` otherwise.
fn parse_eyes_response(raw: &str) -> Option<String> {
    let mut warn_user = false;
    let mut message: Option<String> = None;

    for line in raw.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("warn_user:") {
            let val = rest.trim().to_lowercase();
            warn_user = val == "true";
        } else if let Some(rest) = line.strip_prefix("message:") {
            let msg = rest.trim().to_string();
            if !msg.is_empty() {
                message = Some(msg);
            }
        }
    }

    if warn_user { message } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_warn_true_with_message() {
        let raw = "warn_user: true\nmessage: Hay un error en la terminal.";
        assert_eq!(
            parse_eyes_response(raw),
            Some("Hay un error en la terminal.".to_string())
        );
    }

    #[test]
    fn parse_warn_false_returns_none() {
        let raw = "warn_user: false";
        assert_eq!(parse_eyes_response(raw), None);
    }

    #[test]
    fn parse_warn_true_without_message_returns_none() {
        let raw = "warn_user: true";
        assert_eq!(parse_eyes_response(raw), None);
    }

    #[test]
    fn parse_extra_whitespace_and_casing() {
        let raw = "  warn_user:  True  \n  message:  Something important.  ";
        assert_eq!(
            parse_eyes_response(raw),
            Some("Something important.".to_string())
        );
    }

    #[test]
    fn parse_multiline_response_picks_first_fields() {
        let raw = "warn_user: true\nmessage: Error crítico detectado.\nSome extra line.";
        assert_eq!(
            parse_eyes_response(raw),
            Some("Error crítico detectado.".to_string())
        );
    }
}
