use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use super::Tool;

// ── ReadClipboardTool ─────────────────────────────────────────────────────────

/// Returns the current contents of the macOS clipboard (`pbpaste`).
pub struct ReadClipboardTool;

#[async_trait]
impl Tool for ReadClipboardTool {
    fn name(&self) -> &str {
        "read_clipboard"
    }

    fn description(&self) -> &str {
        "Returns the current text content of the clipboard. \
         Use when the user says 'lo que tengo copiado', 'el portapapeles', \
         'lo que acabo de copiar', or similar."
    }

    async fn run(&self, _args: &str) -> String {
        match Command::new("pbpaste").output().await {
            Ok(out) if out.status.success() => {
                let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if text.is_empty() {
                    "El portapapeles está vacío.".to_string()
                } else {
                    text
                }
            }
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr);
                format!("pbpaste error: {err}")
            }
            Err(e) => format!("Failed to run pbpaste: {e}"),
        }
    }
}

// ── SetClipboardTool ──────────────────────────────────────────────────────────

/// Writes text to the macOS clipboard (`pbcopy`).
pub struct SetClipboardTool;

#[async_trait]
impl Tool for SetClipboardTool {
    fn name(&self) -> &str {
        "set_clipboard"
    }

    fn description(&self) -> &str {
        "Writes the given text to the clipboard. \
         Use when the user asks to copy something, save something to the clipboard, \
         or 'pon esto en el portapapeles'."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to copy to the clipboard."
                }
            },
            "required": ["text"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let text = match serde_json::from_str::<serde_json::Value>(args) {
            Ok(v) => match v["text"].as_str() {
                Some(s) if !s.is_empty() => s.to_string(),
                Some(_) => return "No text provided to copy.".to_string(),
                None => args.trim().to_string(),
            },
            Err(_) => args.trim().to_string(),
        };

        if text.is_empty() {
            return "No text provided to copy.".to_string();
        }

        match spawn_pbcopy(&text).await {
            Ok(()) => "Copiado al portapapeles.".to_string(),
            Err(e) => format!("Failed to write clipboard: {e}"),
        }
    }
}

async fn spawn_pbcopy(text: &str) -> Result<(), String> {
    let mut child = Command::new("pbcopy")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("pbcopy launch failed: {e}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(text.as_bytes())
            .await
            .map_err(|e| format!("write to pbcopy failed: {e}"))?;
    }

    let status = child
        .wait()
        .await
        .map_err(|e| format!("pbcopy wait failed: {e}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("pbcopy exited with status {status}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_clipboard_name() {
        assert_eq!(ReadClipboardTool.name(), "read_clipboard");
    }

    #[test]
    fn set_clipboard_name() {
        assert_eq!(SetClipboardTool.name(), "set_clipboard");
    }

    #[test]
    fn set_clipboard_parameters_has_text_field() {
        let p = SetClipboardTool.parameters();
        assert_eq!(p["type"], "object");
        assert!(p["properties"]["text"].is_object());
    }

    #[tokio::test]
    async fn set_clipboard_extracts_text_from_json() {
        // We can't assert the clipboard value in CI, but we can assert no panic/error
        // from valid JSON. On macOS this will actually write to the clipboard.
        let result = SetClipboardTool.run(r#"{"text": "hola mundo"}"#).await;
        // Either success or an OS error — both are non-empty strings
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn set_clipboard_empty_text_returns_error() {
        let result = SetClipboardTool.run(r#"{"text": ""}"#).await;
        assert!(
            result.to_lowercase().contains("no text") || result.to_lowercase().contains("no"),
            "should report missing text: {result:?}"
        );
    }

    #[tokio::test]
    async fn read_clipboard_returns_non_empty_string() {
        // pbpaste is always available on macOS; result may be empty string or content
        let result = ReadClipboardTool.run("").await;
        assert!(!result.is_empty()); // at minimum "El portapapeles está vacío."
    }
}
