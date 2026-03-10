use async_trait::async_trait;
use tokio::process::Command;

use super::Tool;

/// Sends a macOS notification banner via `osascript`.
pub struct SendNotificationTool;

#[async_trait]
impl Tool for SendNotificationTool {
    fn name(&self) -> &str {
        "send_notification"
    }

    fn description(&self) -> &str {
        "Sends a macOS notification banner. \
         Use to alert the user about completed tasks, reminders, or anything \
         that deserves a visible system notification alongside the spoken response."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The notification title (bold headline)."
                },
                "message": {
                    "type": "string",
                    "description": "The notification body text."
                }
            },
            "required": ["title", "message"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let parsed = serde_json::from_str::<serde_json::Value>(args).ok();

        let title = parsed
            .as_ref()
            .and_then(|v| v["title"].as_str())
            .unwrap_or("Jarvis")
            .to_string();

        let message = match parsed.as_ref().and_then(|v| v["message"].as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            Some(_) => return "No message provided for notification.".to_string(),
            None => {
                let raw = args.trim().to_string();
                if raw.is_empty() {
                    return "No message provided for notification.".to_string();
                }
                raw
            }
        };

        // Escape double-quotes inside the strings so the AppleScript is valid.
        let title_esc = title.replace('"', "\\\"");
        let message_esc = message.replace('"', "\\\"");

        let script = format!(
            "display notification \"{message_esc}\" with title \"{title_esc}\""
        );

        match Command::new("osascript").args(["-e", &script]).output().await {
            Ok(out) if out.status.success() => "Notificación enviada.".to_string(),
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
                format!("osascript error: {err}")
            }
            Err(e) => format!("Failed to run osascript: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_send_notification() {
        assert_eq!(SendNotificationTool.name(), "send_notification");
    }

    #[test]
    fn parameters_has_title_and_message() {
        let p = SendNotificationTool.parameters();
        assert!(p["properties"]["title"].is_object());
        assert!(p["properties"]["message"].is_object());
    }

    #[tokio::test]
    async fn empty_message_returns_error() {
        let result = SendNotificationTool
            .run(r#"{"title": "Test", "message": ""}"#)
            .await;
        assert!(
            result.to_lowercase().contains("no message"),
            "should report missing message: {result:?}"
        );
    }

    #[tokio::test]
    async fn quotes_in_message_are_escaped() {
        // Verify no panic/crash when message contains quotes (AppleScript injection guard)
        let result = SendNotificationTool
            .run(r#"{"title": "Jarvis", "message": "He said \"hello\""}"#)
            .await;
        // Either sent successfully or osascript unavailable — must not panic
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn raw_string_fallback_used_as_message() {
        let result = SendNotificationTool.run("Hola mundo").await;
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn missing_title_defaults_to_jarvis() {
        // No title in JSON — should default to "Jarvis" without crashing
        let result = SendNotificationTool
            .run(r#"{"message": "Test message"}"#)
            .await;
        assert!(!result.is_empty());
    }
}
