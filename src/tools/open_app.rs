use async_trait::async_trait;
use tokio::process::Command;

use super::Tool;

/// Opens a macOS application by name using `open -a`.
pub struct OpenAppTool;

#[async_trait]
impl Tool for OpenAppTool {
    fn name(&self) -> &str {
        "open_app"
    }

    fn description(&self) -> &str {
        "Opens a macOS application by name. \
         Use when the user asks to open, launch, or start an application. \
         Examples: 'abre Cursor', 'lanza Safari', 'abre la terminal'."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The application name as it appears in /Applications, \
                                    e.g. 'Cursor', 'Safari', 'Terminal', 'Finder'."
                }
            },
            "required": ["name"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let name = match serde_json::from_str::<serde_json::Value>(args) {
            Ok(v) => match v["name"].as_str() {
                Some(s) if !s.is_empty() => s.to_string(),
                Some(_) => return "No application name provided.".to_string(),
                None => args.trim().to_string(),
            },
            Err(_) => args.trim().to_string(),
        };

        if name.is_empty() {
            return "No application name provided.".to_string();
        }

        match Command::new("open").args(["-a", &name]).output().await {
            Ok(out) if out.status.success() => {
                format!("Abriendo {name}.")
            }
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
                if err.is_empty() {
                    format!("Could not open '{name}': application not found.")
                } else {
                    format!("Could not open '{name}': {err}")
                }
            }
            Err(e) => format!("Failed to run open: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_open_app() {
        assert_eq!(OpenAppTool.name(), "open_app");
    }

    #[test]
    fn description_is_non_empty() {
        assert!(!OpenAppTool.description().is_empty());
    }

    #[test]
    fn parameters_has_name_field() {
        let p = OpenAppTool.parameters();
        assert_eq!(p["type"], "object");
        assert!(p["properties"]["name"].is_object());
        assert_eq!(p["required"][0], "name");
    }

    #[tokio::test]
    async fn empty_name_returns_error() {
        let result = OpenAppTool.run(r#"{"name": ""}"#).await;
        assert!(
            result.to_lowercase().contains("no application"),
            "should report missing name: {result:?}"
        );
    }

    #[tokio::test]
    async fn nonexistent_app_returns_error_message() {
        let result = OpenAppTool.run(r#"{"name": "NonexistentApp12345"}"#).await;
        assert!(
            result.to_lowercase().contains("could not open")
                || result.to_lowercase().contains("not found")
                || result.to_lowercase().contains("unable"),
            "should report failure: {result:?}"
        );
    }

    #[tokio::test]
    async fn raw_string_fallback_works() {
        // When args is not JSON, treat the whole string as the app name
        let result = OpenAppTool.run("NonexistentApp12345").await;
        // Should fail gracefully, not panic
        assert!(!result.is_empty());
    }
}
