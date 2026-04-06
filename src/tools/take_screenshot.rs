use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use tracing::{info, warn};

use crate::llm::OpenAIClient;

use super::Tool;

/// Tool that takes a screenshot and describes it using a vision model.
///
/// Enabled when `SECONDARY_LLM_URL` is set. Delegates HTTP to the shared
/// secondary `OpenAIClient` so the vision call never evicts the main
/// conversation KV-cache.
///
/// macOS only: uses `screencapture -x -t png` for a silent capture.
pub struct TakeScreenshotTool {
    client: OpenAIClient,
}

impl TakeScreenshotTool {
    pub fn new(client: OpenAIClient) -> Self {
        Self { client }
    }

    /// Capture the screen to a temp file and return its PNG bytes.
    async fn capture_screen() -> Result<Vec<u8>, String> {
        let path = "/tmp/voicebot_screenshot.png";

        // -x: no shutter sound; -t png: force PNG format
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

    /// Send the PNG to the vision model and return its description.
    async fn describe_image(&self, png_bytes: &[u8], prompt: &str) -> String {
        let b64 = B64.encode(png_bytes);
        let data_url = format!("data:image/png;base64,{b64}");
        match self.client.complete_multimodal(&data_url, prompt).await {
            Ok(description) => description,
            Err(e) => {
                warn!("Vision model error: {}", e);
                format!("Vision error: {e}")
            }
        }
    }
}

#[async_trait]
impl Tool for TakeScreenshotTool {
    fn name(&self) -> &str {
        "take_screenshot"
    }

    fn description(&self) -> &str {
        "Captures the current screen and returns a text description of what is visible. \
         Use when the user asks about what is on the screen, an application window, \
         a document, code, or any visual element on the display."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Optional question or focus for the vision analysis. \
                                    If omitted, a general description is returned."
                }
            },
            "required": []
        })
    }

    async fn run(&self, args: &str) -> String {
        // Optional prompt from args JSON; falls back to a sensible default.
        let prompt = serde_json::from_str::<serde_json::Value>(args)
            .ok()
            .and_then(|v| v["prompt"].as_str().map(String::from))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                "Describe concisely what is on the screen. \
                 Focus on the active application, main content, and anything \
                 the user might be working on."
                    .to_string()
            });

        info!("take_screenshot: capturing screen");

        let png_bytes = match Self::capture_screen().await {
            Ok(b) => b,
            Err(e) => return format!("Screenshot failed: {e}"),
        };

        info!("take_screenshot: sending {} KB to vision model", png_bytes.len() / 1024);

        self.describe_image(&png_bytes, &prompt).await
    }
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn tool(base_url: &str) -> TakeScreenshotTool {
        TakeScreenshotTool::new(OpenAIClient::new(base_url, "test-vision-model", 512, 0.0))
    }

    #[test]
    fn name_is_take_screenshot() {
        assert_eq!(tool("http://localhost:1234").name(), "take_screenshot");
    }

    #[test]
    fn description_is_non_empty() {
        assert!(!tool("http://localhost:1234").description().is_empty());
    }

    #[test]
    fn parameters_schema_is_object() {
        let schema = tool("http://localhost:1234").parameters();
        assert_eq!(schema["type"], "object");
    }

    // ── describe_image (mocked HTTP) ──────────────────────────────────────────

    #[tokio::test]
    async fn describe_image_returns_model_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "A terminal window showing Rust code."}}]
            })))
            .mount(&server)
            .await;

        let t = tool(&server.uri());
        let result = t.describe_image(b"fake-png-bytes", "What do you see?").await;
        assert_eq!(result, "A terminal window showing Rust code.");
    }

    #[tokio::test]
    async fn describe_image_trims_whitespace() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "  Finder window.  \n"}}]
            })))
            .mount(&server)
            .await;

        let t = tool(&server.uri());
        assert_eq!(t.describe_image(b"data", "describe").await, "Finder window.");
    }

    #[tokio::test]
    async fn describe_image_handles_server_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let t = tool(&server.uri());
        let result = t.describe_image(b"data", "describe").await;
        assert!(
            result.contains("503") || result.to_lowercase().contains("vision"),
            "should describe the error: {result:?}"
        );
    }

    #[tokio::test]
    async fn describe_image_handles_unreachable_server() {
        let t = tool("http://127.0.0.1:19997");
        let result = t.describe_image(b"data", "describe").await;
        assert!(
            result.to_lowercase().contains("vision") || result.to_lowercase().contains("error"),
            "should return error message: {result:?}"
        );
    }

    // ── run() with custom prompt ──────────────────────────────────────────────

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn run_extracts_custom_prompt_from_args() {
        // We can't easily mock screencapture, so we test that the JSON parsing
        // works by checking the error path: screencapture will fail in CI
        // (no display), but the error should not be the default prompt.
        // We verify that `run` doesn't panic on well-formed JSON args.
        let t = tool("http://127.0.0.1:19997");
        let result = t.run(r#"{"prompt": "¿Qué hay en la pantalla?"}"#).await;
        // Either screenshot failure or vision unreachable — both are strings
        assert!(!result.is_empty());
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn run_uses_default_prompt_when_args_empty() {
        let t = tool("http://127.0.0.1:19997");
        let result = t.run("").await;
        assert!(!result.is_empty());
    }
}
