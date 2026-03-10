use async_trait::async_trait;

use super::Tool;

/// Maximum bytes returned from a single file read.
/// Large enough for most source files; prevents overwhelming the LLM context.
const MAX_BYTES: usize = 16 * 1024; // 16 KB

/// Reads the contents of a file and returns them as text.
///
/// Binary files are detected by the presence of null bytes and rejected.
/// Output is capped at `MAX_BYTES` with a truncation notice.
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads the text content of a file at the given path and returns it. \
         Use when the user asks to read, show, check, or review a file. \
         Output is capped at 16 KB; binary files are rejected."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or home-relative (~) path to the file."
                }
            },
            "required": ["path"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let raw_path = match serde_json::from_str::<serde_json::Value>(args) {
            Ok(v) => match v["path"].as_str() {
                Some(s) if !s.is_empty() => s.to_string(),
                Some(_) => return "No file path provided.".to_string(),
                None => args.trim().to_string(),
            },
            Err(_) => args.trim().to_string(),
        };

        if raw_path.is_empty() {
            return "No file path provided.".to_string();
        }

        let path = expand_tilde(&raw_path);

        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) => return format!("Cannot read '{path}': {e}"),
        };

        // Reject binary files
        if bytes.contains(&0u8) {
            return format!("'{path}' appears to be a binary file and cannot be displayed.");
        }

        let text = String::from_utf8_lossy(&bytes);

        if bytes.len() > MAX_BYTES {
            let truncated = &text[..MAX_BYTES];
            // Trim to last full UTF-8 character boundary
            let safe = truncated
                .char_indices()
                .last()
                .map(|(i, c)| &truncated[..i + c.len_utf8()])
                .unwrap_or(truncated);
            format!(
                "{safe}\n\n[... truncated at 16 KB — file is {} KB total]",
                bytes.len() / 1024
            )
        } else {
            text.into_owned()
        }
    }
}

/// Expands a leading `~` to the user's home directory.
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_read_file() {
        assert_eq!(ReadFileTool.name(), "read_file");
    }

    #[test]
    fn parameters_has_path_field() {
        let p = ReadFileTool.parameters();
        assert_eq!(p["type"], "object");
        assert!(p["properties"]["path"].is_object());
        assert_eq!(p["required"][0], "path");
    }

    #[test]
    fn expand_tilde_replaces_home() {
        // SAFETY: single-threaded test; no other threads read HOME concurrently.
        unsafe { std::env::set_var("HOME", "/Users/test") };
        let result = expand_tilde("~/projects/foo.rs");
        assert_eq!(result, "/Users/test/projects/foo.rs");
    }

    #[test]
    fn expand_tilde_leaves_absolute_paths_unchanged() {
        let result = expand_tilde("/absolute/path/file.txt");
        assert_eq!(result, "/absolute/path/file.txt");
    }

    #[tokio::test]
    async fn reads_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        tokio::fs::write(&path, "hello world").await.unwrap();

        let result = ReadFileTool
            .run(&format!(r#"{{"path": "{}"}}"#, path.display()))
            .await;
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn returns_error_for_missing_file() {
        let result = ReadFileTool
            .run(r#"{"path": "/tmp/nonexistent_voicebot_test_file.txt"}"#)
            .await;
        assert!(
            result.to_lowercase().contains("cannot read")
                || result.to_lowercase().contains("no such file"),
            "should report error: {result:?}"
        );
    }

    #[tokio::test]
    async fn truncates_large_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.txt");
        let big = "x".repeat(MAX_BYTES + 1000);
        tokio::fs::write(&path, &big).await.unwrap();

        let result = ReadFileTool
            .run(&format!(r#"{{"path": "{}"}}"#, path.display()))
            .await;
        assert!(result.contains("truncated"), "should mention truncation: {result:?}");
        assert!(result.len() <= MAX_BYTES + 200); // truncated + notice
    }

    #[tokio::test]
    async fn rejects_binary_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("binary.bin");
        tokio::fs::write(&path, b"hello\x00world").await.unwrap();

        let result = ReadFileTool
            .run(&format!(r#"{{"path": "{}"}}"#, path.display()))
            .await;
        assert!(
            result.to_lowercase().contains("binary"),
            "should reject binary: {result:?}"
        );
    }

    #[tokio::test]
    async fn empty_path_returns_error() {
        let result = ReadFileTool.run(r#"{"path": ""}"#).await;
        assert!(
            result.to_lowercase().contains("no file path"),
            "should report missing path: {result:?}"
        );
    }
}
