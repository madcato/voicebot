use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use chrono::Local;

use super::session_events::{AcpSessionEvent, SessionEventRx};

const LOG_DIR: &str = "/tmp/voicebot_sessions";

/// Formats and displays ACP session events to a log file.
///
/// Consumes events from a bounded channel and writes formatted lines to
/// `/tmp/voicebot_sessions/{session_id}.log`. The worker shuts down cleanly
/// when the channel closes.
pub struct SessionDisplayWorker {
    session_id: String,
    rx: SessionEventRx,
}

impl SessionDisplayWorker {
    pub fn new(session_id: String, rx: SessionEventRx) -> Self {
        Self { session_id, rx }
    }

    /// Resolve the log file path for a session.
    fn log_path(session_id: &str) -> PathBuf {
        let dir = PathBuf::from(LOG_DIR);
        dir.join(format!("{session_id}.log"))
    }

    /// Spawn the display worker as a background task.
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move { self.run().await })
    }

    async fn run(mut self) {
        let dir = PathBuf::from(LOG_DIR);
        let _ = std::fs::create_dir_all(&dir);

        let path = Self::log_path(&self.session_id);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|_| panic!("cannot open log file: {:?}", path));

        while let Some(event) = self.rx.recv().await {
            let line = format_event(&event);
            writeln!(file, "{line}").ok();
            flush_file(&mut file);
        }
    }
}

/// Format an event into a colored log line: `[HH:MM:SS] [TYPE] content`.
fn format_event(event: &AcpSessionEvent) -> String {
    let ts = Local::now().format("%H:%M:%S");
    match event {
        AcpSessionEvent::AgentMessageChunk(text) => {
            format!("[\033[36m{ts}\033[0m] [\033[32mAGENT\033[0m] {text}")
        }
        AcpSessionEvent::AgentThoughtChunk(text) => {
            format!("[\033[36m{ts}\033[0m] [\033[33mTHINK\033[0m] {text}")
        }
        AcpSessionEvent::ToolCall { name } => {
            format!("[\033[36m{ts}\033[0m] [\033[34mTOOL\033[0m] {name}: started")
        }
        AcpSessionEvent::ToolCallUpdate { name, status } => {
            format!("[\033[36m{ts}\033[0m] [\033[34mTOOL\033[0m] {name}: {status}")
        }
        AcpSessionEvent::PermissionRequest {
            description,
            options,
        } => {
            let opts = options.join(", ");
            format!("[\033[36m{ts}\033[0m] [\033[31mPERM\033[0m] {description}? [{opts}]")
        }
    }
}

fn flush_file(file: &mut std::fs::File) {
    file.flush().ok();
}

/// Resolve the log path for a session (public helper for terminal integration).
pub fn session_log_path(session_id: &str) -> PathBuf {
    SessionDisplayWorker::log_path(session_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_agent_message() {
        let event = AcpSessionEvent::AgentMessageChunk("hello".to_string());
        let line = format_event(&event);
        assert!(line.contains("AGENT"));
        assert!(line.contains("hello"));
    }

    #[test]
    fn test_format_tool_call() {
        let event = AcpSessionEvent::ToolCall {
            name: "web_search".to_string(),
        };
        let line = format_event(&event);
        assert!(line.contains("TOOL"));
        assert!(line.contains("web_search"));
        assert!(line.contains("started"));
    }

    #[test]
    fn test_format_thought() {
        let event = AcpSessionEvent::AgentThoughtChunk("reasoning".to_string());
        let line = format_event(&event);
        assert!(line.contains("THINK"));
        assert!(line.contains("reasoning"));
    }

    #[test]
    fn test_format_permission() {
        let event = AcpSessionEvent::PermissionRequest {
            description: "Allow?".to_string(),
            options: vec!["yes".to_string(), "no".to_string()],
        };
        let line = format_event(&event);
        assert!(line.contains("PERM"));
        assert!(line.contains("Allow?"));
        assert!(line.contains("yes, no"));
    }

    #[test]
    fn test_writes_to_file() {
        use tokio::sync::mpsc;

        let tmp_dir = std::env::temp_dir().join("voicebot_test");
        let log_dir = tmp_dir.join("sessions");
        let _ = std::fs::create_dir_all(&log_dir);

        let session_id = "test-session-001";
        let path = log_dir.join(format!("{session_id}.log"));

        let (tx, mut rx) = mpsc::channel::<AcpSessionEvent>(16);
        tx.blocking_send(AcpSessionEvent::AgentMessageChunk("test event".into()))
            .unwrap();
        drop(tx);

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();
        while let Ok(event) = rx.try_recv() {
            let line = format_event(&event);
            writeln!(file, "{line}").unwrap();
        }

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("AGENT"));
        assert!(content.contains("test event"));

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    #[test]
    fn test_format_includes_timestamp() {
        let event = AcpSessionEvent::AgentMessageChunk("x".to_string());
        let line = format_event(&event);
        let stripped = strip_ansi(&line);
        assert!(stripped.starts_with('['));
        assert!(stripped.contains(':'));
    }

    fn strip_ansi(s: &str) -> String {
        s.replace("\u{001b}[36m", "")
            .replace("\u{001b}[32m", "")
            .replace("\u{001b}[33m", "")
            .replace("\u{001b}[34m", "")
            .replace("\u{001b}[31m", "")
            .replace("\u{001b}[0m", "")
    }

    #[test]
    fn test_log_path_creates_file() {
        let path = session_log_path("abc123");
        assert_eq!(path.to_string_lossy(), "/tmp/voicebot_sessions/abc123.log");
    }
}
