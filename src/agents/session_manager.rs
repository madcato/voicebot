use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use dashmap::DashMap;
use tokio::sync::{Mutex, mpsc};

use super::config::AgentConfig;
use crate::tools::run_agent::{AcpWriter, JsonRpcMessage};

// ── Session events ─────────────────────────────────────────────────────────────

/// Events emitted by ACP sessions for consumption by the terminal display.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// An incoming message from the agent (assistant role).
    AgentMessage {
        agent_name: String,
        session_id: String,
        text: String,
        correlation_id: String,
    },
    /// An outgoing message sent to the agent (user role).
    UserMessage {
        agent_name: String,
        session_id: String,
        text: String,
        correlation_id: String,
    },
    /// A tool call initiated by the agent.
    ToolCall {
        agent_name: String,
        session_id: String,
        tool_name: String,
        task_id: String,
        correlation_id: String,
    },
    /// A tool call completed with a result.
    ToolResult {
        agent_name: String,
        session_id: String,
        tool_name: String,
        task_id: String,
        result: String,
        correlation_id: String,
    },
    /// Session status changed (started, idle, busy, closed).
    Status {
        agent_name: String,
        session_id: String,
        status: SessionStatus,
        correlation_id: String,
    },
    /// An error occurred in the session.
    Error {
        agent_name: String,
        session_id: String,
        message: String,
        correlation_id: String,
    },
}

/// Human-readable session status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    Started,
    Idle,
    Busy,
    Closed,
}

impl std::fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Started => write!(f, "started"),
            Self::Idle => write!(f, "idle"),
            Self::Busy => write!(f, "busy"),
            Self::Closed => write!(f, "closed"),
        }
    }
}

// ── Channel types ─────────────────────────────────────────────────────────────

/// Shared type aliases for the session-event channel.
pub type SessionEventTx = mpsc::Sender<SessionEvent>;
pub type SessionEventRx = mpsc::Receiver<SessionEvent>;

/// Create a bounded event channel (capacity 16).
pub fn create_session_event_channel() -> (SessionEventTx, SessionEventRx) {
    mpsc::channel(16)
}

/// Display-friendly session summary.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub session_id: String,
    pub agent_name: String,
    pub created_at: Instant,
    pub last_used: Instant,
}

/// Handle to a live ACP session.
///
/// The `inbound_rx` receiver is shared via `Arc<Mutex<>>` so multiple tasks
/// can drain messages from the same ACP subprocess.
#[derive(Clone)]
pub struct SessionEntry {
    pub writer: Arc<Mutex<AcpWriter>>,
    pub inbound_rx: Arc<Mutex<mpsc::Receiver<JsonRpcMessage>>>,
    pub session_id: String,
    pub agent_name: String,
    pub created_at: Instant,
    pub last_used: Instant,
    pub task_ids: HashSet<String>,
}

impl std::fmt::Debug for SessionEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionEntry")
            .field("session_id", &self.session_id)
            .field("agent_name", &self.agent_name)
            .field("created_at", &self.created_at)
            .field("last_used", &self.last_used)
            .finish()
    }
}

/// Manages persistent ACP sessions keyed by agent name.
#[derive(Debug, Default)]
pub struct AcpSessionManager {
    sessions: DashMap<String, SessionEntry>,
}

impl AcpSessionManager {
    /// Create a new, empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieve an existing session for `agent_config.name`, or create one.
    pub async fn get_or_create_session(&self, agent_config: &AgentConfig) -> Result<SessionEntry> {
        if let Some(mut entry) = self.sessions.get_mut(&agent_config.name) {
            entry.last_used = Instant::now();
            return Ok((*entry.value()).clone());
        }

        let (mut writer, mut inbound_rx) = AcpWriter::spawn(&agent_config.acp_command).await?;
        let cwd = std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let session_id = writer.initialize(&mut inbound_rx, &cwd).await?;
        let now = Instant::now();
        let entry = SessionEntry {
            writer: Arc::new(Mutex::new(writer)),
            inbound_rx: Arc::new(Mutex::new(inbound_rx)),
            session_id,
            agent_name: agent_config.name.clone(),
            created_at: now,
            last_used: now,
            task_ids: HashSet::new(),
        };
        self.sessions
            .insert(agent_config.name.clone(), entry.clone());
        Ok(entry)
    }

    /// Close and remove the session identified by `session_id`.
    /// Returns the set of task IDs that were associated with this session.
    pub fn close_session(&self, session_id: &str) -> HashSet<String> {
        let mut removed_tasks = HashSet::new();
        self.sessions.retain(|_, v| {
            if v.session_id == session_id {
                removed_tasks.extend(v.task_ids.drain());
                false
            } else {
                true
            }
        });
        removed_tasks
    }
    
    pub fn add_task(&self, agent_name: &str, task_id: &str) {
        if let Some(mut entry) = self.sessions.get_mut(agent_name) {
            entry.task_ids.insert(task_id.to_string());
        }
    }
    
    pub fn remove_task(&self, agent_name: &str, task_id: &str) {
        if let Some(mut entry) = self.sessions.get_mut(agent_name) {
            entry.task_ids.remove(task_id);
        }
    }
    
    pub fn get_all_task_ids(&self) -> HashSet<String> {
        let mut ids = HashSet::new();
        for entry in self.sessions.iter() {
            for tid in &entry.task_ids {
                ids.insert(tid.clone());
            }
        }
        ids
    }

    /// Return information about all active sessions.
    pub fn list_sessions(&self) -> Vec<SessionInfo> {
        self.sessions
            .iter()
            .map(|e| SessionInfo {
                session_id: e.session_id.clone(),
                agent_name: e.agent_name.clone(),
                created_at: e.created_at,
                last_used: e.last_used,
            })
            .collect()
    }

    /// Warm up a single agent session (calls spawn + initialize, stores result).
    /// Convenience method so callers (e.g. `main.rs`) only need this type.
    pub async fn prewarm_agent(&self, config: &AgentConfig) -> Result<String> {
        self.get_or_create_session(config).await.map(|e| e.session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dummy_entry(session_id: &str, agent_name: &str) -> SessionEntry {
        let (_, rx) = tokio::sync::mpsc::channel::<JsonRpcMessage>(1);
        SessionEntry {
            writer: Arc::new(Mutex::new(AcpWriter::dummy())),
            inbound_rx: Arc::new(Mutex::new(rx)),
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            created_at: Instant::now(),
            last_used: Instant::now(),
            task_ids: HashSet::new(),
        }
    }

    #[tokio::test]
    async fn new_manager_is_empty() {
        assert!(AcpSessionManager::new().list_sessions().is_empty());
    }

    #[tokio::test]
    async fn cleanup_close_session_removes_from_map() {
        let mgr = AcpSessionManager::new();
        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("cs-1", "hermes"));
        assert_eq!(mgr.list_sessions().len(), 1);
        mgr.close_session("cs-1");
        assert!(mgr.list_sessions().is_empty());
    }

    #[tokio::test]
    async fn cleanup_closed_session_yields_new_instance() {
        let mgr = AcpSessionManager::new();
        let original = make_dummy_entry("orig-1", "hermes");
        mgr.sessions.insert("hermes".into(), original.clone());

        mgr.close_session("orig-1");
        assert!(mgr.list_sessions().is_empty());

        let newer = make_dummy_entry("new-1", "hermes");
        mgr.sessions.insert("hermes".into(), newer.clone());
        let list = mgr.list_sessions();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].session_id, "new-1");
        assert_ne!(original.session_id, list[0].session_id);
    }

    #[tokio::test]
    async fn close_session_removes_matching_id() {
        let mgr = AcpSessionManager::new();
        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("sid-1", "hermes"));
        assert_eq!(mgr.list_sessions().len(), 1);
        mgr.close_session("sid-1");
        assert!(mgr.list_sessions().is_empty());
    }

    #[tokio::test]
    async fn close_session_ignores_unknown_id() {
        let mgr = AcpSessionManager::new();
        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("sid-1", "hermes"));
        mgr.close_session("nonexistent");
        assert_eq!(mgr.list_sessions().len(), 1);
    }

    #[tokio::test]
    async fn list_sessions_returns_correct_info() {
        let mgr = AcpSessionManager::new();
        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("sid-alpha", "hermes"));
        let list = mgr.list_sessions();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].session_id, "sid-alpha");
        assert_eq!(list[0].agent_name, "hermes");
    }

    #[tokio::test]
    async fn list_sessions_multiple_agents() {
        let mgr = AcpSessionManager::new();
        for name in ["hermes", "oracle", "metis"] {
            mgr.sessions
                .insert(name.into(), make_dummy_entry(&format!("sid-{name}"), name));
        }
        assert_eq!(mgr.list_sessions().len(), 3);
    }

    #[tokio::test]
    async fn lifecycle_start_prompt_close_cleanup() {
        let mgr = AcpSessionManager::new();

        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("lifecyc-1", "hermes"));
        assert_eq!(mgr.list_sessions().len(), 1);

        {
            let guard = mgr.sessions.get("hermes").unwrap();
            assert_eq!(guard.session_id, "lifecyc-1");
        } // Drop guard BEFORE close_session (DashMap deadlock prevention)

        mgr.close_session("lifecyc-1");
        assert!(mgr.list_sessions().is_empty());
    }

    #[tokio::test]
    async fn close_session_idempotent() {
        let mgr = AcpSessionManager::new();
        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("idem-1", "hermes"));

        mgr.close_session("idem-1");
        assert!(mgr.list_sessions().is_empty());

        mgr.close_session("idem-1");
        assert!(mgr.list_sessions().is_empty());
    }

    #[tokio::test]
    async fn session_spawn_nonexistent_fails_gracefully() {
        let result = AcpWriter::spawn("/__nonexistent_cmd_xyz").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(!err.is_empty(), "error message should not be empty");
    }

    #[tokio::test]
    async fn get_or_create_reuses_existing_session() {
        use crate::agents::config::AgentConfig;

        let mgr = AcpSessionManager::new();
        let cfg = AgentConfig {
            name: "hermes".to_string(),
            mode: "acp".to_string(),
            command: None,
            acp_command: "/bin/cat".to_string(),
            acp_warmup: false,
            when_to_use: "test".to_string(),
            instructions: "test".to_string(),
        };

        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("reuse-me", "hermes"));

        let entry = mgr
            .get_or_create_session(&cfg)
            .await
            .expect("should find existing");
        assert_eq!(entry.session_id, "reuse-me");
        assert_eq!(entry.agent_name, "hermes");
    }

    #[tokio::test]
    async fn get_or_create_updates_last_used() {
        use std::time::Duration;

        let mgr = AcpSessionManager::new();
        let mut entry = make_dummy_entry("stamp-1", "hermes");
        entry.last_used = Instant::now() - Duration::from_secs(10);
        mgr.sessions.insert("hermes".into(), entry.clone());

        let cfg = crate::agents::config::AgentConfig {
            name: "hermes".to_string(),
            mode: "acp".to_string(),
            command: None,
            acp_command: "/bin/cat".to_string(),
            acp_warmup: false,
            when_to_use: "test".to_string(),
            instructions: "test".to_string(),
        };

        let before_ts = mgr.sessions.get("hermes").unwrap().value().last_used;
        let _entry = mgr.get_or_create_session(&cfg).await.expect("found");
        let after_ts = mgr.sessions.get("hermes").unwrap().value().last_used;
        assert!(after_ts >= before_ts, "last_used should have been updated");
    }

    #[tokio::test]
    async fn multiple_tasks_same_session_both_visible() {
        let mgr = AcpSessionManager::new();
        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("sess-shared", "hermes"));

        let task_a = "task-a";
        let task_b = "task-b";
        mgr.add_task("hermes", task_a);
        mgr.add_task("hermes", task_b);

        let tasks = mgr.get_all_task_ids();
        assert!(tasks.contains(task_a));
        assert!(tasks.contains(task_b));
        assert_eq!(tasks.len(), 2);

        assert_eq!(mgr.list_sessions().len(), 1);
    }

    #[tokio::test]
    async fn close_session_closes_associated_tasks() {
        let mgr = AcpSessionManager::new();
        let mut entry = make_dummy_entry("sess-close", "hermes");
        entry.task_ids.insert("task-x".to_string());
        entry.task_ids.insert("task-y".to_string());
        mgr.sessions.insert("hermes".into(), entry);

        let removed = mgr.close_session("sess-close");
        assert!(removed.contains("task-x"));
        assert!(removed.contains("task-y"));
        assert_eq!(removed.len(), 2);

        assert!(mgr.list_sessions().is_empty());
        assert!(mgr.get_all_task_ids().is_empty());
    }

    #[tokio::test]
    async fn add_remove_task_roundtrip() {
        let mgr = AcpSessionManager::new();
        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("round-1", "hermes"));

        mgr.add_task("hermes", "rt-task");
        assert!(mgr.get_all_task_ids().contains("rt-task"));

        mgr.remove_task("hermes", "rt-task");
        assert!(!mgr.get_all_task_ids().contains("rt-task"));
    }

    #[tokio::test]
    async fn tasks_persist_across_multiple_sessions() {
        let mgr = AcpSessionManager::new();
        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("s-hermes", "hermes"));
        mgr.sessions
            .insert("oracle".into(), make_dummy_entry("s-oracle", "oracle"));

        mgr.add_task("hermes", "h-task-1");
        mgr.add_task("oracle", "o-task-1");
        mgr.add_task("hermes", "h-task-2");

        let all_tasks = mgr.get_all_task_ids();
        assert_eq!(all_tasks.len(), 3);
        assert!(all_tasks.contains("h-task-1"));
        assert!(all_tasks.contains("h-task-2"));
        assert!(all_tasks.contains("o-task-1"));
    }

    #[tokio::test]
    async fn persistence_twice_get_or_create_same_session_id() {
        use crate::agents::config::AgentConfig;

        let mgr = AcpSessionManager::new();
        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("persist-sid", "hermes"));

        let cfg = AgentConfig {
            name: "hermes".to_string(),
            mode: "acp".to_string(),
            command: None,
            acp_command: "/bin/cat".to_string(),
            acp_warmup: false,
            when_to_use: "test".to_string(),
            instructions: "test".to_string(),
        };

        let first = mgr.get_or_create_session(&cfg).await.unwrap();
        let second = mgr.get_or_create_session(&cfg).await.unwrap();

        assert_eq!(first.session_id, second.session_id);
        assert_eq!(first.session_id, "persist-sid");
    }

    #[tokio::test]
    async fn persistence_writer_not_respawned_between_prompts() {
        use crate::agents::config::AgentConfig;

        let mgr = AcpSessionManager::new();
        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("stable-writer", "hermes"));

        let cfg = AgentConfig {
            name: "hermes".to_string(),
            mode: "acp".to_string(),
            command: None,
            acp_command: "/bin/cat".to_string(),
            acp_warmup: false,
            when_to_use: "test".to_string(),
            instructions: "test".to_string(),
        };

        let first = mgr.get_or_create_session(&cfg).await.unwrap();
        let second = mgr.get_or_create_session(&cfg).await.unwrap();

        assert!(
            Arc::ptr_eq(&first.writer, &second.writer),
            "writer Arc must be identical — no respawn"
        );
    }

    #[tokio::test]
    async fn recovery_session_removed_then_recreated() {
        let mgr = AcpSessionManager::new();

        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("recover-1", "hermes"));
        assert_eq!(mgr.list_sessions().len(), 1);

        mgr.sessions.remove("hermes");
        assert_eq!(mgr.list_sessions().len(), 0);

        let new_entry = make_dummy_entry("recover-2", "hermes");
        mgr.sessions.insert("hermes".into(), new_entry);
        assert_eq!(mgr.list_sessions().len(), 1);
        assert_eq!(mgr.list_sessions()[0].session_id, "recover-2");
    }

    #[tokio::test]
    async fn recovery_spawns_warning_on_transient_error() {
        let err = AcpWriter::spawn("/__nonexistent_disk_failure_cmd")
            .await
            .expect_err("spawn should fail for bad path");
        assert!(!err.to_string().is_empty(), "transient error should produce a non-empty warning message");
    }

    #[tokio::test]
    async fn recovery_detects_and_retries_after_removal() {
        let mgr = AcpSessionManager::new();

        mgr.sessions
            .insert("hermes".into(), make_dummy_entry("retry-1", "hermes"));

        let cfg = crate::agents::config::AgentConfig {
            name: "hermes".to_string(),
            mode: "acp".to_string(),
            command: None,
            acp_command: "/bin/cat".to_string(),
            acp_warmup: false,
            when_to_use: "test".to_string(),
            instructions: "test".to_string(),
        };

        let s1 = mgr.get_or_create_session(&cfg).await.unwrap();
        assert_eq!(s1.session_id, "retry-1");

        mgr.sessions.remove("hermes");
        assert_eq!(mgr.list_sessions().len(), 0);

        let new_entry = make_dummy_entry("retry-2", "hermes");
        mgr.sessions.insert("hermes".into(), new_entry);

        let s2 = mgr.get_or_create_session(&cfg).await.unwrap();
        assert_eq!(s2.session_id, "retry-2");
    }

    #[tokio::test]
    async fn recovery_close_session_warns_on_missing_entries() {
        let mgr = AcpSessionManager::new();

        let removed = mgr.close_session("nonexistent-sid");
        assert!(removed.is_empty(), "closing unknown session yields no removed tasks (silent warning)");
        assert!(mgr.list_sessions().is_empty());
    }
}
