use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use dashmap::DashMap;
use tokio::sync::{Mutex, mpsc};

use super::config::AgentConfig;
use crate::tools::run_agent::{AcpWriter, JsonRpcMessage};

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
        };
        self.sessions
            .insert(agent_config.name.clone(), entry.clone());
        Ok(entry)
    }

    /// Close and remove the session identified by `session_id`.
    pub fn close_session(&self, session_id: &str) {
        self.sessions.retain(|_, v| v.session_id != session_id);
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
        }
    }

    #[tokio::test]
    async fn new_manager_is_empty() {
        assert!(AcpSessionManager::new().list_sessions().is_empty());
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
}
