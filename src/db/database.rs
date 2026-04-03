use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use std::path::Path;
use uuid::Uuid;

/// A persistent memory extracted from conversation history.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Memory {
    pub id: i64,
    pub content: String,
    pub category: String,
    pub source_session_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A new memory to be inserted (no id yet).
#[derive(Debug, Clone)]
pub struct NewMemory {
    pub content: String,
    pub category: String,
}

/// SQLite database for persistent chat history.
#[derive(Clone)]
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    pub async fn new(database_path: &str) -> Result<Self> {
        if let Some(parent) = Path::new(database_path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let options = SqliteConnectOptions::new()
            .filename(database_path)
            .create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .context("Failed to connect to database")?;

        let db = Self { pool };
        db.run_migrations().await?;
        Ok(db)
    }

    async fn run_migrations(&self) -> Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                closed_at TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                summary TEXT,
                summary_through_id INTEGER NOT NULL DEFAULT 0
            )",
        )
        .execute(&self.pool)
        .await?;

        // Additive migration: add summary columns to existing databases.
        // SQLite does not support IF NOT EXISTS for ADD COLUMN, so we ignore the error.
        let _ = sqlx::query("ALTER TABLE sessions ADD COLUMN summary TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query(
            "ALTER TABLE sessions ADD COLUMN summary_through_id INTEGER NOT NULL DEFAULT 0",
        )
        .execute(&self.pool)
        .await;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)",
        )
        .execute(&self.pool)
        .await?;

        // User profile: one row per fact key, updated in place.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS user_profile (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        // Persistent memories extracted during context consolidation.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS memories (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                content           TEXT NOT NULL,
                category          TEXT NOT NULL DEFAULT 'general',
                source_session_id TEXT,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                is_active         INTEGER NOT NULL DEFAULT 1
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active)",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Return the last active session ID, or create a new one.
    pub async fn get_or_create_session(&self) -> Result<Uuid> {
        let row = sqlx::query(
            "SELECT id FROM sessions WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let id: String = row.try_get("id")?;
            let uuid = Uuid::parse_str(&id)?;
            tracing::info!(target: "db", "Restored session {}", uuid);
            return Ok(uuid);
        }

        let id = Uuid::new_v4();
        self.create_session(id).await?;
        tracing::info!(target: "db", "Created new session {}", id);
        Ok(id)
    }

    pub async fn create_session(&self, session_id: Uuid) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query("INSERT INTO sessions (id, created_at, is_active) VALUES (?, ?, 1)")
            .bind(session_id.to_string())
            .bind(now)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Load the session's summary (if any) and the messages after the summary cutoff.
    ///
    /// Returns `(summary_text, recent_messages)`. If no summary exists, all messages
    /// are returned. Used on startup to restore the LLM session efficiently.
    pub async fn get_session_context(
        &self,
        session_id: Uuid,
    ) -> Result<(Option<String>, Vec<(String, String)>)> {
        let row = sqlx::query(
            "SELECT summary, summary_through_id FROM sessions WHERE id = ?",
        )
        .bind(session_id.to_string())
        .fetch_one(&self.pool)
        .await?;

        let summary: Option<String> = row.try_get("summary")?;
        let through_id: i64 = row.try_get("summary_through_id").unwrap_or(0);

        let messages = self.get_messages_after_id(session_id, through_id).await?;
        Ok((summary, messages))
    }

    /// Load messages with id > after_id. If after_id is 0, loads all messages.
    pub async fn get_messages_after_id(
        &self,
        session_id: Uuid,
        after_id: i64,
    ) -> Result<Vec<(String, String)>> {
        let rows = sqlx::query(
            "SELECT role, content FROM messages
             WHERE session_id = ? AND id > ?
             ORDER BY id ASC",
        )
        .bind(session_id.to_string())
        .bind(after_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|row| {
                let role: String = row.try_get("role").unwrap_or_default();
                let content: String = row.try_get("content").unwrap_or_default();
                (role, content)
            })
            .collect())
    }

    /// Return the message id at a 0-based offset within a session (ordered by id ASC),
    /// counting only messages with `id > after_id`.
    ///
    /// Pass `after_id = 0` to count from the beginning.
    /// Pass the current `summary_through_id` to count only within the currently-loaded
    /// batch — this ensures the new cutoff is always strictly ahead of the old one.
    pub async fn get_message_id_at_offset(
        &self,
        session_id: Uuid,
        after_id: i64,
        offset: usize,
    ) -> Result<Option<i64>> {
        let row = sqlx::query(
            "SELECT id FROM messages WHERE session_id = ? AND id > ? ORDER BY id ASC LIMIT 1 OFFSET ?",
        )
        .bind(session_id.to_string())
        .bind(after_id)
        .bind(offset as i64)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| r.try_get::<i64, _>("id").unwrap_or(0)))
    }

    /// Return the current `summary_through_id` for a session (0 if no summary yet).
    pub async fn get_summary_through_id(&self, session_id: Uuid) -> Result<i64> {
        let row = sqlx::query(
            "SELECT summary_through_id FROM sessions WHERE id = ?",
        )
        .bind(session_id.to_string())
        .fetch_one(&self.pool)
        .await?;
        Ok(row.try_get("summary_through_id").unwrap_or(0))
    }

    /// Persist the conversation summary and the id of the last summarized message.
    ///
    /// On the next startup, only messages with id > through_message_id will be loaded,
    /// and the summary will be injected into the system prompt.
    pub async fn save_summary(
        &self,
        session_id: Uuid,
        summary: &str,
        through_message_id: i64,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE sessions SET summary = ?, summary_through_id = ? WHERE id = ?",
        )
        .bind(summary)
        .bind(through_message_id)
        .bind(session_id.to_string())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Persist the tool-call exchange messages for a turn (assistant tool_calls + tool results).
    ///
    /// Serialises the full JSON array into a single row with role "ToolExchanges" so that
    /// on next startup the session can reconstruct the exact tool-call context the LLM saw.
    /// Without this, the model only sees the final assistant text response after a tool call
    /// and cannot distinguish correctly-called tools from hallucinated ones.
    pub async fn save_tool_exchanges(
        &self,
        session_id: Uuid,
        exchanges: &[serde_json::Value],
    ) -> Result<()> {
        if exchanges.is_empty() {
            return Ok(());
        }
        let json = serde_json::to_string(exchanges)?;
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        )
        .bind(session_id.to_string())
        .bind("ToolExchanges")
        .bind(json)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Persist a single message turn.
    pub async fn save_message(&self, session_id: Uuid, role: &str, content: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        )
        .bind(session_id.to_string())
        .bind(role)
        .bind(content)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn close_session(&self, session_id: Uuid) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query("UPDATE sessions SET closed_at = ?, is_active = 0 WHERE id = ?")
            .bind(now)
            .bind(session_id.to_string())
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // ── User profile ──────────────────────────────────────────────────────────

    /// Load all profile facts ordered by key.
    pub async fn load_user_profile(&self) -> Result<Vec<(String, String, f64)>> {
        let rows = sqlx::query(
            "SELECT key, value, confidence FROM user_profile ORDER BY key ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|r| {
                let key: String = r.try_get("key").unwrap_or_default();
                let value: String = r.try_get("value").unwrap_or_default();
                let confidence: f64 = r.try_get("confidence").unwrap_or(1.0);
                (key, value, confidence)
            })
            .collect())
    }

    /// Insert or update a profile fact.
    ///
    /// An existing fact is only overwritten when the new confidence is strictly
    /// higher — this prevents low-quality inferences from degrading confirmed facts.
    #[allow(dead_code)]
    pub async fn upsert_profile_fact(
        &self,
        key: &str,
        value: &str,
        confidence: f64,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO user_profile (key, value, confidence, updated_at)
             VALUES (?, ?, ?, ?)
             ON CONFLICT(key) DO UPDATE SET
                 value      = excluded.value,
                 confidence = excluded.confidence,
                 updated_at = excluded.updated_at
             WHERE excluded.confidence > user_profile.confidence",
        )
        .bind(key)
        .bind(value)
        .bind(confidence)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn list_sessions(&self) -> Result<Vec<(Uuid, DateTime<Utc>)>> {
        let rows = sqlx::query("SELECT id, created_at FROM sessions ORDER BY created_at DESC")
            .fetch_all(&self.pool)
            .await?;

        let mut sessions = Vec::new();
        for row in rows {
            let id: String = row.try_get("id")?;
            let created_at: String = row.try_get("created_at")?;
            let uuid = Uuid::parse_str(&id)?;
            let timestamp = DateTime::parse_from_rfc3339(&created_at)?.with_timezone(&Utc);
            sessions.push((uuid, timestamp));
        }
        Ok(sessions)
    }

    // ── Memories ──────────────────────────────────────────────────────────────

    /// Load all active memories, most recently updated first.
    pub async fn load_active_memories(&self) -> Result<Vec<Memory>> {
        let rows = sqlx::query(
            "SELECT id, content, category, source_session_id, created_at, updated_at
             FROM memories WHERE is_active = 1 ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|r| Memory {
                id: r.try_get("id").unwrap_or(0),
                content: r.try_get("content").unwrap_or_default(),
                category: r.try_get("category").unwrap_or_default(),
                source_session_id: r.try_get("source_session_id").ok(),
                created_at: r.try_get("created_at").unwrap_or_default(),
                updated_at: r.try_get("updated_at").unwrap_or_default(),
            })
            .collect())
    }

    /// Insert multiple memories in a single transaction.
    pub async fn save_memories_batch(
        &self,
        memories: &[NewMemory],
        session_id: Uuid,
    ) -> Result<()> {
        if memories.is_empty() {
            return Ok(());
        }
        let now = Utc::now().to_rfc3339();
        let sid = session_id.to_string();

        let mut tx = self.pool.begin().await?;
        for mem in memories {
            sqlx::query(
                "INSERT INTO memories (content, category, source_session_id, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?)",
            )
            .bind(&mem.content)
            .bind(&mem.category)
            .bind(&sid)
            .bind(&now)
            .bind(&now)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Soft-delete a memory by setting is_active = 0.
    pub async fn deactivate_memory(&self, memory_id: i64) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query("UPDATE memories SET is_active = 0, updated_at = ? WHERE id = ?")
            .bind(now)
            .bind(memory_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
