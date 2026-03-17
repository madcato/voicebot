use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use std::path::Path;
use uuid::Uuid;

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

    /// Return the message id at a 0-based offset within a session (ordered by id ASC).
    ///
    /// Used to determine the cutoff point before saving a summary: the summary covers
    /// all messages up to and including this id.
    pub async fn get_message_id_at_offset(
        &self,
        session_id: Uuid,
        offset: usize,
    ) -> Result<Option<i64>> {
        let row = sqlx::query(
            "SELECT id FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT 1 OFFSET ?",
        )
        .bind(session_id.to_string())
        .bind(offset as i64)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| r.try_get::<i64, _>("id").unwrap_or(0)))
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
}
