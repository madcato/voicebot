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
                is_active INTEGER NOT NULL DEFAULT 1
            )",
        )
        .execute(&self.pool)
        .await?;

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

        Ok(())
    }

    /// Return the last active session ID, or create a new one.
    /// This is the main entry point on startup — restores conversation context.
    pub async fn get_or_create_session(&self) -> Result<Uuid> {
        let row = sqlx::query(
            "SELECT id FROM sessions WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let id: String = row.try_get("id")?;
            let uuid = Uuid::parse_str(&id)?;
            tracing::info!("Restored session {}", uuid);
            return Ok(uuid);
        }

        let id = Uuid::new_v4();
        self.create_session(id).await?;
        tracing::info!("Created new session {}", id);
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

    /// Load all text messages for a session as (role, content) pairs.
    /// Used to reconstruct the LLM accumulated prompt on startup.
    pub async fn get_session_messages(&self, session_id: Uuid) -> Result<Vec<(String, String)>> {
        let rows = sqlx::query(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
        )
        .bind(session_id.to_string())
        .fetch_all(&self.pool)
        .await?;

        let messages = rows
            .into_iter()
            .map(|row| {
                let role: String = row.try_get("role").unwrap_or_default();
                let content: String = row.try_get("content").unwrap_or_default();
                (role, content)
            })
            .collect();

        Ok(messages)
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

    pub async fn close_session(&self, session_id: Uuid) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "UPDATE sessions SET closed_at = ?, is_active = 0 WHERE id = ?",
        )
        .bind(now)
        .bind(session_id.to_string())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

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
