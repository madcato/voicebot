use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::{Client, StatusCode, header};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, trace, warn};

/// Client for the Jarvis Voicebot Control API.
///
/// Provides programmatic control over the voicebot pipeline for testing,
/// debugging, and automation. Supports both sync and async operations.
#[derive(Clone)]
pub struct ControlClient {
    client: Client,
    base_url: String,
}

/// Builder for configuring `ControlClient` instances.
pub struct ControlClientBuilder {
    base_url: String,
    timeout: Duration,
    connect_timeout: Duration,
}

/// Response from the health check endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
}

/// Response from the state endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct StateResponse {
    pub state: String,
    pub utterance_id: Option<u64>,
    pub tts_muted: bool,
}

/// Request body for the mute endpoint.
#[derive(Debug, Clone, Serialize)]
struct MuteRequest {
    muted: bool,
}

/// Request body for the input endpoint.
#[derive(Debug, Clone, Serialize)]
struct InputRequest {
    text: String,
}

/// Errors specific to the Control API client.
#[derive(Debug, Error)]
pub enum ControlClientError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("API returned error status {status}: {message}")]
    ApiError { status: StatusCode, message: String },

    #[error("SSE stream closed unexpectedly")]
    StreamClosed,

    #[error("Connection timeout after {duration:?}")]
    ConnectionTimeout { duration: Duration },

    #[error("Invalid SSE event format: {0}")]
    InvalidEventFormat(String),

    #[error("Server not responding to health checks")]
    HealthCheckFailed,

    #[error("Event wait timeout after {0:?}")]
    WaitTimeout(Duration),

    #[error("State assertion failed: expected {expected}, got {actual}")]
    StateAssertionFailed { expected: String, actual: String },
}

/// Events emitted by the voicebot control system.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientControlEvent {
    StateChanged { state: String, utterance_id: Option<u64> },
    Transcript { utterance_id: u64, text: String },
    LlmToken { utterance_id: u64, token: String },
    LlmDone { utterance_id: u64, full_text: String },
    TtsStart { utterance_id: u64 },
    ToolCall { name: String, result: String },
    MuteChanged { muted: bool },
    Error { message: String },
}

impl ControlClientBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            base_url: "http://127.0.0.1:8080".to_string(),
            timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(5),
        }
    }

    /// Set the base URL for the control API.
    pub fn base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }

    /// Set the request timeout.
    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = duration;
        self
    }

    /// Set the connection timeout.
    pub fn connect_timeout(mut self, duration: Duration) -> Self {
        self.connect_timeout = duration;
        self
    }

    /// Build the `ControlClient`.
    pub fn build(self) -> Result<ControlClient> {
        let client = Client::builder()
            .tcp_keepalive(Duration::from_secs(60))
            .tcp_nodelay(true)
            .connect_timeout(self.connect_timeout)
            .timeout(self.timeout)
            .pool_max_idle_per_host(4)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .context("failed to build HTTP client")?;

        Ok(ControlClient {
            client,
            base_url: self.base_url,
        })
    }
}

impl Default for ControlClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ControlClient {
    /// Create a new client with default settings.
    pub async fn new(base_url: &str) -> Result<Self> {
        Self::builder().base_url(base_url).build()
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> ControlClientBuilder {
        ControlClientBuilder::new()
    }

    // ============================================================================
    // HEALTH & STATE
    // ============================================================================

    /// Check if the voicebot control server is healthy.
    pub async fn health_check(&self) -> Result<HealthResponse, ControlClientError> {
        let url = format!("{}/control/health", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(ControlClientError::HttpError)?;

        if !response.status().is_success() {
            let status = response.status();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ControlClientError::ApiError { status, message });
        }

        response
            .json::<HealthResponse>()
            .await
            .map_err(ControlClientError::HttpError)
    }

    /// Get the current pipeline state.
    pub async fn get_state(&self) -> Result<StateResponse, ControlClientError> {
        let url = format!("{}/control/state", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(ControlClientError::HttpError)?;

        if !response.status().is_success() {
            let status = response.status();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ControlClientError::ApiError { status, message });
        }

        response
            .json::<StateResponse>()
            .await
            .map_err(ControlClientError::HttpError)
    }

    /// Get the conversation history.
    pub async fn get_history(&self) -> Result<Vec<serde_json::Value>, ControlClientError> {
        let url = format!("{}/control/history", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(ControlClientError::HttpError)?;

        if !response.status().is_success() {
            let status = response.status();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ControlClientError::ApiError { status, message });
        }

        response
            .json::<Vec<serde_json::Value>>()
            .await
            .map_err(ControlClientError::HttpError)
    }

    // ============================================================================
    // CONTROL ACTIONS
    // ============================================================================

    /// Mute or unmute TTS output.
    pub async fn set_mute(&self, muted: bool) -> Result<(), ControlClientError> {
        let url = format!("{}/control/mute", self.base_url);
        let body = MuteRequest { muted };

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(ControlClientError::HttpError)?;

        if !response.status().is_success() {
            let status = response.status();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ControlClientError::ApiError { status, message });
        }

        Ok(())
    }

    /// Trigger barge-in to interrupt current speech.
    pub async fn barge_in(&self) -> Result<(), ControlClientError> {
        let url = format!("{}/control/barge_in", self.base_url);

        let response = self
            .client
            .post(&url)
            .send()
            .await
            .map_err(ControlClientError::HttpError)?;

        if !response.status().is_success() {
            let status = response.status();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ControlClientError::ApiError { status, message });
        }

        Ok(())
    }

    /// Send text input to the pipeline (bypasses STT).
    pub async fn send_input(&self, text: &str) -> Result<(), ControlClientError> {
        let url = format!("{}/control/input", self.base_url);
        let body = InputRequest {
            text: text.to_string(),
        };

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(ControlClientError::HttpError)?;

        if !response.status().is_success() {
            let status = response.status();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ControlClientError::ApiError { status, message });
        }

        Ok(())
    }

    // ============================================================================
    // SSE STREAMING
    // ============================================================================

    /// Subscribe to the SSE event stream.
    ///
    /// Returns a channel receiver that will receive events as they occur.
    /// The channel has a buffer of 100 events.
    pub async fn subscribe_events(&self) -> Result<mpsc::Receiver<ClientControlEvent>, ControlClientError> {
        let url = format!("{}/control/events", self.base_url);
        let (tx, rx) = mpsc::channel(100);
        let client = self.client.clone();

        tokio::spawn(async move {
            let response = match client
                .get(&url)
                .header(header::ACCEPT, "text/event-stream")
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    error!(target: "control_client", "Failed to connect to SSE stream: {}", e);
                    return;
                }
            };

            if !response.status().is_success() {
                error!(target: "control_client", "SSE connection failed: {}", response.status());
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        // Process complete SSE events in buffer
                        while let Some(pos) = buffer.find("\n\n") {
                            let event_text = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();

                            if let Some(event) = Self::parse_sse_event(&event_text) {
                                if tx.send(event).await.is_err() {
                                    trace!(target: "control_client", "Event receiver dropped, closing SSE stream");
                                    return;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!(target: "control_client", "SSE stream error: {}", e);
                        break;
                    }
                }
            }

            debug!(target: "control_client", "SSE stream closed");
        });

        Ok(rx)
    }

    /// Parse a single SSE event from text.
    fn parse_sse_event(text: &str) -> Option<ClientControlEvent> {
        let mut data = None;

        for line in text.lines() {
            if let Some(d) = line.strip_prefix("data: ") {
                data = Some(d);
            }
        }

        let data = data?;

        match serde_json::from_str::<ClientControlEvent>(data) {
            Ok(event) => Some(event),
            Err(e) => {
                warn!(target: "control_client", "Failed to parse SSE event: {} (data: {})", e, data);
                None
            }
        }
    }

    // ============================================================================
    // AI-AGENT TESTING UTILITIES
    // ============================================================================

    /// Wait for the pipeline to reach a specific state.
    pub async fn wait_for_state(
        &self,
        expected_state: &str,
        timeout: Duration,
    ) -> Result<(), ControlClientError> {
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            if tokio::time::Instant::now() > deadline {
                return Err(ControlClientError::WaitTimeout(timeout));
            }

            let state = self.get_state().await?;
            if state.state == expected_state {
                return Ok(());
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Wait for a specific event matching the predicate.
    pub async fn wait_for_event<F>(
        &self,
        predicate: F,
        timeout: Duration,
    ) -> Result<ClientControlEvent, ControlClientError>
    where
        F: Fn(&ClientControlEvent) -> bool,
    {
        let mut rx = self.subscribe_events().await?;
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());

            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Some(event)) => {
                    if predicate(&event) {
                        return Ok(event);
                    }
                }
                Ok(None) => {
                    return Err(ControlClientError::StreamClosed);
                }
                Err(_) => {
                    return Err(ControlClientError::WaitTimeout(timeout));
                }
            }
        }
    }

    /// Poll state until a predicate returns true.
    pub async fn poll_state<F>(
        &self,
        predicate: F,
        timeout: Duration,
        interval: Duration,
    ) -> Result<StateResponse, ControlClientError>
    where
        F: Fn(&StateResponse) -> bool,
    {
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            if tokio::time::Instant::now() > deadline {
                return Err(ControlClientError::WaitTimeout(timeout));
            }

            let state = self.get_state().await?;
            if predicate(&state) {
                return Ok(state);
            }

            tokio::time::sleep(interval).await;
        }
    }

    /// Assert that the current state matches expected value.
    pub async fn assert_state(&self, expected: &str) -> Result<(), ControlClientError> {
        let state = self.get_state().await?;
        if state.state != expected {
            return Err(ControlClientError::StateAssertionFailed {
                expected: expected.to_string(),
                actual: state.state,
            });
        }
        Ok(())
    }

    /// Execute an action and wait for expected state transition.
    ///
    /// This is useful for testing state machine transitions:
    /// - Ensure we're in `from_state`
    /// - Execute the action
    /// - Wait for transition to `to_state`
    pub async fn transaction<F>(
        &self,
        from_state: &str,
        action: F,
        to_state: &str,
        timeout: Duration,
    ) -> Result<(), ControlClientError>
    where
        F: std::future::Future<Output = Result<(), ControlClientError>>,
    {
        // Verify starting state
        self.assert_state(from_state).await?;

        // Execute the action
        action.await?;

        // Wait for destination state
        self.wait_for_state(to_state, timeout).await
    }

    /// Convenience: Send input and wait for processing to complete.
    pub async fn send_input_and_wait(
        &self,
        text: &str,
        timeout: Duration,
    ) -> Result<String, ControlClientError> {
        // Subscribe to events before sending input
        let mut rx = self.subscribe_events().await?;

        // Send the input
        self.send_input(text).await?;

        // Collect the full LLM response
        let mut full_response = String::new();
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());

            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Some(event)) => {
                    match event {
                        ClientControlEvent::LlmToken { token, .. } => {
                            full_response.push_str(&token);
                        }
                        ClientControlEvent::LlmDone { full_text, .. } => {
                            full_response = full_text;
                            return Ok(full_response);
                        }
                        _ => {}
                    }
                }
                Ok(None) => {
                    return Err(ControlClientError::StreamClosed);
                }
                Err(_) => {
                    return Err(ControlClientError::WaitTimeout(timeout));
                }
            }
        }
    }
}

use futures_util::StreamExt;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_client_builder() {
        let client = ControlClient::builder()
            .base_url("http://localhost:9090")
            .timeout(Duration::from_secs(60))
            .build()
            .expect("failed to build client");

        assert_eq!(client.base_url, "http://localhost:9090");
    }

    #[test]
    fn test_parse_sse_event() {
        let event_text = "data: {\"type\": \"state_changed\", \"state\": \"Idle\", \"utterance_id\": null}";
        let event = ControlClient::parse_sse_event(event_text);

        assert!(event.is_some());
        match event.unwrap() {
            ClientControlEvent::StateChanged { state, utterance_id } => {
                assert_eq!(state, "Idle");
                assert_eq!(utterance_id, None);
            }
            _ => panic!("wrong event type"),
        }
    }

    #[test]
    fn test_parse_sse_event_transcript() {
        let event_text = "data: {\"type\": \"transcript\", \"utterance_id\": 42, \"text\": \"Hello\"}";
        let event = ControlClient::parse_sse_event(event_text);

        assert!(event.is_some());
        match event.unwrap() {
            ClientControlEvent::Transcript { utterance_id, text } => {
                assert_eq!(utterance_id, 42);
                assert_eq!(text, "Hello");
            }
            _ => panic!("wrong event type"),
        }
    }
}
