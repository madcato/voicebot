use anyhow::{Context, Result};
use async_channel::{Receiver, Sender};
use futures_util::{SinkExt, StreamExt};
use rand;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};
use url::Url;

use crate::audio_transform::TransformedAudio;

const RECONNECT_DELAY_MS: u64 = 2000;
const MAX_RECONNECT_DELAY_MS: u64 = 16000;
const CONNECTION_TIMEOUT_MS: u64 = 10000;

pub struct WebSocketClient {
    url: String,
    name: String,
    connection_status_tx: Option<Sender<bool>>,
}

impl WebSocketClient {
    /// Creates a new WebSocketClient
    /// 
    /// Validates that the URL is a valid WebSocket URL (ws:// or wss://)
    /// Optionally accepts a sender to notify when a connection is successfully established
    pub fn new(url: String, name: String, connection_status_tx: Option<Sender<bool>>) -> Self {
        // URL will be validated before use, but log a warning if it doesn't look valid
        if !url.starts_with("ws://") && !url.starts_with("wss://") {
            warn!("[{}] WebSocket URL '{}' does not start with ws:// or wss://", name, url);
        }
        
        if let Err(e) = Url::parse(&url) {
            warn!("[{}] Invalid WebSocket URL '{}': {}", name, url, e);
        }
        
        Self { url, name, connection_status_tx }
    }

    /// Start the WebSocket client, reading from the provided channel
    /// and sending audio data to the remote service
    pub async fn run(&self, rx: Receiver<TransformedAudio>) -> Result<()> {
        let mut reconnect_delay = RECONNECT_DELAY_MS;
        let mut successful_connection = false;
        let mut connection_attempts = 0;

        loop {
            connection_attempts += 1;
            info!(
                "[{}] Connection attempt #{} to {}",
                self.name, connection_attempts, self.url
            );
            
            match self.connect_and_stream(&rx).await {
                Ok(true) => {
                    // Connection was established and user requested to stop
                    info!("[{}] Connection closed gracefully by user request", self.name);
                    break;
                }
                Ok(false) => {
                    // Connection was established but closed for other reasons (server disconnect, etc.)
                    if !successful_connection {
                        info!("[{}] First successful connection established", self.name);
                        successful_connection = true;
                        
                        // Notify about successful connection if a channel is available
                        if let Some(tx) = &self.connection_status_tx {
                            if let Err(e) = tx.send(true).await {
                                warn!("[{}] Failed to send connection notification: {}", self.name, e);
                            } else {
                                debug!("[{}] Sent successful connection notification", self.name);
                            }
                        }
                    }
                    
                    // Reset reconnect delay since we had a successful connection
                    reconnect_delay = RECONNECT_DELAY_MS;
                    info!("[{}] Connection closed, reconnecting in {}ms...", self.name, reconnect_delay);
                    sleep(Duration::from_millis(reconnect_delay)).await;
                }
                Err(e) => {
                    // Connection error, use exponential backoff
                    let error_msg = e.to_string();
                    let is_dns_error = error_msg.contains("dns error") || 
                                      error_msg.contains("nodename nor servname provided");
                    let is_connection_refused = error_msg.contains("connection refused") ||
                                               error_msg.contains("Connection refused");
                    
                    if is_dns_error {
                        error!(
                            "[{}] DNS resolution error for '{}': {}. Reconnecting in {}ms...",
                            self.name, self.url, e, reconnect_delay
                        );
                    } else if is_connection_refused {
                        error!(
                            "[{}] Connection refused for '{}': Server is not running or not accessible. Reconnecting in {}ms...",
                            self.name, self.url, reconnect_delay
                        );
                    } else {
                        error!(
                            "[{}] Connection error: {}. Reconnecting in {}ms...",
                            self.name, e, reconnect_delay
                        );
                    }
                    
                    sleep(Duration::from_millis(reconnect_delay)).await;

                    // Exponential backoff with a bit of jitter to avoid thundering herd
                    let jitter = (rand::random::<u64>() % 1000) as u64;
                    reconnect_delay = ((reconnect_delay * 2) + jitter).min(MAX_RECONNECT_DELAY_MS);
                }
            }
        }

        Ok(())
    }

    /// Connect to WebSocket server and stream audio data
    /// 
    /// Returns:
    /// - Ok(true) if connection was closed by user request (channel closed)
    /// - Ok(false) if connection closed for other reasons (server disconnect)
    /// - Err if there was an error establishing or maintaining the connection
    async fn connect_and_stream(&self, rx: &Receiver<TransformedAudio>) -> Result<bool> {
        info!("[{}] Connecting to {} (timeout: {}ms)", self.name, self.url, CONNECTION_TIMEOUT_MS);
        
        // Apply timeout to the connection attempt
        let connection_result = timeout(
            Duration::from_millis(CONNECTION_TIMEOUT_MS),
            connect_async(&self.url)
        ).await;

        // Handle timeout separately from connection errors
        let (ws_stream, response) = match connection_result {
            Ok(result) => result.context("Failed to connect to WebSocket")?,
            Err(_) => {
                return Err(anyhow::anyhow!("Connection timeout after {}ms", CONNECTION_TIMEOUT_MS).into());
            }
        };

        info!(
            "[{}] Connected! Response status: {}",
            self.name,
            response.status()
        );

        // Notify about successful connection if a channel is available
        if let Some(tx) = &self.connection_status_tx {
            if let Err(e) = tx.send(true).await {
                warn!("[{}] Failed to send connection notification: {}", self.name, e);
            } else {
                debug!("[{}] Sent successful connection notification", self.name);
            }
        }

        let (mut write, mut read) = ws_stream.split();

        // Spawn a task to handle incoming messages (for potential responses/acks)
        let name_clone = self.name.clone();
        let read_task = tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        debug!("[{}] Received text message: {}", name_clone, text);
                    }
                    Ok(Message::Binary(data)) => {
                        debug!("[{}] Received binary message: {} bytes", name_clone, data.len());
                    }
                    Ok(Message::Ping(_data)) => {
                        debug!("[{}] Received ping", name_clone);
                    }
                    Ok(Message::Pong(_)) => {
                        debug!("[{}] Received pong", name_clone);
                    }
                    Ok(Message::Close(frame)) => {
                        info!("[{}] Received close frame: {:?}", name_clone, frame);
                        break;
                    }
                    Ok(Message::Frame(_)) => {}
                    Err(e) => {
                        error!("[{}] Error receiving message: {}", name_clone, e);
                        break;
                    }
                }
            }
        });

        // Send audio data
        loop {
            tokio::select! {
                result = rx.recv() => {
                    match result {
                        Ok(audio) => {
                            // Send audio data as binary WebSocket message
                            match write.send(Message::Binary(audio.data.into())).await {
                                Ok(_) => {
                                    debug!("[{}] Sent audio chunk", self.name);
                                },
                                Err(e) => {
                                    error!("[{}] Failed to send audio data: {}", self.name, e);
                                    // Connection error, don't consider this user-initiated
                                    let _ = write.send(Message::Close(None)).await;
                                    read_task.abort();
                                    return Ok(false);
                                }
                            }
                        }
                        Err(e) => {
                            warn!("[{}] Channel closed: {}", self.name, e);
                            // This is considered a user-initiated closure (rx channel closed)
                            // Clean up the connection and return true
                            let _ = write.send(Message::Close(None)).await;
                            read_task.abort();
                            return Ok(true);
                        }
                    }
                }
                _ = tokio::time::sleep(Duration::from_secs(30)) => {
                    // Send ping to keep connection alive
                    match write.send(Message::Ping(vec![].into())).await {
                        Ok(_) => {
                            debug!("[{}] Sent ping", self.name);
                        },
                        Err(e) => {
                            error!("[{}] Failed to send ping: {}", self.name, e);
                            // Connection error, don't consider this user-initiated
                            // let _ = write.send(Message::Close(None)).await;
                            // read_task.abort();
                            // return Ok(false);
                        }
                    }
                }
            }
        }

        // Clean up
        // let _ = write.send(Message::Close(None)).await;
        // read_task.abort();

        // // Return false to indicate this was not a user-initiated closure
        // Ok(false)
    }
}
