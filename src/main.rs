mod audio_capture;
mod audio_transform;
mod config;
mod websocket_client;

use anyhow::Result;
use async_channel::{bounded, Sender};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use audio_capture::{AudioCapture, AudioChunk};
use audio_transform::{AudioTransformer, TransformedAudio};
use config::Config;
use websocket_client::WebSocketClient;

const AUDIO_CHANNEL_CAPACITY: usize = 100;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    info!("Starting microphone streamer...");

    // Load configuration from environment
    let config = Config::from_env()?;
    
    // Handle list devices flag
    if config.list_devices {
        AudioCapture::print_devices()?;
        return Ok(());
    }
    
    info!("Configuration loaded: {:?}", config);

    // Initialize audio capture with optional device selection
    let audio_capture = AudioCapture::new(config.audio_device.as_deref())?;
    let source_sample_rate = audio_capture.sample_rate();
    let source_channels = audio_capture.channels();

    info!(
        "Audio source: {} Hz, {} channels",
        source_sample_rate, source_channels
    );

    // Create channels for audio data flow
    let (raw_tx, raw_rx) = bounded::<AudioChunk>(AUDIO_CHANNEL_CAPACITY);
    // let (vad_tx, vad_rx) = bounded::<TransformedAudio>(AUDIO_CHANNEL_CAPACITY);
    let (stt_tx, stt_rx) = bounded::<TransformedAudio>(AUDIO_CHANNEL_CAPACITY);
    
    // Create a channel for connection status notification
    let (connection_status_tx, connection_status_rx) = bounded::<bool>(1);

    // Start audio capture
    let samples_per_chunk = config.samples_per_chunk();
    let _stream = audio_capture.start_capture(raw_tx, samples_per_chunk)?;

    // Spawn WebSocket client tasks first
    // let vad_client = WebSocketClient::new(config.vad_ws_url.clone(), "VAD".to_string(), None);

    // let vad_handle = tokio::spawn(async move {
    //     if let Err(e) = vad_client.run(vad_rx).await {
    //         error!("VAD client error: {}", e);
    //     }
    // });

    let stt_client = WebSocketClient::new(
        config.stt_ws_url.clone(), 
        "STT".to_string(), 
        Some(connection_status_tx)
    );
    
    let stt_handle = tokio::spawn(async move {
        if let Err(e) = stt_client.run(stt_rx).await {
            error!("STT client error: {}", e);
        }
    });
    
    // Spawn audio transformation task - only starts after WebSocket connection succeeds
    let transform_config = config.clone();
    let transform_handle = tokio::spawn(async move {
        run_transformer_after_connection(
            transform_config, 
            source_sample_rate, 
            source_channels, 
            raw_rx, 
            stt_tx,
            connection_status_rx
        ).await
    });

    // Handle shutdown signals
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
        result = transform_handle => {
            if let Err(e) = result {
                error!("Transform task error: {}", e);
            }
        }
        // result = vad_handle => {
        //     if let Err(e) = result {
        //         error!("VAD task error: {}", e);
        //     }
        // }
        result = stt_handle => {
            if let Err(e) = result {
                error!("STT task error: {}", e);
            }
        }
    }

    info!("Shutting down...");
    Ok(())
}

async fn run_transformer_after_connection(
    config: Config,
    source_sample_rate: u32,
    source_channels: u16,
    raw_rx: async_channel::Receiver<AudioChunk>,
    // vad_tx: Sender<TransformedAudio>,
    stt_tx: Sender<TransformedAudio>,
    connection_status_rx: async_channel::Receiver<bool>,
) {
    // Wait for connection to be established first
    info!("Waiting for WebSocket connection to be established before starting transformer...");
    
    match connection_status_rx.recv().await {
        Ok(true) => {
            info!("WebSocket connection established, starting audio transformer");
            run_transformer(config, source_sample_rate, source_channels, raw_rx, stt_tx).await;
        },
        Ok(false) => {
            error!("WebSocket connection failed, not starting audio transformer");
            return;
        },
        Err(e) => {
            error!("Failed to receive connection status: {}", e);
            return;
        }
    }
}

async fn run_transformer(
    config: Config,
    source_sample_rate: u32,
    source_channels: u16,
    raw_rx: async_channel::Receiver<AudioChunk>,
    // vad_tx: Sender<TransformedAudio>,
    stt_tx: Sender<TransformedAudio>,
) {
    let mut transformer = match AudioTransformer::new(&config, source_sample_rate, source_channels) {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to create audio transformer: {}", e);
            return;
        }
    };

    info!("Audio transformer initialized");

    while let Ok(chunk) = raw_rx.recv().await {
        match transformer.transform(chunk) {
            Ok(transformed) => {
                // Send to both VAD and STT services
                // Clone the data since we need to send to two destinations
                // let vad_audio = transformed.clone();
                let stt_audio = transformed;

                // if let Err(e) = vad_tx.try_send(vad_audio) {
                //     error!("Failed to send to VAD channel: {}", e);
                // }

                if let Err(e) = stt_tx.try_send(stt_audio) {
                    error!("Failed to send to STT channel: {}", e);
                }
            }
            Err(e) => {
                error!("Audio transformation error: {}", e);
            }
        }
    }

    info!("Transformer task exiting");
}
