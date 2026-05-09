use std::time::Duration;
use voicebot::control_client::{ControlClient, ClientControlEvent, ControlClientError};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = ControlClient::new("http://127.0.0.1:9001").await?;
    
    println!("Control Client Test - Formula 1 Search");
    println!("=====================================\n");
    
    println!("This test demonstrates how an AI agent would:");
    println!("1. Connect to Jarvis control API");
    println!("2. Send text input: 'Search for the last winner of a Formula 1 race'");
    println!("3. Monitor events for search execution and response\n");
    
    match client.health_check().await {
        Ok(health) => {
            println!("Health check response: {:?}", health);
        }
        Err(e) => {
            println!("Note: Jarvis voicebot is not running");
            println!("Error: {:?}\n", e);
            println!("To run this test against a real Jarvis instance:");
            println!("  1. Ensure you have a microphone and speaker connected");
            println!("  2. Ensure LLM server is running on port 8080");
            println!("  3. Run: CONTROL_PORT=9001 cargo run --release --features control,avspeech");
            println!("  4. Then run this example again");
            return Ok(());
        }
    }
    
    let state = client.get_state().await?;
    println!("Jarvis state: {}", state.state);
    println!("TTS muted: {}\n", state.tts_muted);
    
    println!("Sending: 'Search for the last winner of a Formula 1 race'");
    client.send_input("Search for the last winner of a Formula 1 race").await?;
    println!("Input sent successfully\n");
    
    let mut events = client.subscribe_events().await?;
    println!("Subscribed to events. Waiting for responses...\n");
    
    let start = std::time::Instant::now();
    let mut token_count = 0;
    let mut response_complete = false;
    
    while start.elapsed() < Duration::from_secs(30) && !response_complete {
        match tokio::time::timeout(Duration::from_millis(500), events.recv()).await {
            Ok(Some(event)) => {
                match event {
                    ClientControlEvent::Transcript { utterance_id, text } => {
                        println!("[STT] Utterance {}: '{}'", utterance_id, text);
                    }
                    ClientControlEvent::ToolCall { name, result } => {
                        println!("\n[Tool] {} executed", name);
                        if name == "web_search" {
                            let preview = if result.len() > 200 {
                                format!("{}...", &result[..200])
                            } else {
                                result.clone()
                            };
                            println!("Search results: {}\n", preview);
                        }
                    }
                    ClientControlEvent::LlmToken { token, .. } => {
                        token_count += 1;
                        print!("{}", token);
                        if token_count % 50 == 0 {
                            println!();
                        }
                    }
                    ClientControlEvent::LlmDone { full_text, .. } => {
                        println!("\n\n[Complete] LLM response received ({} tokens streamed)", token_count);
                        println!("Response length: {} characters\n", full_text.len());
                        
                        let text_lower = full_text.to_lowercase();
                        if text_lower.contains("verstappen") {
                            println!("Detected: Max Verstappen");
                        } else if text_lower.contains("hamilton") {
                            println!("Detected: Lewis Hamilton");
                        } else if text_lower.contains("pérez") || text_lower.contains("perez") {
                            println!("Detected: Sergio Pérez");
                        } else if text_lower.contains("leclerc") {
                            println!("Detected: Charles Leclerc");
                        } else if text_lower.contains("norris") {
                            println!("Detected: Lando Norris");
                        }
                        
                        response_complete = true;
                    }
                    ClientControlEvent::TtsStart { utterance_id } => {
                        println!("\n[TTS] Starting speech synthesis for utterance {}", utterance_id);
                    }
                    ClientControlEvent::Error { message } => {
                        println!("[Error] {}", message);
                    }
                    _ => {}
                }
            }
            Ok(None) => {
                println!("Event stream closed");
                break;
            }
            Err(_) => {
                print!(".");
            }
        }
    }
    
    if !response_complete {
        println!("\nTimeout waiting for complete response");
        if token_count > 0 {
            println!("Partial response received ({} tokens)", token_count);
        }
    } else {
        println!("\nTest completed successfully!");
    }
    
    Ok(())
}
