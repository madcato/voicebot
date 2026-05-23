use std::time::Duration;
use voicebot::control_client::{ClientControlEvent, ControlClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = ControlClient::new("http://127.0.0.1:9001").await?;

    println!("Connected to Jarvis control API");

    let health = client.health_check().await?;
    println!("Health check: {:?}", health);

    let state = client.get_state().await?;
    println!(
        "Current state: {} (utterance_id: {:?})",
        state.state, state.utterance_id
    );

    println!("\nSending: 'Search for the last winner of a Formula 1 race'");

    let mut events_rx = client.subscribe_events().await?;

    client
        .send_input("Search for the last winner of a Formula 1 race")
        .await?;
    println!("Input sent\n");

    println!("Listening for events...\n");

    let timeout = Duration::from_secs(60);
    let start = std::time::Instant::now();
    let mut llm_response = String::new();
    let mut complete_response = String::new();
    let mut search_results = String::new();

    while start.elapsed() < timeout {
        match tokio::time::timeout(Duration::from_millis(100), events_rx.recv()).await {
            Ok(Some(event)) => {
                match &event {
                    ClientControlEvent::StateChanged {
                        state,
                        utterance_id,
                    } => {
                        println!("[State] {} (utterance: {:?})", state, utterance_id);
                    }
                    ClientControlEvent::Transcript { utterance_id, text } => {
                        println!("[Transcript] utterance {}: '{}'", utterance_id, text);
                    }
                    ClientControlEvent::LlmToken {
                        utterance_id,
                        token,
                    } => {
                        llm_response.push_str(token);
                        print!("{}", token);
                        std::io::Write::flush(&mut std::io::stdout())?;
                    }
                    ClientControlEvent::LlmDone {
                        utterance_id,
                        full_text,
                    } => {
                        complete_response = full_text.clone();
                        println!(
                            "\n\n[LlmDone] utterance {} - Response complete ({} chars)",
                            utterance_id,
                            full_text.len()
                        );
                    }
                    ClientControlEvent::TtsStart { utterance_id } => {
                        println!(
                            "\n[TtsStart] utterance {} - Starting speech synthesis",
                            utterance_id
                        );
                    }
                    ClientControlEvent::ToolCall { name, result } => {
                        println!("\n[ToolCall] {} executed", name);
                        if name == "web_search" {
                            search_results = result.clone();
                            println!(
                                "Search results preview: {}...",
                                &result[..result.len().min(200)]
                            );
                        }
                    }
                    ClientControlEvent::MuteChanged { muted } => {
                        println!("[MuteChanged] muted = {}", muted);
                    }
                    ClientControlEvent::Error { message } => {
                        println!("[Error] {}", message);
                    }
                }

                if !complete_response.is_empty() && !search_results.is_empty() {
                    println!("\nTest complete! Formula 1 search results received.");
                    println!("\nSummary:");
                    println!("   - Search executed successfully");
                    println!("   - LLM response: {} characters", complete_response.len());
                    println!("   - Tool result: {} characters", search_results.len());

                    if complete_response.to_lowercase().contains("verstappen")
                        || complete_response.to_lowercase().contains("hamilton")
                        || complete_response.to_lowercase().contains("pérez")
                        || complete_response.to_lowercase().contains("leclerc")
                        || complete_response.to_lowercase().contains("norris")
                    {
                        println!("   Driver names detected in response");
                    } else {
                        println!(
                            "   No specific driver names found - may need to check search results"
                        );
                    }

                    return Ok(());
                }
            }
            Ok(None) => {
                println!("Event stream closed");
                break;
            }
            Err(_) => {
                if !complete_response.is_empty() {
                    println!("\nTimeout waiting for additional events, but response was received");
                    println!("\nPartial Results:");
                    println!("   - LLM response: {} characters", complete_response.len());
                    println!(
                        "   - Response: {}",
                        &complete_response[..complete_response.len().min(300)]
                    );
                    return Ok(());
                }
            }
        }
    }

    if llm_response.is_empty() {
        println!("\nNo response received within timeout");
        return Err(anyhow::anyhow!("Test failed - no LLM response"));
    }

    Ok(())
}
