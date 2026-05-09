use std::time::Duration;
use voicebot::control_client::ControlClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = ControlClient::new("http://127.0.0.1:8080").await?;

    client.health_check().await?;
    println!("Connected to voicebot control API");

    let state = client.get_state().await?;
    println!("Current state: {}", state.state);

    client.send_input("Hello, Jarvis!").await?;
    println!("Sent text input");

    let response = client
        .send_input_and_wait("What is the weather?", Duration::from_secs(30))
        .await?;
    println!("Response: {}", response);

    Ok(())
}
