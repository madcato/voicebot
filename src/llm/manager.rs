use anyhow::{Context, Result};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

const MAX_RESTARTS: u32 = 3;
const READY_POLL_INTERVAL_MS: u64 = 1000;
const READY_TIMEOUT_SECS: u64 = 120;

pub async fn start_and_wait_ready(command: &str, health_url: &str) -> Result<Child> {
    let child = spawn_process(command)?;
    wait_until_ready(health_url).await?;
    Ok(child)
}

pub async fn supervise(
    mut child: Child,
    command: String,
    health_url: String,
    notify_tx: mpsc::Sender<String>,
) {
    let mut restart_count: u32 = 0;

    loop {
        match child.wait().await {
            Ok(status) => {
                warn!(target: "llm_manager", "LLM server exited (status={status})");
            }
            Err(e) => {
                warn!(target: "llm_manager", "LLM server wait error: {e}");
            }
        }

        restart_count += 1;
        if restart_count > MAX_RESTARTS {
            let msg = format!(
                "LLM server crashed {} times. Please check the logs and restart voicebot.",
                restart_count
            );
            error!(target: "llm_manager", "{}", msg);
            let _ = notify_tx.send(msg).await;
            break;
        }

        warn!(
            target: "llm_manager",
            "Restarting LLM server (attempt {}/{})",
            restart_count, MAX_RESTARTS
        );

        match spawn_process(&command) {
            Ok(new_child) => match wait_until_ready(&health_url).await {
                Ok(()) => {
                    info!(target: "llm_manager", "LLM server ready after restart {restart_count}");
                    child = new_child;
                }
                Err(e) => {
                    error!(target: "llm_manager", "LLM server did not become ready after restart {restart_count}: {e}");
                    // count this as another failure on next loop iteration
                    child = new_child;
                }
            },
            Err(e) => {
                error!(target: "llm_manager", "Failed to respawn LLM server: {e}");
                // treat as immediate exit on next iteration
                restart_count += 1;
                if restart_count > MAX_RESTARTS {
                    let msg =
                        "LLM server could not be restarted. Please restart voicebot.".to_string();
                    error!(target: "llm_manager", "{}", msg);
                    let _ = notify_tx.send(msg).await;
                    break;
                }
            }
        }
    }
}

fn spawn_process(command: &str) -> Result<Child> {
    let parts: Vec<&str> = command.split_whitespace().collect();
    let program = parts.first().context("LLM_COMMAND is empty")?;
    let args = &parts[1..];

    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("voicebot.log")
        .map(std::process::Stdio::from)
        .unwrap_or_else(|_| std::process::Stdio::null());

    let child = Command::new(program)
        .args(args)
        .stdout(log_file)
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("Failed to spawn LLM server: {command}"))?;

    info!(target: "llm_manager", "LLM server process started: {command}");
    Ok(child)
}

async fn wait_until_ready(health_url: &str) -> Result<()> {
    let models_url = format!("{}/v1/models", health_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()?;

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(READY_TIMEOUT_SECS);

    info!(target: "llm_manager", "Waiting for LLM server to be ready at {models_url}...");

    loop {
        if std::time::Instant::now() >= deadline {
            anyhow::bail!(
                "LLM server did not respond within {}s at {models_url}",
                READY_TIMEOUT_SECS
            );
        }

        match client.get(&models_url).send().await {
            Ok(resp) if resp.status().is_success() || resp.status().as_u16() == 404 => {
                info!(target: "llm_manager", "LLM server is ready");
                return Ok(());
            }
            _ => {
                tokio::time::sleep(std::time::Duration::from_millis(READY_POLL_INTERVAL_MS)).await;
            }
        }
    }
}
