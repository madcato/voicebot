//! Debug/test TUI chat with an ACP agent via JSON-RPC 2.0 over stdio.
//!
//! Supports slash commands for testing the full ACP protocol surface.
//! Uses the same configuration as the main voicebot binary (`.env` / env vars).
//! Relevant config: `AGENT_ACP_COMMAND` (default "hermes acp").
//!
//! Run: `cargo run --bin acp_agent_chat`

use std::io::{self, Write as _};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{mpsc, Mutex};
use tracing::debug;
use tracing_subscriber::EnvFilter;

// Re-use library modules from the voicebot crate.
use voicebot::config::Config;
use voicebot::tools::run_agent::{AcpWriter, JsonRpcMessage};

/// How permission requests from the agent are handled.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PermissionMode {
    /// Automatically allow all permissions (default).
    Auto,
    /// Prompt the user in the terminal before responding.
    Ask,
    /// Automatically deny all permissions.
    Deny,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();

    dotenvy::dotenv().ok();
    let config = Config::from_env()?;

    let acp_command = &config.agent_acp_command;
    eprintln!("Spawning ACP agent: {acp_command}");

    let (mut writer, mut rx) = AcpWriter::spawn(acp_command).await?;

    let cwd = std::env::current_dir()?
        .to_string_lossy()
        .to_string();

    let session_id = writer.initialize(&mut rx, &cwd).await?;
    eprintln!("Session initialized: {session_id}\n");

    let writer = Arc::new(Mutex::new(writer));
    let mut current_session_id = session_id;
    let mut permission_mode = PermissionMode::Auto;
    let mut last_prompt_id: Option<u64> = None;
    let start_time = std::time::Instant::now();
    let mut message_count: u64 = 0;

    println!("╔══════════════════════════════════════════════════╗");
    println!("║          Hermes Agent Chat (ACP)                ║");
    println!("║  Type your message and press Enter.             ║");
    println!("║  Type /help for commands, /quit to exit.        ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    loop {
        // Prompt
        print!("\x1b[1;34mYou>\x1b[0m ");
        io::stdout().flush()?;

        // Read user input
        let mut input = String::new();
        let n = io::stdin().read_line(&mut input)?;
        if n == 0 {
            println!("\nBye!");
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // ── Slash command dispatch ──────────────────────────────────────────
        if input.starts_with('/') {
            let parts: Vec<&str> = input.splitn(2, ' ').collect();
            let cmd = parts[0];
            let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

            match cmd {
                "/quit" | "/exit" => {
                    println!("Bye!");
                    break;
                }

                "/help" => {
                    print_help();
                }

                "/verbose" => {
                    let w = writer.lock().await;
                    let prev = w.verbose.load(Ordering::Relaxed);
                    w.verbose.store(!prev, Ordering::Relaxed);
                    let state = if !prev { "ON" } else { "OFF" };
                    println!("Verbose mode: {state}");
                }

                "/cancel" => {
                    if let Some(pid) = last_prompt_id {
                        let mut w = writer.lock().await;
                        w.send_cancel(pid).await?;
                        println!("Cancel sent for request id={pid}");
                    } else {
                        println!("No prompt in flight to cancel.");
                    }
                }

                "/session" => {
                    let uptime = start_time.elapsed();
                    println!("Session ID: {current_session_id}");
                    println!("Messages:   {message_count}");
                    println!("Uptime:     {}m {}s", uptime.as_secs() / 60, uptime.as_secs() % 60);
                    let w = writer.lock().await;
                    let verbose = w.verbose.load(Ordering::Relaxed);
                    println!("Verbose:    {}", if verbose { "ON" } else { "OFF" });
                    println!("Permissions: {:?}", permission_mode);
                }

                "/sessions" => {
                    let req_id = {
                        let mut w = writer.lock().await;
                        w.send_list_sessions(&cwd).await?
                    };
                    println!("Listing sessions (request id={req_id})...");
                    let response = wait_for_response(&mut rx, req_id).await;
                    println!("{response}");
                }

                "/new" => {
                    let req_id = {
                        let mut w = writer.lock().await;
                        w.send_new_session(&cwd).await?
                    };
                    match wait_for_session_id(&mut rx, req_id).await {
                        Ok(sid) => {
                            let mut w = writer.lock().await;
                            w.session_id = Some(sid.clone());
                            current_session_id = sid;
                            message_count = 0;
                            println!("New session: {current_session_id}");
                        }
                        Err(e) => println!("[Error: {e}]"),
                    }
                }

                "/fork" => {
                    let req_id = {
                        let mut w = writer.lock().await;
                        w.send_fork_session(&current_session_id, &cwd).await?
                    };
                    match wait_for_session_id(&mut rx, req_id).await {
                        Ok(sid) => {
                            let mut w = writer.lock().await;
                            w.session_id = Some(sid.clone());
                            current_session_id = sid;
                            println!("Forked to new session: {current_session_id}");
                        }
                        Err(e) => println!("[Error: {e}]"),
                    }
                }

                "/load" => {
                    if arg.is_empty() {
                        println!("Usage: /load <session_id>");
                    } else {
                        let req_id = {
                            let mut w = writer.lock().await;
                            w.send_load_session(arg, &cwd).await?
                        };
                        match wait_for_session_id(&mut rx, req_id).await {
                            Ok(sid) => {
                                let mut w = writer.lock().await;
                                w.session_id = Some(sid.clone());
                                current_session_id = sid;
                                message_count = 0;
                                println!("Loaded session: {current_session_id}");
                            }
                            Err(e) => println!("[Error: {e}]"),
                        }
                    }
                }

                "/resume" => {
                    if arg.is_empty() {
                        println!("Usage: /resume <session_id>");
                    } else {
                        let req_id = {
                            let mut w = writer.lock().await;
                            w.send_resume_session(arg, &cwd).await?
                        };
                        match wait_for_session_id(&mut rx, req_id).await {
                            Ok(sid) => {
                                let mut w = writer.lock().await;
                                w.session_id = Some(sid.clone());
                                current_session_id = sid;
                                println!("Resumed session: {current_session_id}");
                            }
                            Err(e) => println!("[Error: {e}]"),
                        }
                    }
                }

                "/permissions" => {
                    match arg {
                        "auto" => {
                            permission_mode = PermissionMode::Auto;
                            println!("Permission mode: Auto (allow all)");
                        }
                        "ask" => {
                            permission_mode = PermissionMode::Ask;
                            println!("Permission mode: Ask (prompt before responding)");
                        }
                        "deny" => {
                            permission_mode = PermissionMode::Deny;
                            println!("Permission mode: Deny (reject all)");
                        }
                        _ => {
                            println!("Usage: /permissions <auto|ask|deny>");
                            println!("Current: {:?}", permission_mode);
                        }
                    }
                }

                "/raw" => {
                    if arg.is_empty() {
                        println!("Usage: /raw <json>");
                    } else {
                        match serde_json::from_str::<serde_json::Value>(arg) {
                            Ok(json) => {
                                let mut w = writer.lock().await;
                                match w.write_json(&json).await {
                                    Ok(()) => println!("Sent."),
                                    Err(e) => println!("[Send error: {e}]"),
                                }
                            }
                            Err(e) => println!("[Invalid JSON: {e}]"),
                        }
                    }
                }

                _ => {
                    println!("Unknown command: {cmd}. Type /help for available commands.");
                }
            }
            println!();
            continue;
        }

        // ── Regular prompt ──────────────────────────────────────────────────
        let prompt_id = {
            let mut w = writer.lock().await;
            w.send_prompt(&current_session_id, input).await?
        };
        last_prompt_id = Some(prompt_id);
        message_count += 1;

        // Collect response
        print!("\x1b[1;32mHermes>\x1b[0m ");
        io::stdout().flush()?;

        let response = collect_response(&writer, &mut rx, prompt_id, permission_mode).await;
        println!("{response}");
        println!();
    }

    // Kill the ACP process
    let mut w = writer.lock().await;
    w.kill().await;

    Ok(())
}

fn print_help() {
    println!("Available commands:");
    println!("  /help                        — Show this help");
    println!("  /verbose                     — Toggle raw JSON-RPC message logging");
    println!("  /session                     — Show current session info");
    println!("  /sessions                    — List active sessions  [unstable]");
    println!("  /new                         — Start a fresh session");
    println!("  /fork                        — Fork current session  [unstable]");
    println!("  /load <session_id>           — Load a previous session");
    println!("  /resume <session_id>         — Resume a suspended session  [unstable]");
    println!("  /cancel                      — Cancel current operation");
    println!("  /permissions <auto|ask|deny> — Change permission handling mode");
    println!("  /raw <json>                  — Send raw JSON-RPC message");
    println!("  /quit                        — Exit");
    println!();
    println!("  [unstable] — Requires hermes-agent ACP adapter with use_unstable_protocol=True.");
    println!("  This is now enabled by default in the patched entry.py.");
}

/// Wait for a response to a specific request ID and return the raw result.
async fn wait_for_response(
    rx: &mut mpsc::Receiver<JsonRpcMessage>,
    request_id: u64,
) -> String {
    loop {
        match rx.recv().await {
            Some(JsonRpcMessage::Response { id, result, error }) if id == request_id => {
                if let Some(err) = error {
                    return format!("[Error: {err}]");
                }
                return serde_json::to_string_pretty(&result.unwrap_or_default())
                    .unwrap_or_else(|_| "(empty)".to_string());
            }
            Some(_) => continue,
            None => return "[ACP process closed unexpectedly]".to_string(),
        }
    }
}

/// Wait for a response that contains a sessionId field.
async fn wait_for_session_id(
    rx: &mut mpsc::Receiver<JsonRpcMessage>,
    request_id: u64,
) -> Result<String> {
    loop {
        match rx.recv().await {
            Some(JsonRpcMessage::Response { id, result, error }) if id == request_id => {
                if let Some(err) = error {
                    anyhow::bail!("{err}");
                }
                let result = result.unwrap_or_default();
                let sid = result["sessionId"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("Response missing sessionId: {result}"))?
                    .to_string();
                return Ok(sid);
            }
            Some(_) => continue,
            None => anyhow::bail!("ACP process closed unexpectedly"),
        }
    }
}

/// Collect the full ACP response for a single prompt request.
///
/// Handles streaming `session/update` notifications and permission requests.
/// Returns the accumulated text when the prompt response arrives.
async fn collect_response(
    writer: &Arc<Mutex<AcpWriter>>,
    rx: &mut mpsc::Receiver<JsonRpcMessage>,
    prompt_request_id: u64,
    permission_mode: PermissionMode,
) -> String {
    let mut accumulated_text = String::new();
    let mut progress: Vec<String> = Vec::new();

    loop {
        let msg = match rx.recv().await {
            Some(m) => m,
            None => return "[ACP process closed unexpectedly]".to_string(),
        };

        match msg {
            // Prompt response → done
            JsonRpcMessage::Response { id, result, error } if id == prompt_request_id => {
                if let Some(err) = error {
                    return format!("[Error: {err}]");
                }
                let stop_reason = result
                    .as_ref()
                    .and_then(|r| r["stopReason"].as_str())
                    .unwrap_or("unknown");
                debug!("Prompt complete, stopReason={stop_reason}");

                if accumulated_text.is_empty() && !progress.is_empty() {
                    return format!("[Progreso: {}]", progress.join("; "));
                }
                if !accumulated_text.is_empty() && !progress.is_empty() {
                    return format!(
                        "{}\n\n[Progreso: {}]",
                        accumulated_text.trim(),
                        progress.join("; ")
                    );
                }
                if accumulated_text.is_empty() {
                    return format!("[Agent finished with stopReason={stop_reason}]");
                }
                return accumulated_text.trim().to_string();
            }

            // Streaming content
            JsonRpcMessage::Notification { method, params } if method == "session/update" => {
                let params = params.unwrap_or_default();
                let update = &params["update"];
                let session_update = update["sessionUpdate"].as_str().unwrap_or("");

                match session_update {
                    "agent_message_chunk" => {
                        if let Some(text) = update["content"]["text"].as_str() {
                            accumulated_text.push_str(text);
                        }
                    }
                    "agent_thought_chunk" => {
                        // silently skip thoughts
                    }
                    "tool_call" => {
                        let tool_name = update["name"].as_str().unwrap_or("unknown");
                        eprint!("\x1b[2m[using {tool_name}...]\x1b[0m ");
                        let _ = io::stderr().flush();
                        progress.push(format!("using {tool_name}"));
                    }
                    "tool_call_update" | "tool_result" => {}
                    _ => {
                        debug!("Ignored session update: {session_update}");
                    }
                }
            }

            // Permission request
            JsonRpcMessage::Request { id, method, params }
                if method == "session/request_permission" =>
            {
                let params = params.unwrap_or_default();
                let tool_name = params["toolCall"]["name"]
                    .as_str()
                    .unwrap_or("unknown");

                let option_id = match permission_mode {
                    PermissionMode::Auto => {
                        eprint!("\x1b[33m[auto-allowing: {tool_name}]\x1b[0m ");
                        let _ = io::stderr().flush();
                        find_allow_option(&params)
                    }
                    PermissionMode::Deny => {
                        eprint!("\x1b[31m[auto-denying: {tool_name}]\x1b[0m ");
                        let _ = io::stderr().flush();
                        None
                    }
                    PermissionMode::Ask => {
                        eprintln!();
                        eprintln!("\x1b[33m[Permission requested: {tool_name}]\x1b[0m");

                        // Show tool call details
                        if let Some(input) = params["toolCall"]["input"].as_str() {
                            let preview = if input.len() > 200 {
                                format!("{}...", &input[..200])
                            } else {
                                input.to_string()
                            };
                            eprintln!("  Input: {preview}");
                        }

                        // Show options
                        let options: Vec<String> = params["options"]
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|o| o["optionId"].as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        eprintln!("  Options: {}", options.join(", "));

                        eprint!("  Choose (or press Enter for first): ");
                        let _ = io::stderr().flush();

                        let mut choice = String::new();
                        let _ = io::stdin().read_line(&mut choice);
                        let choice = choice.trim();

                        if choice.is_empty() {
                            // Use the allow option or first option
                            find_allow_option(&params)
                        } else if options.contains(&choice.to_string()) {
                            Some(choice.to_string())
                        } else {
                            eprintln!("  Invalid choice, cancelling.");
                            None
                        }
                    }
                };

                let result = if let Some(oid) = option_id {
                    serde_json::json!({"outcome": "selected", "optionId": oid})
                } else {
                    serde_json::json!({"outcome": "cancelled"})
                };

                let mut w = writer.lock().await;
                let _ = w.send_response(id, result).await;
            }

            // Unrelated response
            JsonRpcMessage::Response { .. } => {}

            // Other notifications/requests
            _ => {}
        }
    }
}

/// Find the "allow" option in a permission request's options array.
fn find_allow_option(params: &serde_json::Value) -> Option<String> {
    params["options"]
        .as_array()
        .and_then(|arr| {
            arr.iter()
                .find_map(|o| {
                    let oid = o["optionId"].as_str()?;
                    if oid == "allow" || o["kind"].as_str() == Some("allow") {
                        Some(oid.to_string())
                    } else {
                        None
                    }
                })
                .or_else(|| {
                    arr.first()
                        .and_then(|o| o["optionId"].as_str())
                        .map(String::from)
                })
        })
}
