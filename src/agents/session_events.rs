use serde_json::Value;

/// Events extracted from `session/update` ACP notifications.
#[derive(Clone, Debug)]
pub enum AcpSessionEvent {
    AgentMessageChunk(String),
    AgentThoughtChunk(String),
    ToolCall { name: String },
    ToolCallUpdate { name: String, status: String },
    PermissionRequest {
        description: String,
        options: Vec<String>,
    },
}

/// Parse a `session/update` notification payload into an `AcpSessionEvent`.
/// Returns `None` for unrecognized updates.
pub fn parse_session_update(params: &Value) -> Option<AcpSessionEvent> {
    let update = params.get("update")?.get("sessionUpdate")?.as_str()?;
    let content = params.get("update")?;

    match update {
        "agent_message_chunk" => {
            let text = content.get("content")?.get("text")?.as_str()?.to_string();
            Some(AcpSessionEvent::AgentMessageChunk(text))
        }
        "agent_thought_chunk" => {
            let text = content.get("content")?.get("text")?.as_str()?.to_string();
            Some(AcpSessionEvent::AgentThoughtChunk(text))
        }
        "tool_call" => {
            let name = content.get("name")?.as_str()?.to_string();
            Some(AcpSessionEvent::ToolCall { name })
        }
        "tool_call_update" => {
            let name = content.get("name")?.as_str()?.to_string();
            let status = content.get("status")?.as_str()?.to_string();
            Some(AcpSessionEvent::ToolCallUpdate { name, status })
        }
        "permission_request" => {
            let description = content.get("description")?.as_str()?.to_string();
            let options = content
                .get("options")?
                .as_array()?
                .iter()
                .filter_map(|o| o.get("label")?.as_str()?.to_string().into())
                .collect();
            Some(AcpSessionEvent::PermissionRequest {
                description,
                options,
            })
        }
        _ => None,
    }
}

/// Bounded sender for session events.
pub type SessionEventTx = tokio::sync::mpsc::Sender<AcpSessionEvent>;

/// Bounded receiver for session events.
pub type SessionEventRx = tokio::sync::mpsc::Receiver<AcpSessionEvent>;

/// Create a bounded channel for session events (capacity 16).
pub fn create_event_channel() -> (SessionEventTx, SessionEventRx) {
    tokio::sync::mpsc::channel(16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_params(update_type: &str, content: &str) -> Value {
        json!({
            "update": {
                "sessionUpdate": update_type,
                "content": {
                    "text": content
                }
            }
        })
    }

    fn make_tool_params(tool_name: &str) -> Value {
        json!({
            "update": {
                "sessionUpdate": "tool_call",
                "name": tool_name
            }
        })
    }

    fn make_tool_update_params(tool_name: &str, status: &str) -> Value {
        json!({
            "update": {
                "sessionUpdate": "tool_call_update",
                "name": tool_name,
                "status": status
            }
        })
    }

    fn make_perm_params(desc: &str, labels: &[&str]) -> Value {
        json!({
            "update": {
                "sessionUpdate": "permission_request",
                "description": desc,
                "options": labels.iter().map(|l| json!({"label": l})).collect::<Vec<_>>()
            }
        })
    }

    #[test]
    fn test_parse_agent_message_chunk() {
        let params = make_params("agent_message_chunk", "hello");
        let ev = parse_session_update(&params).expect("should parse");
        assert!(matches!(ev, AcpSessionEvent::AgentMessageChunk(ref t) if t == "hello"));
    }

    #[test]
    fn test_parse_agent_thought_chunk() {
        let params = make_params("agent_thought_chunk", "thinking...");
        let ev = parse_session_update(&params).expect("should parse");
        assert!(matches!(ev, AcpSessionEvent::AgentThoughtChunk(ref t) if t == "thinking..."));
    }

    #[test]
    fn test_parse_tool_call() {
        let params = make_tool_params("web_search");
        let ev = parse_session_update(&params).expect("should parse");
        assert!(matches!(ev, AcpSessionEvent::ToolCall { ref name } if name == "web_search"));
    }

    #[test]
    fn test_parse_tool_call_update() {
        let params = make_tool_update_params("web_search", "completed");
        let ev = parse_session_update(&params).expect("should parse");
        assert!(matches!(ev, AcpSessionEvent::ToolCallUpdate { ref name, ref status } if name == "web_search" && status == "completed"));
    }

    #[test]
    fn test_parse_permission_request() {
        let params = make_perm_params("Allow file access?", &["yes", "no"]);
        let ev = parse_session_update(&params).expect("should parse");
        match ev {
            AcpSessionEvent::PermissionRequest {
                ref description,
                ref options,
            } => {
                assert_eq!(description, "Allow file access?");
                assert_eq!(options, &["yes", "no"]);
            }
            _ => panic!("expected PermissionRequest"),
        }
    }

    #[test]
    fn test_parse_unknown_returns_none() {
        let params = make_params("unknown_update", "");
        assert!(parse_session_update(&params).is_none());
    }

    #[test]
    fn test_channel_capacity() {
        let (tx, mut rx) = create_event_channel();
        assert!(tx.try_send(AcpSessionEvent::AgentMessageChunk("test".into())).is_ok());
        drop(tx);
        assert!(rx.try_recv().is_ok());
    }
}
