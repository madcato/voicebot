use serde::Deserialize;
use tracing::{debug, info, warn};

use crate::db::{Memory, NewMemory};
use crate::llm::{Message, OpenAIClient};

/// Maximum number of memories injected into the system prompt.
const MAX_MEMORIES_IN_PROMPT: usize = 50;

/// Build the `[MEMORIES]` block injected into the system prompt.
///
/// Returns an empty string if there are no memories. Caps output at
/// [`MAX_MEMORIES_IN_PROMPT`] entries to prevent unbounded growth.
pub fn build_memory_context(memories: &[Memory]) -> String {
    if memories.is_empty() {
        return String::new();
    }

    let mut block = String::from("\n\n[MEMORIES]\n");
    for mem in memories.iter().take(MAX_MEMORIES_IN_PROMPT) {
        block.push_str("- ");
        block.push_str(&mem.content);
        block.push('\n');
    }
    block
}

/// Action the LLM wants to take on a memory.
#[derive(Debug, Clone, Deserialize)]
struct RawMemoryAction {
    content: String,
    #[serde(default = "default_category")]
    category: String,
    #[serde(default = "default_action")]
    action: String,
    /// ID of an existing memory to archive (only when action == "archive").
    #[serde(default)]
    archive_id: Option<i64>,
}

fn default_category() -> String {
    "general".to_string()
}

fn default_action() -> String {
    "add".to_string()
}

/// Result of memory extraction: new memories to add and IDs to archive.
pub struct MemoryExtractionResult {
    pub new_memories: Vec<NewMemory>,
    pub archive_ids: Vec<i64>,
}

/// Ask the LLM to extract persistent memories from a conversation excerpt.
///
/// `existing_memories` is passed so the LLM can avoid duplicates and mark
/// outdated memories for archival.
pub async fn extract_memories(
    client: &OpenAIClient,
    conversation_text: &str,
    existing_memories: &[Memory],
) -> MemoryExtractionResult {
    if conversation_text.trim().is_empty() {
        return MemoryExtractionResult {
            new_memories: vec![],
            archive_ids: vec![],
        };
    }

    let mut existing_block = String::new();
    if !existing_memories.is_empty() {
        existing_block.push_str("\n\nExisting memories (do NOT duplicate these — only add genuinely new information, or archive outdated ones):\n");
        for mem in existing_memories {
            existing_block.push_str(&format!("[id={}] {}\n", mem.id, mem.content));
        }
    }

    let messages = vec![
        Message::system(format!(
            "Extract persistent memories from the conversation below.\n\
             Focus on:\n\
             - Projects the user is working on and their status\n\
             - Important decisions made\n\
             - Preferences expressed (beyond simple profile facts like name/age/city)\n\
             - Relationships and people mentioned\n\
             - Plans, goals, and deadlines\n\
             - Technical context (stack, problems, solutions)\n\
             - Anything the user would expect to be remembered next time\n\n\
             Do NOT extract:\n\
             - Basic profile facts (name, age, city, job) — those are handled separately\n\
             - Transient conversation details (greetings, small talk)\n\
             - Information already in existing memories unless it needs updating\n\n\
             Return ONLY a JSON array. Each element:\n\
             {{\"content\": \"...\", \"category\": \"general|project|preference|decision|relationship\", \
             \"action\": \"add|archive\", \"archive_id\": null}}\n\
             - Use \"add\" for new memories\n\
             - Use \"archive\" with the archive_id to mark an existing memory as outdated\n\
             If no memories worth saving, return [].{existing_block}"
        )),
        Message::user(format!("Conversation:\n\n{conversation_text}")),
    ];

    let raw = match client.complete(&messages).await {
        Ok(r) => r,
        Err(e) => {
            warn!(target: "memory", "Memory extraction LLM call failed: {}", e);
            return MemoryExtractionResult {
                new_memories: vec![],
                archive_ids: vec![],
            };
        }
    };

    parse_memory_response(&raw)
}

fn parse_memory_response(raw: &str) -> MemoryExtractionResult {
    let json_str = strip_code_fence(raw.trim());

    let parsed: Result<Vec<RawMemoryAction>, _> = serde_json::from_str(json_str);

    match parsed {
        Ok(actions) => {
            let mut new_memories = Vec::new();
            let mut archive_ids = Vec::new();

            for action in actions {
                if action.content.is_empty() && action.action != "archive" {
                    continue;
                }

                match action.action.as_str() {
                    "archive" => {
                        if let Some(id) = action.archive_id {
                            info!(target: "memory", "Archiving memory id={}", id);
                            archive_ids.push(id);
                        }
                    }
                    _ => {
                        // "add" or unrecognized action → treat as add
                        let category = validate_category(&action.category);
                        debug!(target: "memory", "New memory [{}]: {}", category, action.content);
                        new_memories.push(NewMemory {
                            content: action.content.trim().to_string(),
                            category: category.to_string(),
                        });
                    }
                }
            }

            MemoryExtractionResult {
                new_memories,
                archive_ids,
            }
        }
        Err(e) => {
            debug!(target: "memory", "Could not parse memory extraction JSON: {} — raw: {:?}", e, raw);
            MemoryExtractionResult {
                new_memories: vec![],
                archive_ids: vec![],
            }
        }
    }
}

fn validate_category(cat: &str) -> &str {
    match cat {
        "general" | "project" | "preference" | "decision" | "relationship" => cat,
        _ => "general",
    }
}

fn strip_code_fence(s: &str) -> &str {
    let s = s.trim_start_matches("```json").trim_start_matches("```");
    let s = if let Some(pos) = s.rfind("```") {
        &s[..pos]
    } else {
        s
    };
    s.trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(id: i64, content: &str) -> Memory {
        Memory {
            id,
            content: content.to_string(),
            category: "general".to_string(),
            source_session_id: None,
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    // ── build_memory_context ─────────────────────────────────────────────────

    #[test]
    fn context_empty_for_no_memories() {
        assert!(build_memory_context(&[]).is_empty());
    }

    #[test]
    fn context_contains_header_and_memories() {
        let memories = vec![
            mem(1, "User is building a Rust voicebot"),
            mem(2, "User prefers Spanish for UI"),
        ];
        let ctx = build_memory_context(&memories);
        assert!(ctx.contains("[MEMORIES]"));
        assert!(ctx.contains("- User is building a Rust voicebot"));
        assert!(ctx.contains("- User prefers Spanish for UI"));
    }

    #[test]
    fn context_caps_at_max_memories() {
        let memories: Vec<Memory> = (0..60).map(|i| mem(i, &format!("Memory {i}"))).collect();
        let ctx = build_memory_context(&memories);
        let count = ctx.matches("\n- ").count();
        assert_eq!(count, MAX_MEMORIES_IN_PROMPT);
    }

    // ── parse_memory_response ────────────────────────────────────────────────

    #[test]
    fn parse_add_memories() {
        let json = r#"[
            {"content": "User works on Jarvis voicebot", "category": "project", "action": "add"},
            {"content": "User prefers concise responses", "category": "preference", "action": "add"}
        ]"#;
        let result = parse_memory_response(json);
        assert_eq!(result.new_memories.len(), 2);
        assert_eq!(result.new_memories[0].category, "project");
        assert_eq!(result.new_memories[1].category, "preference");
        assert!(result.archive_ids.is_empty());
    }

    #[test]
    fn parse_archive_action() {
        let json = r#"[
            {"content": "", "action": "archive", "archive_id": 42}
        ]"#;
        let result = parse_memory_response(json);
        assert!(result.new_memories.is_empty());
        assert_eq!(result.archive_ids, vec![42]);
    }

    #[test]
    fn parse_mixed_actions() {
        let json = r#"[
            {"content": "New fact", "category": "general", "action": "add"},
            {"content": "", "action": "archive", "archive_id": 5}
        ]"#;
        let result = parse_memory_response(json);
        assert_eq!(result.new_memories.len(), 1);
        assert_eq!(result.archive_ids, vec![5]);
    }

    #[test]
    fn parse_empty_array() {
        let result = parse_memory_response("[]");
        assert!(result.new_memories.is_empty());
        assert!(result.archive_ids.is_empty());
    }

    #[test]
    fn parse_invalid_json_returns_empty() {
        let result = parse_memory_response("No memories found.");
        assert!(result.new_memories.is_empty());
        assert!(result.archive_ids.is_empty());
    }

    #[test]
    fn parse_strips_code_fences() {
        let json = "```json\n[{\"content\": \"A memory\", \"category\": \"general\", \"action\": \"add\"}]\n```";
        let result = parse_memory_response(json);
        assert_eq!(result.new_memories.len(), 1);
    }

    #[test]
    fn parse_defaults_category_and_action() {
        let json = r#"[{"content": "Something important"}]"#;
        let result = parse_memory_response(json);
        assert_eq!(result.new_memories.len(), 1);
        assert_eq!(result.new_memories[0].category, "general");
    }

    #[test]
    fn parse_validates_unknown_category() {
        let json = r#"[{"content": "Fact", "category": "unknown_cat", "action": "add"}]"#;
        let result = parse_memory_response(json);
        assert_eq!(result.new_memories[0].category, "general");
    }

    #[test]
    fn parse_skips_empty_content_for_add() {
        let json = r#"[{"content": "", "category": "general", "action": "add"}]"#;
        let result = parse_memory_response(json);
        assert!(result.new_memories.is_empty());
    }
}
