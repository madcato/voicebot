use serde::Deserialize;
use tracing::{debug, warn};

use crate::llm::{LlamaClient, Message};

/// Minimum confidence required for a fact to be injected into the system prompt.
const MIN_INJECT_CONFIDENCE: f64 = 0.5;

/// A single fact known about the user.
#[derive(Debug, Clone)]
pub struct ProfileFact {
    pub key: String,
    pub value: String,
    pub confidence: f64,
}

/// Build the [USER PROFILE] block injected into the system prompt.
///
/// Returns an empty string if there are no facts above the confidence threshold.
pub fn build_profile_context(facts: &[ProfileFact]) -> String {
    let relevant: Vec<&ProfileFact> = facts
        .iter()
        .filter(|f| f.confidence >= MIN_INJECT_CONFIDENCE)
        .collect();

    if relevant.is_empty() {
        return String::new();
    }

    let mut block = String::from("\n\n[USER PROFILE]\n");
    for fact in relevant {
        block.push_str(&fact.key);
        block.push_str(": ");
        block.push_str(&fact.value);
        block.push('\n');
    }
    block
}

/// Ask the LLM to extract user facts from a single exchange.
///
/// Returns a (possibly empty) list of discovered facts. Runs in a background
/// task — errors are logged but do not affect the conversation.
pub async fn extract_facts(
    client: &LlamaClient,
    user_text: &str,
    assistant_text: &str,
) -> Vec<ProfileFact> {
    let messages = vec![
        Message::system(
            "Extract facts about the user from the conversation excerpt below.\n\
             Return ONLY a JSON array. Each element: {\"key\": \"...\", \"value\": \"...\", \"confidence\": 0.0-1.0}\n\
             Keys must be lowercase with underscores. Use standard names when possible:\n\
             name, age, city, country, language, job, company, field, skill, hobby, \
             pet, family, preference, communication_style, personality_trait.\n\
             For multiple values of the same category use key suffixes: hobby_1, hobby_2, etc.\n\
             Only extract facts explicitly stated or strongly implied by the USER's words.\n\
             If no user facts are present, return [].",
        ),
        Message::user(format!("User: {user_text}\nAssistant: {assistant_text}")),
    ];

    let raw = match client.complete_short(&messages).await {
        Ok(r) => r,
        Err(e) => {
            warn!("Profile extraction LLM call failed: {}", e);
            return vec![];
        }
    };

    parse_facts(&raw)
}

// ── JSON parsing ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RawFact {
    key: String,
    value: String,
    #[serde(default = "default_confidence")]
    confidence: f64,
}

fn default_confidence() -> f64 {
    0.8
}

fn parse_facts(raw: &str) -> Vec<ProfileFact> {
    // Strip markdown code fences if the LLM wrapped the JSON
    let json_str = strip_code_fence(raw.trim());

    let parsed: Result<Vec<RawFact>, _> = serde_json::from_str(json_str);

    match parsed {
        Ok(facts) => facts
            .into_iter()
            .filter(|f| !f.key.is_empty() && !f.value.is_empty())
            .map(|f| ProfileFact {
                key: normalize_key(&f.key),
                value: f.value.trim().to_string(),
                confidence: f.confidence.clamp(0.0, 1.0),
            })
            .collect(),
        Err(e) => {
            debug!("Could not parse profile extraction JSON: {} — raw: {:?}", e, raw);
            vec![]
        }
    }
}

fn normalize_key(key: &str) -> String {
    key.trim()
        .to_lowercase()
        .replace(' ', "_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

fn strip_code_fence(s: &str) -> &str {
    // Handle ```json ... ``` or ``` ... ```
    let s = s.trim_start_matches("```json").trim_start_matches("```");
    let s = if let Some(pos) = s.rfind("```") { &s[..pos] } else { s };
    s.trim()
}
