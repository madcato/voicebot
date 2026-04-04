use serde::Deserialize;
use tracing::{debug, warn};

use crate::llm::{OpenAIClient, Message};

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
#[allow(dead_code)]
pub async fn extract_facts(
    client: &OpenAIClient,
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
            warn!(target: "profile", "Profile extraction LLM call failed: {}", e);
            return vec![];
        }
    };

    parse_facts(&raw)
}

// ── JSON parsing ──────────────────────────────────────────────────────────────

#[allow(dead_code)]
#[derive(Deserialize)]
struct RawFact {
    key: String,
    value: String,
    #[serde(default = "default_confidence")]
    confidence: f64,
}

#[allow(dead_code)]
fn default_confidence() -> f64 {
    0.8
}

#[allow(dead_code)]
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
            debug!(target: "profile", "Could not parse profile extraction JSON: {} — raw: {:?}", e, raw);
            vec![]
        }
    }
}

#[allow(dead_code)]
fn normalize_key(key: &str) -> String {
    key.trim()
        .to_lowercase()
        .replace(' ', "_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

#[allow(dead_code)]
fn strip_code_fence(s: &str) -> &str {
    // Handle ```json ... ``` or ``` ... ```
    let s = s.trim_start_matches("```json").trim_start_matches("```");
    let s = if let Some(pos) = s.rfind("```") { &s[..pos] } else { s };
    s.trim()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{OpenAIClient, LlmSession};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn fact(key: &str, value: &str, confidence: f64) -> ProfileFact {
        ProfileFact { key: key.to_string(), value: value.to_string(), confidence }
    }

    // ── build_profile_context ─────────────────────────────────────────────────

    #[test]
    fn context_empty_for_no_facts() {
        assert!(build_profile_context(&[]).is_empty());
    }

    #[test]
    fn context_empty_when_all_facts_below_threshold() {
        let facts = vec![fact("name", "Daniel", 0.4), fact("city", "Madrid", 0.3)];
        assert!(build_profile_context(&facts).is_empty());
    }

    #[test]
    fn context_includes_facts_at_exact_threshold() {
        // MIN_INJECT_CONFIDENCE = 0.5 — boundary value must be included.
        let facts = vec![fact("job", "engineer", 0.5)];
        let ctx = build_profile_context(&facts);
        assert!(ctx.contains("job: engineer"));
    }

    #[test]
    fn context_excludes_facts_below_threshold() {
        let facts = vec![
            fact("name", "Daniel", 0.9),
            fact("city", "Madrid", 0.49), // just below the 0.5 threshold
        ];
        let ctx = build_profile_context(&facts);
        assert!(ctx.contains("name: Daniel"));
        assert!(!ctx.contains("Madrid"), "fact below threshold must be excluded");
    }

    #[test]
    fn context_contains_user_profile_header() {
        let facts = vec![fact("name", "Daniel", 0.9)];
        assert!(build_profile_context(&facts).contains("[USER PROFILE]"));
    }

    #[test]
    fn context_formats_each_fact_as_key_colon_value() {
        let facts = vec![
            fact("name", "Daniel", 0.95),
            fact("hobby_1", "Rust", 0.85),
        ];
        let ctx = build_profile_context(&facts);
        assert!(ctx.contains("name: Daniel"));
        assert!(ctx.contains("hobby_1: Rust"));
    }

    // ── extract_facts — LLM response parsing (via mock server) ───────────────

    #[tokio::test]
    async fn extract_facts_parses_single_fact() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content":
                    r#"[{"key": "name", "value": "Daniel", "confidence": 0.95}]"#
                }}]
            })))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "test", 256, 0.1, 0, -1);
        let facts = extract_facts(&client, "Me llamo Daniel.", "Encantado, Daniel.").await;

        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].key, "name");
        assert_eq!(facts[0].value, "Daniel");
        assert!((facts[0].confidence - 0.95).abs() < 0.001);
    }

    #[tokio::test]
    async fn extract_facts_parses_multiple_facts() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content":
                    r#"[
                        {"key": "name", "value": "Daniel", "confidence": 0.95},
                        {"key": "city", "value": "Madrid", "confidence": 0.8},
                        {"key": "job",  "value": "software engineer", "confidence": 0.9}
                    ]"#
                }}]
            })))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "test", 256, 0.1, 0, -1);
        let facts = extract_facts(
            &client,
            "Me llamo Daniel, vivo en Madrid y soy ingeniero de software.",
            "Gracias por contarme, Daniel.",
        )
        .await;

        assert_eq!(facts.len(), 3);
        assert!(facts.iter().any(|f| f.key == "name" && f.value == "Daniel"));
        assert!(facts.iter().any(|f| f.key == "city" && f.value == "Madrid"));
        assert!(facts.iter().any(|f| f.key == "job" && f.value == "software engineer"));
    }

    #[tokio::test]
    async fn extract_facts_returns_empty_vec_when_llm_finds_no_facts() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "[]"}}]
            })))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "test", 256, 0.1, 0, -1);
        let facts = extract_facts(&client, "¿Qué hora es?", "Son las 14:00.").await;
        assert!(facts.is_empty());
    }

    #[tokio::test]
    async fn extract_facts_strips_markdown_code_fences() {
        let json_in_fences =
            "```json\n[{\"key\": \"city\", \"value\": \"Madrid\", \"confidence\": 0.8}]\n```";
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": json_in_fences}}]
            })))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "test", 256, 0.1, 0, -1);
        let facts = extract_facts(&client, "Vivo en Madrid.", "Entendido.").await;

        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].key, "city");
        assert_eq!(facts[0].value, "Madrid");
    }

    #[tokio::test]
    async fn extract_facts_strips_plain_code_fences() {
        let json_in_fences =
            "```\n[{\"key\": \"hobby_1\", \"value\": \"Rust\", \"confidence\": 0.9}]\n```";
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": json_in_fences}}]
            })))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "test", 256, 0.1, 0, -1);
        let facts = extract_facts(&client, "Me encanta programar en Rust.", "Es un lenguaje excelente.").await;

        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].key, "hobby_1");
    }

    #[tokio::test]
    async fn extract_facts_normalizes_keys_to_lowercase_underscores() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content":
                    r#"[{"key": "Favorite Hobby", "value": "Rust", "confidence": 0.9}]"#
                }}]
            })))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "test", 256, 0.1, 0, -1);
        let facts = extract_facts(&client, "Me encanta Rust.", "Es genial.").await;

        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].key, "favorite_hobby");
    }

    #[tokio::test]
    async fn extract_facts_clamps_confidence_to_valid_range() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content":
                    // LLM returns confidence out of range
                    r#"[{"key": "name", "value": "Daniel", "confidence": 1.5}]"#
                }}]
            })))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "test", 256, 0.1, 0, -1);
        let facts = extract_facts(&client, "Soy Daniel.", "Hola.").await;

        assert_eq!(facts.len(), 1);
        assert!(facts[0].confidence <= 1.0, "confidence must be clamped to 1.0");
    }

    #[tokio::test]
    async fn extract_facts_uses_default_confidence_when_missing() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content":
                    // No confidence field — should default to 0.8
                    r#"[{"key": "name", "value": "Daniel"}]"#
                }}]
            })))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "test", 256, 0.1, 0, -1);
        let facts = extract_facts(&client, "Me llamo Daniel.", "Hola.").await;

        assert_eq!(facts.len(), 1);
        assert!((facts[0].confidence - 0.8).abs() < 0.001, "default confidence should be 0.8");
    }

    #[tokio::test]
    async fn extract_facts_handles_llm_server_error_gracefully() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "test", 256, 0.1, 0, -1);
        // Must not panic — errors are swallowed and an empty vec is returned.
        let facts = extract_facts(&client, "Hola.", "Hola.").await;
        assert!(facts.is_empty());
    }

    #[tokio::test]
    async fn extract_facts_handles_non_json_response_gracefully() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "No user facts found in this exchange."}}]
            })))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "test", 256, 0.1, 0, -1);
        let facts = extract_facts(&client, "Hola.", "Hola.").await;
        assert!(facts.is_empty(), "non-JSON LLM output must yield empty facts without panic");
    }

    // ── Full injection chain ───────────────────────────────────────────────────
    // Mirrors the startup flow in main.rs:
    //   load profile facts → build_profile_context → prepend to system prompt
    //   → LlmSession::new → all_messages()[0] contains the full context

    #[test]
    fn profile_facts_injected_into_system_message() {
        let base_prompt = "Eres Jarvis, el asistente personal de Daniel.";
        let facts = vec![
            fact("name", "Daniel", 0.95),
            fact("job", "software engineer", 0.9),
            fact("hobby_1", "Rust", 0.85),
        ];

        let profile_ctx = build_profile_context(&facts);
        let system_prompt = format!("{base_prompt}{profile_ctx}");
        let session = LlmSession::new(&system_prompt, 0);

        let msgs = session.all_messages();
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.starts_with(base_prompt));
        assert!(msgs[0].content.contains("[USER PROFILE]"));
        assert!(msgs[0].content.contains("name: Daniel"));
        assert!(msgs[0].content.contains("job: software engineer"));
        assert!(msgs[0].content.contains("hobby_1: Rust"));
    }

    #[test]
    fn low_confidence_facts_not_injected_into_system_message() {
        let base_prompt = "Eres Jarvis.";
        let facts = vec![
            fact("name", "Daniel", 0.95),
            fact("city", "Madrid", 0.3), // below 0.5 threshold — must be excluded
        ];

        let system_prompt = format!("{base_prompt}{}", build_profile_context(&facts));
        let session = LlmSession::new(&system_prompt, 0);

        let msgs = session.all_messages();
        assert!(msgs[0].content.contains("name: Daniel"));
        assert!(!msgs[0].content.contains("city: Madrid"));
    }

    #[test]
    fn empty_profile_leaves_system_prompt_unchanged() {
        let base_prompt = "Eres Jarvis.";
        let profile_ctx = build_profile_context(&[]);
        let system_prompt = format!("{base_prompt}{profile_ctx}");
        let session = LlmSession::new(&system_prompt, 0);

        let msgs = session.all_messages();
        assert_eq!(msgs[0].content, base_prompt);
        assert!(!msgs[0].content.contains("[USER PROFILE]"));
    }

    #[test]
    fn profile_block_appended_after_base_prompt_not_before() {
        let base_prompt = "Eres Jarvis.";
        let facts = vec![fact("name", "Daniel", 0.9)];
        let system_prompt = format!("{base_prompt}{}", build_profile_context(&facts));

        let pos_prompt = system_prompt.find("Eres Jarvis.").unwrap();
        let pos_profile = system_prompt.find("[USER PROFILE]").unwrap();
        assert!(
            pos_prompt < pos_profile,
            "base prompt must appear before [USER PROFILE] block"
        );
    }

    #[tokio::test]
    async fn full_extraction_and_injection_cycle() {
        // End-to-end: LLM extracts facts from a conversation → facts are
        // filtered by confidence → injected into the system prompt for the
        // next session, as happens at startup after DB load.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": r#"[
                    {"key": "name",   "value": "Daniel",            "confidence": 0.95},
                    {"key": "city",   "value": "Madrid",            "confidence": 0.8},
                    {"key": "hobby_1","value": "Rust",              "confidence": 0.9},
                    {"key": "skill",  "value": "software engineer", "confidence": 0.3}
                ]"#}}]
            })))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "test", 256, 0.1, 0, -1);

        // Step 1: extract facts from the last turn
        let facts = extract_facts(
            &client,
            "Me llamo Daniel, vivo en Madrid y programo en Rust.",
            "Anotado, Daniel.",
        )
        .await;
        assert_eq!(facts.len(), 4);

        // Step 2: build the profile context (filters to confidence >= 0.5)
        let profile_ctx = build_profile_context(&facts);
        assert!(profile_ctx.contains("name: Daniel"));
        assert!(profile_ctx.contains("city: Madrid"));
        assert!(profile_ctx.contains("hobby_1: Rust"));
        assert!(!profile_ctx.contains("skill"), "confidence 0.3 must be filtered out");

        // Step 3: inject into the next session's system prompt
        let base = "Eres Jarvis, el asistente de Daniel.";
        let session = LlmSession::new(&format!("{base}{profile_ctx}"), 0);
        let msgs = session.all_messages();

        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.contains("[USER PROFILE]"));
        assert!(msgs[0].content.contains("name: Daniel"));
        assert!(msgs[0].content.contains("city: Madrid"));
        assert!(msgs[0].content.contains("hobby_1: Rust"));
        assert!(!msgs[0].content.contains("skill"));
    }

    /// Integration test for `extract_facts` using a real LLM server.
    ///
    /// Reads LLM config from `.env` (LLM_URL, LLM_MODEL, LLM_PROVIDER, LLM_API_KEY).
    ///
    /// Run manually:
    /// ```sh
    /// cargo test test_extract_facts_real_llm --bin voicebot -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore]
    async fn test_extract_facts_real_llm() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("debug")
            .try_init();

        let _ = dotenvy::dotenv();
        let llm_url = std::env::var("LLM_URL")
            .unwrap_or_else(|_| "http://localhost:8080".to_string());
        let llm_model = std::env::var("LLM_MODEL")
            .unwrap_or_else(|_| "local-model".to_string());
        let llm_provider = std::env::var("LLM_PROVIDER")
            .unwrap_or_else(|_| "llama".to_string());
        let llm_api_key = std::env::var("LLM_API_KEY").unwrap_or_default();
        let client = OpenAIClient::new(
            &llm_url,
            &llm_model,
            400,
            0.3,
            0,
            -1,
        )
        .with_provider(&llm_provider)
        .with_api_key(&llm_api_key);

        // ── Case 1: User reveals personal facts ─────────────────────────────
        let facts = extract_facts(
            &client,
            "Me llamo Daniel, vivo en Madrid y trabajo como ingeniero de software.",
            "Encantado de conocerte, Daniel. Madrid es una ciudad estupenda.",
        )
        .await;

        println!("Extracted facts: {:#?}", facts);
        assert!(
            !facts.is_empty(),
            "LLM should extract at least one fact from a self-introduction"
        );
        assert!(
            facts.iter().any(|f| f.key == "name" && f.value.to_lowercase().contains("daniel")),
            "Should extract the user's name 'Daniel', got: {:?}",
            facts
        );

        // ── Case 2: No user facts present ───────────────────────────────────
        let facts_empty = extract_facts(
            &client,
            "¿Qué hora es?",
            "Son las 14:00.",
        )
        .await;

        println!("Facts from factless exchange: {:#?}", facts_empty);
        assert!(
            facts_empty.is_empty(),
            "Should return empty vec when no user facts are present, got: {:?}",
            facts_empty
        );

        // ── Case 3: Multiple facts ──────────────────────────────────────────
        let facts_multi = extract_facts(
            &client,
            "Tengo 35 años, me encanta programar en Rust y tengo un perro llamado Max.",
            "Qué bien, Rust es un gran lenguaje. Max debe ser un gran compañero.",
        )
        .await;

        println!("Multi-fact extraction: {:#?}", facts_multi);
        assert!(
            facts_multi.len() >= 2,
            "Should extract at least 2 facts from a rich message, got {}",
            facts_multi.len()
        );

        // Confidence values should be valid
        for fact in &facts_multi {
            assert!(
                fact.confidence > 0.0 && fact.confidence <= 1.0,
                "Confidence should be in (0, 1], got {} for key '{}'",
                fact.confidence,
                fact.key
            );
            assert!(!fact.key.is_empty(), "Key should not be empty");
            assert!(!fact.value.is_empty(), "Value should not be empty");
        }

        println!("\n✓ extract_facts integration test passed");
    }
}
