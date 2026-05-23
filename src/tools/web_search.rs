use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use tracing::{info, warn};

use super::Tool;
use crate::llm::{Message, OpenAIClient};

/// Hard timeout for SearXNG HTTP requests.
const SEARCH_TIMEOUT_SECS: u64 = 10;

/// Maximum output returned to the LLM (bytes).
const MAX_OUTPUT_BYTES: usize = 4_000;

/// Default number of results returned to the LLM.
const DEFAULT_MAX_RESULTS: usize = 5;

// ── SearXNG response types ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct SearxResponse {
    #[serde(default)]
    results: Vec<SearxResult>,
}

#[derive(Deserialize)]
struct SearxResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: String,
}

// ── Tool implementation ─────────────────────────────────────────────────────

/// Web search tool backed by a SearXNG instance.
///
/// Enabled only when `SEARXNG_URL` is set. Configured via:
/// - `SEARXNG_URL` — base URL of the SearXNG instance
/// - `SEARXNG_SECRET` — Bearer token for authentication
pub struct WebSearchTool {
    base_url: String,
    secret: String,
    client: reqwest::Client,
    /// When set, the secondary LLM synthesizes raw SearXNG results into a
    /// concise voice-ready summary before returning to the primary LLM.
    synthesis_client: Option<Arc<OpenAIClient>>,
}

impl WebSearchTool {
    pub fn new(base_url: String, secret: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(SEARCH_TIMEOUT_SECS))
            .build()
            .expect("failed to build HTTP client for web_search");
        Self {
            base_url,
            secret,
            client,
            synthesis_client: None,
        }
    }

    /// Attach a secondary LLM client for result synthesis.
    pub fn with_synthesis(mut self, client: Arc<OpenAIClient>) -> Self {
        self.synthesis_client = Some(client);
        self
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Busca informaci\u{00f3}n en internet. Usa esta herramienta cuando el usuario pregunte \
         sobre informaci\u{00f3}n actual, noticias, eventos recientes, datos que no conoces, \
         o necesites verificar algo. Devuelve los primeros resultados con t\u{00ed}tulo, \
         fragmento y URL."
    }

    fn is_background(&self) -> bool {
        true
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5)"
                }
            },
            "required": ["query"]
        })
    }

    async fn run(&self, args: &str) -> String {
        // Parse arguments.
        let (query, max_results) = match serde_json::from_str::<serde_json::Value>(args) {
            Ok(v) => {
                let q = v["query"].as_str().unwrap_or("").trim().to_string();
                let n = v["max_results"]
                    .as_u64()
                    .map(|n| n as usize)
                    .unwrap_or(DEFAULT_MAX_RESULTS);
                (q, n)
            }
            Err(_) => (args.trim().to_string(), DEFAULT_MAX_RESULTS),
        };

        if query.is_empty() {
            return "Error: no search query provided.".to_string();
        }

        info!(target: "tools", "web_search query={:?} max_results={}", query, max_results);

        // Build request.
        let url = format!("{}/search", self.base_url.trim_end_matches('/'));
        let mut req = self
            .client
            .get(&url)
            .query(&[("q", &query), ("format", &"json".to_string())]);

        if !self.secret.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.secret));
        }

        // Execute request.
        let response = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                warn!(target: "tools", "web_search HTTP error: {}", e);
                if e.is_timeout() {
                    return "Error: search request timed out.".to_string();
                }
                return format!("Error: could not reach search service: {e}");
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            warn!(target: "tools", "web_search HTTP {}", status);
            return format!("Error: search service returned HTTP {status}");
        }

        // Parse response.
        let body = match response.json::<SearxResponse>().await {
            Ok(b) => b,
            Err(e) => {
                warn!(target: "tools", "web_search JSON parse error: {}", e);
                return format!("Error: could not parse search results: {e}");
            }
        };

        if body.results.is_empty() {
            return "No results found.".to_string();
        }

        // Format raw results.
        let raw = format_results(&body.results, max_results);

        // If a synthesis client is configured, ask the secondary LLM to distill
        // the raw results into a concise voice-ready summary.
        if let Some(ref client) = self.synthesis_client {
            let prompt = format!(
                "El usuario preguntó: \"{query}\"\n\
                 Resultados de búsqueda:\n{raw}\n\n\
                 Resume en 2-3 frases concisas lo más relevante para responder al usuario. \
                 Solo el resumen, sin intro ni explicación."
            );
            match client.complete_short(&[Message::user(&prompt)]).await {
                Ok(summary) if !summary.is_empty() => {
                    info!(target: "tools", "web_search: synthesized result ({} chars → {} chars)", raw.len(), summary.len());
                    return summary;
                }
                Ok(_) => {}
                Err(e) => warn!(target: "tools", "web_search synthesis error: {}", e),
            }
        }

        raw
    }
}

/// Format SearXNG results as a numbered list, truncated to [`MAX_OUTPUT_BYTES`].
fn format_results(results: &[SearxResult], max: usize) -> String {
    let mut out = String::new();
    for (i, r) in results.iter().take(max).enumerate() {
        let entry = format!("{}. {}\n   {}\n   {}\n\n", i + 1, r.title, r.content, r.url,);
        if out.len() + entry.len() > MAX_OUTPUT_BYTES {
            break;
        }
        out.push_str(&entry);
    }
    out.trim_end().to_string()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> WebSearchTool {
        WebSearchTool::new("http://localhost:8080".into(), "test-token".into())
    }

    #[test]
    fn name_is_web_search() {
        assert_eq!(tool().name(), "web_search");
    }

    #[test]
    fn is_background_true() {
        assert!(tool().is_background());
    }

    #[test]
    fn description_is_non_empty() {
        assert!(!tool().description().is_empty());
    }

    #[test]
    fn parameters_has_query_field() {
        let params = tool().parameters();
        assert!(params["properties"]["query"].is_object());
    }

    #[test]
    fn parameters_query_is_required() {
        let params = tool().parameters();
        let required = params["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "query"));
    }

    #[tokio::test]
    async fn empty_query_returns_error() {
        let t = tool();
        let result = t.run(r#"{"query": ""}"#).await;
        assert!(result.contains("Error"), "got: {result}");
    }

    #[tokio::test]
    async fn raw_string_fallback_empty_returns_error() {
        let t = tool();
        let result = t.run("  ").await;
        assert!(result.contains("Error"), "got: {result}");
    }

    #[test]
    fn format_results_numbered_list() {
        let results = vec![
            SearxResult {
                title: "Rust lang".into(),
                url: "https://rust-lang.org".into(),
                content: "A systems programming language".into(),
            },
            SearxResult {
                title: "Cargo".into(),
                url: "https://doc.rust-lang.org/cargo".into(),
                content: "Rust package manager".into(),
            },
        ];
        let out = format_results(&results, 5);
        assert!(out.contains("1. Rust lang"));
        assert!(out.contains("2. Cargo"));
        assert!(out.contains("https://rust-lang.org"));
    }

    #[test]
    fn format_results_respects_max() {
        let results: Vec<SearxResult> = (0..10)
            .map(|i| SearxResult {
                title: format!("Result {i}"),
                url: format!("https://example.com/{i}"),
                content: format!("Content {i}"),
            })
            .collect();
        let out = format_results(&results, 3);
        assert!(out.contains("1. Result 0"));
        assert!(out.contains("3. Result 2"));
        assert!(!out.contains("4. Result 3"));
    }

    #[test]
    fn format_results_truncates_at_max_bytes() {
        let results: Vec<SearxResult> = (0..100)
            .map(|i| SearxResult {
                title: format!("Result {i} with a very long title to fill up bytes quickly"),
                url: format!("https://example.com/very/long/path/{i}"),
                content: "A".repeat(200),
            })
            .collect();
        let out = format_results(&results, 100);
        assert!(out.len() <= MAX_OUTPUT_BYTES + 500); // some slack for last entry
    }

    #[test]
    fn format_results_empty() {
        let out = format_results(&[], 5);
        assert!(out.is_empty());
    }
}
