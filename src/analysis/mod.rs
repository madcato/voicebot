pub mod identity;

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// A single piece of contextual information produced by an analyzer.
/// Entries expire via TTL so stale data never reaches the LLM.
pub struct ContextEntry {
    pub key: &'static str,
    pub value: String,
    pub confidence: f32,
    pub valid_until: Instant,
    pub source: &'static str,
}

/// The analysis blackboard: a shared store of fresh contextual entries.
/// Analyzers write to it; the LLM task reads it before each request.
pub struct ContextLens {
    entries: HashMap<&'static str, ContextEntry>,
}

impl ContextLens {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert or replace an entry by key.
    pub fn upsert(&mut self, entry: ContextEntry) {
        self.entries.insert(entry.key, entry);
    }

    /// Return the entry for a given key if it is still fresh.
    pub fn get(&self, key: &str) -> Option<&ContextEntry> {
        self.entries
            .get(key)
            .filter(|e| e.valid_until > Instant::now())
    }

    /// Remove all expired entries.
    pub fn purge_expired(&mut self) {
        let now = Instant::now();
        self.entries.retain(|_, e| e.valid_until > now);
    }

    /// Format all fresh entries as an `[Analysis Context]` block for injection
    /// into the LLM system prompt. Returns `None` when nothing is fresh.
    pub fn format_for_llm(&self) -> Option<String> {
        let now = Instant::now();
        let fresh: Vec<&ContextEntry> = self
            .entries
            .values()
            .filter(|e| e.valid_until > now)
            .collect();

        if fresh.is_empty() {
            return None;
        }

        let mut out = String::from("\n[Analysis Context]\n");
        for e in fresh {
            out.push_str(&format!("{}: {}\n", e.key, e.value));
        }
        Some(out)
    }
}

impl Default for ContextLens {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for background audio analyzers that write results to the ContextLens.
pub trait AudioAnalyzer: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn analyze(
        &self,
        audio: Arc<Vec<f32>>,
        sample_rate: u32,
    ) -> impl Future<Output = Option<ContextEntry>> + Send;
}

/// An audio clip ready for analysis.
pub struct AudioClip {
    pub samples: Arc<Vec<f32>>,
    pub sample_rate: u32,
}

/// Routes audio clips to registered background analyzers.
pub struct AnalysisDispatcher {
    senders: Vec<mpsc::Sender<Arc<AudioClip>>>,
}

impl AnalysisDispatcher {
    pub fn new() -> Self {
        Self {
            senders: Vec::new(),
        }
    }

    /// Register an audio analyzer and return the receiver end of its channel.
    /// The caller is responsible for spawning a task that drains the receiver.
    pub fn register_audio_channel(&mut self, capacity: usize) -> mpsc::Receiver<Arc<AudioClip>> {
        let (tx, rx) = mpsc::channel(capacity);
        self.senders.push(tx);
        rx
    }

    /// Dispatch an audio clip to all registered analyzers (non-blocking, best-effort).
    pub fn dispatch(&self, clip: Arc<AudioClip>) {
        for tx in &self.senders {
            let _ = tx.try_send(Arc::clone(&clip));
        }
    }
}

impl Default for AnalysisDispatcher {
    fn default() -> Self {
        Self::new()
    }
}
