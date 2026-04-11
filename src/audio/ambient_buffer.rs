use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A single buffered utterance from the ambient environment.
pub struct AmbientEntry {
    pub speaker_label: String,
    pub transcript: String,
    pub timestamp: Instant,
}

/// Rolling window of recent ambient speech from all speakers (enrolled and unknown).
///
/// Used to give the LLM context about what was said before the user invoked
/// the wake word — e.g., ongoing conversations, TV audio, radio, etc.
pub struct AmbientBuffer {
    entries: VecDeque<AmbientEntry>,
    max_entries: usize,
    max_duration: Duration,
}

impl AmbientBuffer {
    pub fn new(max_entries: usize, max_duration_minutes: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
            max_duration: Duration::from_secs(max_duration_minutes * 60),
        }
    }

    /// Add a transcribed utterance to the buffer, evicting old entries as needed.
    pub fn push(&mut self, speaker_label: String, transcript: String) {
        // Evict entries older than the rolling window.
        let cutoff = Instant::now() - self.max_duration;
        while self.entries.front().is_some_and(|e| e.timestamp < cutoff) {
            self.entries.pop_front();
        }
        // Evict oldest if at capacity.
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(AmbientEntry {
            speaker_label,
            transcript,
            timestamp: Instant::now(),
        });
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Format buffered entries as a `[Contexto reciente]` block for the LLM.
    /// Returns `None` if the buffer is empty.
    pub fn format_context(&self) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        let mut lines = vec!["[Contexto reciente]".to_string()];
        for entry in &self.entries {
            lines.push(format!("{}: {}", entry.speaker_label, entry.transcript));
        }
        Some(lines.join("\n"))
    }
}

/// Returns true if the text contains a demonstrative or referential expression
/// that suggests the user is referring to something heard in the environment.
pub fn has_referential(text: &str) -> bool {
    let lower = text.to_lowercase();
    // Spanish referentials
    let es = [
        "eso", "esa", "ese", "esto", "aquello",
        "lo que dijo", "lo que dijeron", "lo que decían",
        "lo que dijiste", "lo que mencionó", "lo que mencionaron",
        "de qué hablan", "de qué hablaban", "qué dijeron",
        "qué dijo", "qué decían",
    ];
    // English referentials
    let en = [
        "that thing", "what they said", "what he said", "what she said",
        "what they were", "what was that", "what is that",
        "what are they", "that particle", "that topic",
    ];
    es.iter().chain(en.iter()).any(|&phrase| lower.contains(phrase))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_format() {
        let mut buf = AmbientBuffer::new(10, 3);
        buf.push("Speaker_1".to_string(), "Hola, ¿cómo estás?".to_string());
        buf.push("Ambiente".to_string(), "Scientists discovered a new particle.".to_string());

        let ctx = buf.format_context().unwrap();
        assert!(ctx.contains("[Contexto reciente]"));
        assert!(ctx.contains("Speaker_1: Hola"));
        assert!(ctx.contains("Ambiente: Scientists"));
    }

    #[test]
    fn empty_buffer_returns_none() {
        let buf = AmbientBuffer::new(10, 3);
        assert!(buf.format_context().is_none());
    }

    #[test]
    fn evicts_at_capacity() {
        let mut buf = AmbientBuffer::new(3, 60);
        for i in 0..5u8 {
            buf.push(format!("S{i}"), format!("msg {i}"));
        }
        // Only 3 entries remain.
        assert_eq!(buf.entries.len(), 3);
        // Oldest evicted — first entry should be msg 2.
        assert_eq!(buf.entries[0].transcript, "msg 2");
    }

    #[test]
    fn has_referential_spanish() {
        assert!(has_referential("Jarvis, ¿qué es eso?"));
        assert!(has_referential("¿qué dijo?"));
        assert!(!has_referential("¿cuánto es dos más dos?"));
    }

    #[test]
    fn has_referential_english() {
        assert!(has_referential("Jarvis, what is that?"));
        assert!(has_referential("What they said earlier"));
        assert!(!has_referential("What time is it?"));
    }
}
