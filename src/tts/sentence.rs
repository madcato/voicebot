use regex::Regex;
use std::sync::OnceLock;

static SENTENCE_END: OnceLock<Regex> = OnceLock::new();

fn sentence_end_re() -> &'static Regex {
    SENTENCE_END.get_or_init(|| {
        // Sentence boundary: one or more punctuation chars followed by whitespace or end of string
        // Require actual whitespace after punctuation — the $ anchor caused false positives
        // in streaming mode (e.g. "10:" or "3." at end of buffer would fire prematurely).
        // Decimal numbers (3.1415) and times (10:30) are safe because their punctuation
        // is always followed by a digit, never by \s. flush() handles the final fragment.
        Regex::new(r"[.!?;:]+(?:\s|\n)").unwrap()
    })
}

/// Buffers incoming text tokens and emits complete sentences.
///
/// Mirrors the sentence splitting logic from butler/text-to-speech/main.py.
/// Tokens are accumulated until a sentence-ending punctuation mark followed
/// by whitespace (or end of input) is detected, then the complete sentence
/// is returned for TTS synthesis.
pub struct SentenceSplitter {
    buffer: String,
}

impl SentenceSplitter {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Push a new token. Returns a complete sentence if one is ready, otherwise `None`.
    pub fn push(&mut self, token: &str) -> Option<String> {
        self.buffer.push_str(token);

        let re = sentence_end_re();
        if let Some(m) = re.find(&self.buffer) {
            let end = m.end();
            let sentence = self.buffer[..end].trim().to_string();
            self.buffer = self.buffer[end..].to_string();
            if !sentence.is_empty() {
                return Some(sentence);
            }
        }

        None
    }

    /// Flush any remaining buffered text as a final sentence (call after stream ends).
    pub fn flush(&mut self) -> Option<String> {
        let remaining = self.buffer.trim().to_string();
        self.buffer.clear();
        if remaining.is_empty() {
            None
        } else {
            Some(remaining)
        }
    }
}

impl Default for SentenceSplitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_period() {
        let mut s = SentenceSplitter::new();
        assert!(s.push("Hola").is_none());
        assert!(s.push(" mundo").is_none());
        let sentence = s.push(". ");
        assert_eq!(sentence.as_deref(), Some("Hola mundo."));
    }

    #[test]
    fn splits_on_question_mark() {
        let mut s = SentenceSplitter::new();
        s.push("¿Cómo estás");
        let sentence = s.push("? ");
        assert_eq!(sentence.as_deref(), Some("¿Cómo estás?"));
    }

    #[test]
    fn flush_returns_remaining() {
        let mut s = SentenceSplitter::new();
        s.push("Sin puntuación final");
        assert_eq!(s.flush().as_deref(), Some("Sin puntuación final"));
        assert!(s.flush().is_none());
    }

    #[test]
    fn empty_flush_returns_none() {
        let mut s = SentenceSplitter::new();
        assert!(s.flush().is_none());
    }

    #[test]
    fn no_false_positive_on_time_mid_stream() {
        // "10:" at end of buffer must not fire; only splits when whitespace follows
        let mut s = SentenceSplitter::new();
        assert!(s.push("Son las 10").is_none());
        assert!(s.push(":").is_none());   // buffer ends with ":", no whitespace yet
        assert!(s.push("30").is_none());
        assert!(s.push(" hoy.").is_none());
        assert_eq!(s.flush().as_deref(), Some("Son las 10:30 hoy."));
    }

    #[test]
    fn no_false_positive_on_decimal_mid_stream() {
        // "3." at end of buffer must not fire
        let mut s = SentenceSplitter::new();
        assert!(s.push("El valor es 3").is_none());
        assert!(s.push(".").is_none());   // buffer ends with ".", no whitespace yet
        assert!(s.push("1415").is_none());
        assert_eq!(s.flush().as_deref(), Some("El valor es 3.1415"));
    }
}
