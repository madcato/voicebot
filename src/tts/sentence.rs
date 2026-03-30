use regex::Regex;
use std::sync::OnceLock;

static SENTENCE_END: OnceLock<Regex> = OnceLock::new();
/// Early split points for the first sentence: comma, dash, semicolon, colon
/// followed by whitespace. Gets audio to the speaker faster on the first chunk.
static EARLY_SPLIT: OnceLock<Regex> = OnceLock::new();

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

fn early_split_re() -> &'static Regex {
    EARLY_SPLIT.get_or_init(|| {
        // Early boundaries: comma or dash followed by whitespace.
        // These let the first ~5-15 words reach TTS before a full sentence ends.
        Regex::new(r"[,\-—]+\s").unwrap()
    })
}

/// Minimum characters before an early split is allowed. Prevents emitting
/// tiny fragments like "Bueno," (7 chars) which sound choppy.
const EARLY_SPLIT_MIN_CHARS: usize = 20;

/// Maximum characters to buffer before forcing an early split on the first
/// sentence. If no comma/dash appears, emit after this many chars at the
/// next whitespace boundary.
const EARLY_SPLIT_MAX_CHARS: usize = 80;

/// Buffers incoming text tokens and emits complete sentences.
///
/// Mirrors the sentence splitting logic from butler/text-to-speech/main.py.
/// Tokens are accumulated until a sentence-ending punctuation mark followed
/// by whitespace (or end of input) is detected, then the complete sentence
/// is returned for TTS synthesis.
///
/// **First-sentence acceleration**: the first sentence of each response uses
/// more aggressive splitting (commas, dashes, or a word-count threshold) to
/// get audio to the speaker faster. Subsequent sentences use normal
/// sentence-boundary splitting.
pub struct SentenceSplitter {
    buffer: String,
    /// Whether the first sentence of this response has already been emitted.
    first_emitted: bool,
}

impl SentenceSplitter {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            first_emitted: false,
        }
    }

    /// Push a new token. Returns a complete sentence if one is ready, otherwise `None`.
    pub fn push(&mut self, token: &str) -> Option<String> {
        self.buffer.push_str(token);

        // Always check for a full sentence boundary first.
        let re = sentence_end_re();
        if let Some(m) = re.find(&self.buffer) {
            let end = m.end();
            let sentence = self.buffer[..end].trim().to_string();
            self.buffer = self.buffer[end..].to_string();
            if !sentence.is_empty() {
                self.first_emitted = true;
                return Some(sentence);
            }
        }

        // First-sentence acceleration: split at commas/dashes or word-count limit.
        if !self.first_emitted && self.buffer.len() >= EARLY_SPLIT_MIN_CHARS {
            // Try comma/dash split — find the first match past the minimum length.
            let early_re = early_split_re();
            let split_end = early_re
                .find_iter(&self.buffer)
                .map(|m| m.end())
                .find(|&end| end >= EARLY_SPLIT_MIN_CHARS);
            if let Some(end) = split_end {
                let sentence = self.buffer[..end].trim().to_string();
                self.buffer = self.buffer[end..].to_string();
                if !sentence.is_empty() {
                    self.first_emitted = true;
                    return Some(sentence);
                }
            }

            // Fallback: if buffer is very long without any punctuation, split at the
            // last whitespace before MAX_CHARS to avoid holding tokens indefinitely.
            if self.buffer.len() >= EARLY_SPLIT_MAX_CHARS {
                if let Some(pos) = self.buffer[..EARLY_SPLIT_MAX_CHARS].rfind(' ') {
                    if pos >= EARLY_SPLIT_MIN_CHARS {
                        let sentence = self.buffer[..pos].trim().to_string();
                        self.buffer = self.buffer[pos + 1..].to_string();
                        if !sentence.is_empty() {
                            self.first_emitted = true;
                            return Some(sentence);
                        }
                    }
                }
            }
        }

        None
    }

    /// Flush any remaining buffered text as a final sentence (call after stream ends).
    pub fn flush(&mut self) -> Option<String> {
        let remaining = self.buffer.trim().to_string();
        self.buffer.clear();
        self.first_emitted = false;
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

    // ── First-sentence acceleration tests ────────────────────────────────────

    #[test]
    fn first_sentence_splits_at_comma_when_long_enough() {
        let mut s = SentenceSplitter::new();
        // Feed a response that has a comma after 20+ chars
        s.push("Bueno, la respuesta es bastante sencilla, ");
        // Should have split at the first comma that passes the min-char threshold
        // "Bueno, la respuesta es bastante sencilla," = 42 chars
        // First comma at pos 6 ("Bueno,") — too short (< 20)
        // Second comma at pos 41 — long enough
        // Actually the push should return the split
        let mut s = SentenceSplitter::new();
        let result = s.push("Bueno, la respuesta es bastante sencilla, y luego");
        // The early_split_re matches ", " — first match "Bueno, " at pos 7 (< 20 chars)
        // Second match "sencilla, " at pos ~42 — this is >= 20
        assert!(result.is_some(), "should split at comma after 20+ chars");
        let text = result.unwrap();
        assert!(text.contains("sencilla,"), "split should include up to the comma: {}", text);
    }

    #[test]
    fn second_sentence_does_not_split_at_comma() {
        let mut s = SentenceSplitter::new();
        // Emit first sentence normally
        s.push("Primera oración completa. ");
        // Now the second sentence should NOT split at commas
        assert!(s.push("Segunda oración, con coma").is_none());
        // Only splits at sentence boundary
        let result = s.push(". ");
        assert_eq!(result.as_deref(), Some("Segunda oración, con coma."));
    }

    #[test]
    fn first_sentence_falls_back_to_max_chars_split() {
        let mut s = SentenceSplitter::new();
        // Long text without any commas or sentence-ending punctuation
        let long_text = "Esta es una respuesta muy larga que no tiene ninguna coma ni punto y sigue y sigue sin parar hasta que sea muy larga";
        s.push(long_text);
        // Should have split at a word boundary before EARLY_SPLIT_MAX_CHARS
        // The buffer is > 80 chars, so it should force a split
        assert!(s.first_emitted, "should have emitted first sentence via max-chars fallback");
    }

    #[test]
    fn flush_resets_first_emitted() {
        let mut s = SentenceSplitter::new();
        s.push("Primera respuesta. ");
        assert!(s.first_emitted);
        s.flush();
        assert!(!s.first_emitted, "flush should reset first_emitted for next response");
    }
}
