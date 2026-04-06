use super::events::{InputSource, PipelineState, TuiEvent};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers, MouseEventKind};

/// Action returned by key event handling.
pub enum Action {
    Quit,
    Submit(String),
    ToggleTts,
    ScrollUp,
    ScrollDown,
}

/// Role label for conversation messages.
#[derive(Clone, Debug)]
pub enum Role {
    User(InputSource),
    Assistant,
    Tool,
    Error,
}

/// A single message in the conversation view.
#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Local>,
}

/// TUI application state.
pub struct App {
    /// Finalized conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Current streaming assistant text (accumulates tokens).
    pub streaming_buffer: String,
    /// Current pipeline state.
    pub state: PipelineState,
    /// Text input buffer.
    pub input: String,
    /// Cursor position within input.
    pub cursor: usize,
    /// Scroll offset in conversation view (0 = bottom, positive = lines above bottom).
    pub scroll: u16,
    /// TTS enabled.
    pub tts_enabled: bool,
    /// Whether the app should quit.
    pub should_quit: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            streaming_buffer: String::new(),
            state: PipelineState::Idle,
            input: String::new(),
            cursor: 0,
            scroll: 0,
            tts_enabled: true,
            should_quit: false,
        }
    }

    /// Process a pipeline event and update app state.
    pub fn handle_tui_event(&mut self, event: TuiEvent) {
        match event {
            TuiEvent::StateChange(s) => {
                self.state = s;
            }
            TuiEvent::UserMessage { text, source } => {
                self.messages.push(ChatMessage {
                    role: Role::User(source),
                    content: text,
                    timestamp: chrono::Local::now(),
                });
                self.scroll = 0; // auto-scroll to bottom
            }
            TuiEvent::AssistantToken(token) => {
                self.streaming_buffer.push_str(&token);
                // Do not reset scroll — if the user has scrolled up to read,
                // preserve their position. scroll == 0 already tracks the bottom
                // dynamically as content grows.
            }
            TuiEvent::AssistantDone => {
                if !self.streaming_buffer.is_empty() {
                    let content = std::mem::take(&mut self.streaming_buffer);
                    self.messages.push(ChatMessage {
                        role: Role::Assistant,
                        content,
                        timestamp: chrono::Local::now(),
                    });
                }
                // Preserve scroll position — user may be reading earlier content.
            }
            TuiEvent::Error(msg) => {
                self.messages.push(ChatMessage {
                    role: Role::Error,
                    content: msg,
                    timestamp: chrono::Local::now(),
                });
                self.scroll = 0;
            }
            TuiEvent::ToolCall { name, result } => {
                // Truncate long results for display.
                let short = if result.len() > 120 {
                    format!("{}...", &result[..120])
                } else {
                    result
                };
                self.messages.push(ChatMessage {
                    role: Role::Tool,
                    content: format!("{name} -> {short}"),
                    timestamp: chrono::Local::now(),
                });
                self.scroll = 0;
            }
        }
    }

    /// Process a crossterm key event. Returns an Action if one should be taken.
    pub fn handle_key_event(&mut self, event: Event) -> Option<Action> {
        if let Event::Mouse(mouse) = event {
            match mouse.kind {
                MouseEventKind::ScrollUp => {
                    self.scroll = self.scroll.saturating_add(3);
                }
                MouseEventKind::ScrollDown => {
                    self.scroll = self.scroll.saturating_sub(3);
                }
                _ => {}
            }
            return None;
        }

        let Event::Key(KeyEvent {
            code, modifiers, ..
        }) = event
        else {
            return None;
        };

        match (modifiers, code) {
            (KeyModifiers::CONTROL, KeyCode::Char('c')) | (_, KeyCode::Esc) => {
                Some(Action::Quit)
            }
            (KeyModifiers::CONTROL, KeyCode::Char('t')) => Some(Action::ToggleTts),
            (_, KeyCode::PageUp) => Some(Action::ScrollUp),
            (_, KeyCode::PageDown) => Some(Action::ScrollDown),
            (_, KeyCode::Enter) => {
                let text = self.input.trim().to_string();
                if text.is_empty() {
                    return None;
                }
                self.input.clear();
                self.cursor = 0;
                Some(Action::Submit(text))
            }
            (_, KeyCode::Backspace) => {
                if self.cursor > 0 {
                    // Find previous char boundary.
                    let prev = self.input[..self.cursor]
                        .char_indices()
                        .next_back()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    self.input.drain(prev..self.cursor);
                    self.cursor = prev;
                }
                None
            }
            (_, KeyCode::Delete) => {
                if self.cursor < self.input.len() {
                    // Find next char boundary after cursor.
                    let next = self.input[self.cursor..]
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| self.cursor + i)
                        .unwrap_or(self.input.len());
                    self.input.drain(self.cursor..next);
                }
                None
            }
            (_, KeyCode::Left) => {
                if self.cursor > 0 {
                    self.cursor = self.input[..self.cursor]
                        .char_indices()
                        .next_back()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                }
                None
            }
            (_, KeyCode::Right) => {
                if self.cursor < self.input.len() {
                    self.cursor = self.input[self.cursor..]
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| self.cursor + i)
                        .unwrap_or(self.input.len());
                }
                None
            }
            (_, KeyCode::Home) => {
                self.cursor = 0;
                None
            }
            (_, KeyCode::End) => {
                self.cursor = self.input.len();
                None
            }
            (KeyModifiers::NONE | KeyModifiers::SHIFT, KeyCode::Char(c)) => {
                self.input.insert(self.cursor, c);
                self.cursor += c.len_utf8();
                None
            }
            _ => None,
        }
    }
}
