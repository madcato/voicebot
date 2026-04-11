use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, Paragraph},
    Frame,
};

use super::app::{App, ChatMessage, Role};
use super::events::{InputSource, PipelineState};
use crate::tools::ConversationMode;

/// Render using full viewport: [message list (scrollable)] [input] [status bar].
/// The message list takes all available space minus input and status bar heights.
pub fn render(frame: &mut Frame, app: &mut App) {
    let total = frame.area();
    let width = total.width as usize;

    // Input height: wraps at terminal width (no border so full width available).
    let display_lines = input_display_lines(&app.input, width);
    let input_height = (display_lines as u16).clamp(1, 4);

    // Status bar always 1 row
    let status_height = 1;

    // Message list gets remaining space (top of screen)
    let message_list_height = total.height.saturating_sub(input_height + status_height);

    // Split into message list, input, and status
    let [message_list_area, input_area, status_area] = Layout::vertical([
        Constraint::Length(message_list_height),
        Constraint::Length(input_height),
        Constraint::Length(status_height),
    ])
    .areas(total);

    render_message_list(frame, app, message_list_area);
    render_input(frame, app, input_area);
    render_status(frame, app, status_area);
}

/// Render the scrollable message list.
fn render_message_list(frame: &mut Frame, app: &mut App, area: Rect) {
    let mut all_lines: Vec<Line<'static>> = vec![];

    // Add streaming buffer
    if !app.streaming_buffer.is_empty() {
        let streaming_lines = render_streaming_lines(&app.streaming_buffer, area.width as usize);
        all_lines.extend(streaming_lines);
        all_lines.push(Line::raw(""));
    }

    // Render each message (oldest to newest)
    for msg in app.messages.iter() {
        let lines = message_lines(msg, area.width);
        all_lines.extend(lines);
        all_lines.push(Line::raw(""));
    }

    // Auto-scroll: only clip if content exceeds viewport height
    let visible_height = area.height as usize;
    let display = if all_lines.len() <= visible_height {
        Text::from(all_lines)
    } else {
        let skip = all_lines.len() - visible_height;
        Text::from(all_lines[skip..].to_vec())
    };

    frame.render_widget(Paragraph::new(display), area);
}

/// Render the VOICEBOT splash screen (blue, centered).
fn render_splash(text: &str, width: usize) -> Vec<Line<'static>> {
    let text = text.to_string(); // Clone to make it 'static
    let mut lines: Vec<Line<'static>> = vec![];

    // Add top border
    lines.push(Line::from(vec![
        Span::raw("┌"),
        Span::raw("─".repeat(width.saturating_sub(2))),
        Span::raw("┐"),
    ]));

    // Add splash content with blue styling
    for line in text.lines() {
        let trimmed = line.trim().to_string();
        if !trimmed.is_empty() {
            lines.push(Line::from(vec![
                Span::raw("│ "),
                Span::styled(
                    trimmed,
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
            ]));
        }
    }

    // Add bottom border
    lines.push(Line::from(vec![
        Span::raw("└"),
        Span::raw("─".repeat(width.saturating_sub(2))),
        Span::raw("┘"),
    ]));

    lines
}

/// Build display lines for streaming buffer.
fn render_streaming_lines(buffer: &str, width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = vec![Line::from(vec![
        Span::raw("┌ "),
        Span::styled(
            "Jarvis [streaming]",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
    ])];

    for content_line in buffer.lines() {
        let wrapped = word_wrap_plain(&format!("│ {content_line}"), width);
        for row in wrapped {
            lines.push(Line::raw(row));
        }
    }

    lines.push(Line::from(vec![
        Span::raw("└"),
        Span::raw("─".repeat(width - 2)),
        Span::raw("┘"),
    ]));

    lines
}

/// Build display lines for a finalized message (used by mod.rs for insert_before).
pub fn message_lines(msg: &ChatMessage, width: u16) -> Vec<Line<'static>> {
    let w = width as usize;
    let mut lines: Vec<Line<'static>> = vec![];

    match &msg.role {
        Role::Splash => {
            // Splash screen - show VOICEBOT ASCII art
            let splash_text = r#"
  __  __          _              ___         _ 
 |  \/  |___ _  _(_)__ _ _  _   / __|___ _ _ | |
 | |\/| / _ \ || | / _` | || | | (_ / -_) ' \|_|
 |_| |_\___/\_,_|_\__, |\_, |  \___\___|_||_(_) 
                   |___/ |__/                    
 "#;
            lines.extend(render_splash(splash_text, w));
        }
        Role::User(source) => {
            let source_label = match source {
                InputSource::Voice => "voice",
                InputSource::Text => "text",
            };
            let time = msg.timestamp.format("%H:%M:%S").to_string();

            // User message header
            lines.push(Line::from(vec![
                Span::raw("┌ "),
                Span::styled(
                    format!("You [{source_label}]"),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(time, Style::default().fg(Color::Rgb(100, 100, 100))),
            ]));

            // User message content with border - wrap at width-2 (for "│ " prefix)
            for content_line in msg.content.lines() {
                let wrapped = word_wrap_plain(content_line, w.saturating_sub(2));
                for line in wrapped {
                    lines.push(Line::from(vec![Span::raw("│ "), Span::raw(line)]));
                }
            }
            // Add closing border line if we have content lines
            let content_lines = msg.content.lines().count();
            if content_lines > 0 {
                lines.push(Line::from(vec![
                    Span::raw("└"),
                    Span::raw("─".repeat(w.saturating_sub(2))),
                    Span::raw("┘"),
                ]));
            }
        }
        Role::Assistant => {
            let time = msg.timestamp.format("%H:%M:%S").to_string();

            // Assistant message header
            lines.push(Line::from(vec![
                Span::raw("┌ "),
                Span::styled(
                    "Jarvis",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(time, Style::default().fg(Color::Rgb(100, 100, 100))),
            ]));

            // Assistant message content with border - wrap at width-2 (for "│ " prefix)
            for content_line in msg.content.lines() {
                let wrapped = word_wrap_plain(content_line, w.saturating_sub(2));
                for line in wrapped {
                    lines.push(Line::from(vec![Span::raw("│ "), Span::raw(line)]));
                }
            }
            // Add closing border line if we have content lines
            let content_lines = msg.content.lines().count();
            if content_lines > 0 {
                lines.push(Line::from(vec![
                    Span::raw("└"),
                    Span::raw("─".repeat(w.saturating_sub(2))),
                    Span::raw("┘"),
                ]));
            }
        }
        Role::Tool => {
            // Tool call - gray, indented
            let tool_text = format!("  > tool: {}", msg.content);
            for row in word_wrap_plain(&tool_text, w) {
                lines.push(Line::from(vec![Span::styled(
                    row,
                    Style::default().fg(Color::Rgb(100, 100, 100)).italic(),
                )]));
            }
        }
        Role::Error => {
            let time = msg.timestamp.format("%H:%M:%S").to_string();

            // Error header
            lines.push(Line::from(vec![
                Span::raw("┌ "),
                Span::styled(
                    "ERROR",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(time, Style::default().fg(Color::Rgb(100, 100, 100))),
            ]));

            // Error content with border - wrap at width-2 (for "│ " prefix)
            for content_line in msg.content.lines() {
                let wrapped = word_wrap_plain(content_line, w.saturating_sub(2));
                for line in wrapped {
                    lines.push(Line::from(vec![
                        Span::raw("│ "),
                        Span::styled(line, Style::default().fg(Color::Red)),
                    ]));
                }
            }
            // Add closing border line if we have content lines
            let content_lines = msg.content.lines().count();
            if content_lines > 0 {
                lines.push(Line::from(vec![
                    Span::raw("└"),
                    Span::raw("─".repeat(w.saturating_sub(2))),
                    Span::raw("┘"),
                ]));
            }
        }
    }

    lines
}

/// Show the live streaming assistant text, auto-scrolled to the bottom of the area.
#[allow(dead_code)]
fn render_streaming(frame: &mut Frame, app: &App, area: Rect) {
    if app.streaming_buffer.is_empty() && area.height == 0 {
        return;
    }
    let width = area.width as usize;
    let mut all_lines: Vec<Line<'static>> = vec![];

    // Streaming header with border
    all_lines.push(Line::from(vec![
        Span::raw("┌ "),
        Span::styled(
            "Jarvis [streaming]",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    for content_line in app.streaming_buffer.lines() {
        for row in word_wrap_plain(&format!("│ {content_line}"), width) {
            all_lines.push(Line::raw(row));
        }
    }

    // Add closing border line (always show to maintain visual consistency)
    all_lines.push(Line::from(vec![
        Span::raw("└"),
        Span::raw("─".repeat(width - 2)),
        Span::raw("┘"),
    ]));

    // Clip to the last `area.height` rows (auto-scroll to bottom).
    let skip = all_lines.len().saturating_sub(area.height as usize);
    let display = Text::from(all_lines[skip..].to_vec());
    frame.render_widget(Paragraph::new(display), area);
}

/// Render the text input — no border, full width.
fn render_input(frame: &mut Frame, app: &App, area: Rect) {
    let width = area.width as usize;

    let text = if app.input.is_empty() {
        Text::from(Line::from(vec![
            Span::styled("┌ ", Style::default().fg(Color::Rgb(100, 100, 100))),
            Span::styled(
                "Type a message... (Enter to send)",
                Style::default().fg(Color::Rgb(100, 100, 100)),
            ),
        ]))
    } else {
        let chars: Vec<char> = app.input.chars().collect();
        let lines: Vec<Line> = if width == 0 {
            vec![Line::from(vec![
                Span::styled("│ ", Style::default().fg(Color::Rgb(100, 100, 100))),
                Span::raw(app.input.as_str()),
            ])]
        } else {
            chars
                .chunks(width)
                .map(|chunk| {
                    Line::from(vec![
                        Span::styled("│ ", Style::default().fg(Color::Rgb(100, 100, 100))),
                        Span::raw(chunk.iter().collect::<String>()),
                    ])
                })
                .collect()
        };
        Text::from(lines)
    };

    frame.render_widget(Paragraph::new(text), area);

    // Position cursor - account for "│ " prefix (2 chars)
    let char_pos = app.input[..app.cursor].chars().count();
    let (row, col) = if width == 0 {
        (0u16, 2u16 + char_pos as u16)
    } else {
        let prefix_offset = 2; // "│ " is 2 characters
        let line_num = char_pos / width;
        let col_in_line = char_pos % width;
        (line_num as u16, prefix_offset as u16 + col_in_line as u16)
    };
    frame.set_cursor_position((area.x + col, area.y + row));
}

/// Render the status bar at the bottom of the viewport.
fn render_status(frame: &mut Frame, app: &App, area: Rect) {
    let (state_label, state_color) = match app.state {
        PipelineState::Idle => ("● IDLE", Color::Rgb(100, 100, 100)),
        PipelineState::Listening => ("● LISTENING", Color::Green),
        PipelineState::Transcribing => ("● TRANSCRIBING", Color::Yellow),
        PipelineState::Thinking => ("● THINKING", Color::Blue),
        PipelineState::Speaking => ("● SPEAKING", Color::Magenta),
    };

    let tts_label = if app.tts_enabled { "TTS ON" } else { "TTS OFF" };
    let tts_color = if app.tts_enabled {
        Color::Green
    } else {
        Color::Rgb(100, 100, 100)
    };

    let (conv_label, conv_color) = match *app.conv_mode.lock().unwrap() {
        ConversationMode::Active => ("ACTIVE", Color::Cyan),
        ConversationMode::Ambient => ("AMBIENT", Color::Rgb(100, 100, 100)),
        ConversationMode::AmbientLocked => ("AMBIENT🔒", Color::Yellow),
    };

    let text = Text::from(vec![Line::from(vec![
        Span::styled(
            " voicebot ",
            Style::default().fg(Color::Rgb(200, 200, 200)).bold(),
        ),
        Span::raw(" "),
        Span::styled(state_label, Style::default().fg(state_color)),
        Span::raw(" │ "),
        Span::styled(tts_label, Style::default().fg(tts_color)),
        Span::raw(" │ "),
        Span::styled(conv_label, Style::default().fg(conv_color)),
        Span::raw(" │ "),
        Span::styled(
            "Ctrl+T: toggle TTS  Esc: quit",
            Style::default().fg(Color::Rgb(100, 100, 100)),
        ),
    ])]);

    let block = Block::default().style(Style::default().bg(Color::Rgb(40, 40, 50)));

    frame.render_widget(Paragraph::new(text).block(block), area);
}

/// Number of visual rows the input text occupies with hard-wrap at `width`.
fn input_display_lines(input: &str, width: usize) -> usize {
    if width == 0 || input.is_empty() {
        return 1;
    }
    let char_count = input.chars().count();
    char_count.div_ceil(width)
}

/// Word-wrap `text` to `width` columns. Returns one owned `String` per visual row.
fn word_wrap_plain(text: &str, width: usize) -> Vec<String> {
    if width == 0 || text.is_empty() {
        return vec![text.to_string()];
    }

    let mut rows: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_w: usize = 0;

    let content = text.trim_start_matches(' ');
    let leading = text.len() - content.len();
    for _ in 0..leading {
        if current_w < width {
            current.push(' ');
            current_w += 1;
        } else {
            rows.push(std::mem::take(&mut current));
            current.push(' ');
            current_w = 1;
        }
    }

    let mut after_leading = leading > 0;

    for word in content.split_whitespace() {
        let ww = word.chars().count();
        if after_leading {
            after_leading = false;
            if current_w + ww <= width {
                current.push_str(word);
                current_w += ww;
            } else {
                rows.push(std::mem::take(&mut current));
                current_w = 0;
                place_word_at_row_start(&mut rows, &mut current, &mut current_w, word, ww, width);
            }
        } else if current_w == 0 {
            place_word_at_row_start(&mut rows, &mut current, &mut current_w, word, ww, width);
        } else if current_w + 1 + ww <= width {
            current.push(' ');
            current.push_str(word);
            current_w += 1 + ww;
        } else {
            rows.push(std::mem::take(&mut current));
            current_w = 0;
            place_word_at_row_start(&mut rows, &mut current, &mut current_w, word, ww, width);
        }
    }

    if !current.is_empty() || rows.is_empty() {
        rows.push(current);
    }
    rows
}

fn place_word_at_row_start(
    rows: &mut Vec<String>,
    current: &mut String,
    current_w: &mut usize,
    word: &str,
    ww: usize,
    width: usize,
) {
    if ww <= width {
        current.push_str(word);
        *current_w = ww;
    } else {
        for ch in word.chars() {
            if *current_w >= width {
                rows.push(std::mem::take(current));
                *current_w = 0;
            }
            current.push(ch);
            *current_w += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_returns_one_row() {
        assert_eq!(word_wrap_plain("", 80), vec![""]);
    }

    #[test]
    fn short_line_fits_in_one_row() {
        assert_eq!(word_wrap_plain("hello world", 80), vec!["hello world"]);
    }

    #[test]
    fn line_exactly_at_width_is_one_row() {
        assert_eq!(word_wrap_plain("ab cd", 5), vec!["ab cd"]);
    }

    #[test]
    fn line_one_char_over_wraps_to_two_rows() {
        assert_eq!(word_wrap_plain("ab cde", 5), vec!["ab", "cde"]);
    }

    #[test]
    fn long_line_wraps_correctly() {
        let text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj";
        assert_eq!(
            word_wrap_plain(text, 20),
            vec!["aaaa bbbb cccc dddd", "eeee ffff gggg hhhh", "iiii jjjj"]
        );
    }

    #[test]
    fn word_wider_than_width_is_hard_wrapped() {
        assert_eq!(word_wrap_plain("abcdefghij", 4), vec!["abcd", "efgh", "ij"]);
    }

    #[test]
    fn indented_line_preserves_leading_spaces() {
        assert_eq!(word_wrap_plain("  hello world", 80), vec!["  hello world"]);
    }

    #[test]
    fn indented_line_counts_spaces_in_width() {
        assert_eq!(word_wrap_plain("  ab cd", 6), vec!["  ab", "cd"]);
    }

    #[test]
    fn zero_width_returns_original() {
        assert_eq!(word_wrap_plain("hello world", 0), vec!["hello world"]);
    }
}
