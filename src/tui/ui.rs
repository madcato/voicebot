use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::Paragraph,
};

use super::app::{App, ChatMessage, Role};
use super::events::{InputSource, PipelineState};
use crate::tools::ConversationMode;

/// Height of the inline viewport (streaming preview + input + status bar).
pub const VIEWPORT_HEIGHT: u16 = 12;

/// Render the inline viewport: [streaming area] [input] [status bar].
pub fn render(frame: &mut Frame, app: &App) {
    let total = frame.area();
    let width = total.width as usize;

    // Input height: wraps at terminal width (no border so full width available).
    let display_lines = input_display_lines(&app.input, width);
    let input_height = (display_lines as u16).max(1).min(4);

    // Streaming area gets whatever space is left above input + status bar.
    let streaming_height = total.height.saturating_sub(input_height + 1);

    let [streaming_area, input_area, status_area] = Layout::vertical([
        Constraint::Length(streaming_height),
        Constraint::Length(input_height),
        Constraint::Length(1),
    ])
    .areas(total);

    render_streaming(frame, app, streaming_area);
    render_input(frame, app, input_area);
    render_status(frame, app, status_area);
}

/// Build display lines for a finalized message (used by mod.rs for insert_before).
pub fn message_lines(msg: &ChatMessage, width: u16) -> Vec<Line<'static>> {
    let w = width as usize;
    let mut lines: Vec<Line<'static>> = vec![Line::raw("")]; // blank spacer

    match &msg.role {
        Role::User(source) => {
            let source_label = match source {
                InputSource::Voice => " (voice)",
                InputSource::Text => "",
            };
            let time = msg.timestamp.format("%H:%M:%S").to_string();
            lines.push(Line::from(vec![
                Span::styled(
                    format!("You{source_label} "),
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
                Span::styled(time, Style::default().fg(Color::DarkGray)),
            ]));
            for content_line in msg.content.lines() {
                for row in word_wrap_plain(&format!("  {content_line}"), w) {
                    lines.push(Line::raw(row));
                }
            }
        }
        Role::Assistant => {
            let time = msg.timestamp.format("%H:%M:%S").to_string();
            lines.push(Line::from(vec![
                Span::styled(
                    "Assistant ",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(time, Style::default().fg(Color::DarkGray)),
            ]));
            for content_line in msg.content.lines() {
                for row in word_wrap_plain(&format!("  {content_line}"), w) {
                    lines.push(Line::raw(row));
                }
            }
        }
        Role::Tool => {
            let tool_text = format!("  > tool: {}", msg.content);
            for row in word_wrap_plain(&tool_text, w) {
                lines.push(Line::from(vec![Span::styled(
                    row,
                    Style::default().fg(Color::DarkGray).italic(),
                )]));
            }
        }
        Role::Error => {
            let time = msg.timestamp.format("%H:%M:%S").to_string();
            lines.push(Line::from(vec![
                Span::styled(
                    "ERROR ",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
                Span::styled(time, Style::default().fg(Color::DarkGray)),
            ]));
            for content_line in msg.content.lines() {
                for row in word_wrap_plain(&format!("  {content_line}"), w) {
                    lines.push(Line::from(vec![Span::styled(
                        row,
                        Style::default().fg(Color::Red),
                    )]));
                }
            }
        }
    }

    lines
}

/// Show the live streaming assistant text, auto-scrolled to the bottom of the area.
fn render_streaming(frame: &mut Frame, app: &App, area: Rect) {
    if app.streaming_buffer.is_empty() || area.height == 0 {
        return;
    }
    let width = area.width as usize;
    let mut all_lines: Vec<Line<'static>> = vec![Line::from(vec![
        Span::styled(
            "Assistant ",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("[streaming...]", Style::default().fg(Color::Yellow)),
    ])];
    for content_line in app.streaming_buffer.lines() {
        for row in word_wrap_plain(&format!("  {content_line}"), width) {
            all_lines.push(Line::raw(row));
        }
    }

    // Clip to the last `area.height` rows (auto-scroll to bottom).
    let skip = all_lines.len().saturating_sub(area.height as usize);
    let display = Text::from(all_lines[skip..].to_vec());
    frame.render_widget(Paragraph::new(display), area);
}

/// Render the text input — no border, full width.
fn render_input(frame: &mut Frame, app: &App, area: Rect) {
    let width = area.width as usize;

    let text = if app.input.is_empty() {
        Text::from(Line::from(vec![Span::styled(
            "Type a message... (Enter to send)",
            Style::default().fg(Color::DarkGray),
        )]))
    } else {
        let chars: Vec<char> = app.input.chars().collect();
        let lines: Vec<Line> = if width == 0 {
            vec![Line::raw(app.input.as_str())]
        } else {
            chars
                .chunks(width)
                .map(|chunk| Line::raw(chunk.iter().collect::<String>()))
                .collect()
        };
        Text::from(lines)
    };

    frame.render_widget(Paragraph::new(text), area);

    // Position cursor.
    let char_pos = app.input[..app.cursor].chars().count();
    let (row, col) = if width == 0 {
        (0u16, char_pos as u16)
    } else {
        ((char_pos / width) as u16, (char_pos % width) as u16)
    };
    frame.set_cursor_position((area.x + col, area.y + row));
}

/// Render the status bar at the bottom of the viewport.
fn render_status(frame: &mut Frame, app: &App, area: Rect) {
    let (state_label, state_color) = match app.state {
        PipelineState::Idle => ("IDLE", Color::DarkGray),
        PipelineState::Listening => ("LISTENING", Color::Green),
        PipelineState::Transcribing => ("TRANSCRIBING", Color::Yellow),
        PipelineState::Thinking => ("THINKING", Color::Blue),
        PipelineState::Speaking => ("SPEAKING", Color::Magenta),
    };

    let tts_label = if app.tts_enabled { "TTS ON" } else { "TTS OFF" };
    let tts_color = if app.tts_enabled {
        Color::Green
    } else {
        Color::DarkGray
    };

    let (conv_label, conv_color) = match *app.conv_mode.lock().unwrap() {
        ConversationMode::Active => ("ACTIVE", Color::Cyan),
        ConversationMode::Ambient => ("AMBIENT", Color::DarkGray),
        ConversationMode::AmbientLocked => ("AMBIENT🔒", Color::Yellow),
    };

    let bar = Line::from(vec![
        Span::raw(" voicebot "),
        Span::raw("| "),
        Span::styled(state_label, Style::default().fg(state_color).bold()),
        Span::raw(" | "),
        Span::styled(tts_label, Style::default().fg(tts_color)),
        Span::raw(" | "),
        Span::styled(conv_label, Style::default().fg(conv_color).bold()),
        Span::raw(" | "),
        Span::styled(
            "Ctrl+T: toggle TTS  Esc: quit",
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    frame.render_widget(
        Paragraph::new(bar).style(Style::default().bg(Color::Rgb(30, 30, 40))),
        area,
    );
}

/// Number of visual rows the input text occupies with hard-wrap at `width`.
fn input_display_lines(input: &str, width: usize) -> usize {
    if width == 0 || input.is_empty() {
        return 1;
    }
    let char_count = input.chars().count();
    (char_count + width - 1) / width
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
        assert_eq!(
            word_wrap_plain("abcdefghij", 4),
            vec!["abcd", "efgh", "ij"]
        );
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
