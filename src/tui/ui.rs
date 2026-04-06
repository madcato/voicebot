use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph},
};

use super::app::{App, Role};
use super::events::{InputSource, PipelineState};

/// Render the entire TUI frame.
pub fn render(frame: &mut Frame, app: &App) {
    let total = frame.area();
    // Inner width of the input box (subtract left + right borders).
    let inner_width = total.width.saturating_sub(2) as usize;
    let display_lines = input_display_lines(&app.input, inner_width);
    // Height = content lines + 2 borders; at least 3, capped at 10.
    let input_height = ((display_lines + 2) as u16).max(3).min(10);

    let [header_area, conversation_area, input_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(4),
        Constraint::Length(input_height),
    ])
    .areas(total);

    render_header(frame, app, header_area);
    render_conversation(frame, app, conversation_area);
    render_input(frame, app, input_area);
}

/// Number of visual rows the input text occupies with hard-wrap at `inner_width`.
fn input_display_lines(input: &str, inner_width: usize) -> usize {
    if inner_width == 0 || input.is_empty() {
        return 1;
    }
    let char_count = input.chars().count();
    (char_count + inner_width - 1) / inner_width
}

fn render_header(frame: &mut Frame, app: &App, area: Rect) {
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

    let header = Line::from(vec![
        Span::raw(" voicebot "),
        Span::raw("| "),
        Span::styled(state_label, Style::default().fg(state_color).bold()),
        Span::raw(" | "),
        Span::styled(tts_label, Style::default().fg(tts_color)),
        Span::raw(" | "),
        Span::styled(
            "Ctrl+T: toggle TTS  Esc: quit",
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    frame.render_widget(
        Paragraph::new(header).style(Style::default().bg(Color::Rgb(30, 30, 40))),
        area,
    );
}

fn render_conversation(frame: &mut Frame, app: &App, area: Rect) {
    let visible_height = area.height.saturating_sub(2) as usize;
    let visible_width = area.width.saturating_sub(2) as usize;

    // Build the full list of pre-wrapped visual rows. Each element is exactly
    // one terminal row, so `all_lines.len()` is the exact visual line count —
    // no approximation needed.
    let all_lines = build_visual_lines(app, visible_width);
    let total = all_lines.len();

    // scroll == 0 → auto-scroll to bottom.
    // scroll > 0  → user has scrolled N rows above the bottom.
    let scroll_pos = if app.scroll == 0 {
        total.saturating_sub(visible_height)
    } else {
        total.saturating_sub(visible_height.saturating_add(app.scroll as usize))
    };

    let display = Text::from(all_lines[scroll_pos..].to_vec());
    let conversation = Paragraph::new(display)
        .block(Block::default().borders(Borders::ALL));
    // No .wrap() — lines are already word-wrapped to visible_width.
    // No .scroll() — vector slicing handles the exact scroll position.

    frame.render_widget(conversation, area);
}

/// Build one `Line<'static>` per visual terminal row for the conversation view.
///
/// Content lines are pre-wrapped with [`word_wrap_plain`] so that
/// `result.len()` equals the exact number of rows ratatui will render.
/// Header lines (short, single row) are kept as styled multi-span Lines.
fn build_visual_lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    for msg in &app.messages {
        lines.push(Line::raw("")); // blank spacer before each message
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
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(time, Style::default().fg(Color::DarkGray)),
                ]));
                for content_line in msg.content.lines() {
                    for row in word_wrap_plain(&format!("  {content_line}"), width) {
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
                    for row in word_wrap_plain(&format!("  {content_line}"), width) {
                        lines.push(Line::raw(row));
                    }
                }
            }
            Role::Tool => {
                let tool_text = format!("  > tool: {}", msg.content);
                for row in word_wrap_plain(&tool_text, width) {
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
                        Style::default()
                            .fg(Color::Red)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(time, Style::default().fg(Color::DarkGray)),
                ]));
                for content_line in msg.content.lines() {
                    for row in word_wrap_plain(&format!("  {content_line}"), width) {
                        lines.push(Line::from(vec![Span::styled(
                            row,
                            Style::default().fg(Color::Red),
                        )]));
                    }
                }
            }
        }
    }

    // Streaming assistant message (if any).
    if !app.streaming_buffer.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::styled(
                "Assistant ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("[streaming...]", Style::default().fg(Color::Yellow)),
        ]));
        for content_line in app.streaming_buffer.lines() {
            for row in word_wrap_plain(&format!("  {content_line}"), width) {
                lines.push(Line::raw(row));
            }
        }
    }

    lines
}

fn render_input(frame: &mut Frame, app: &App, area: Rect) {
    let inner_width = area.width.saturating_sub(2) as usize;

    let text = if app.input.is_empty() {
        Text::from(Line::from(vec![Span::styled(
            "Type a message... (Enter to send)",
            Style::default().fg(Color::DarkGray),
        )]))
    } else {
        // Hard-wrap into rows of inner_width chars so cursor math stays exact.
        let chars: Vec<char> = app.input.chars().collect();
        let lines: Vec<Line> = if inner_width == 0 {
            vec![Line::raw(app.input.as_str())]
        } else {
            chars
                .chunks(inner_width)
                .map(|chunk| Line::raw(chunk.iter().collect::<String>()))
                .collect()
        };
        Text::from(lines)
    };

    let input = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title(" Input "));

    frame.render_widget(input, area);

    // Compute cursor row + column from char offset, respecting hard-wrap width.
    let char_pos = app.input[..app.cursor].chars().count();
    let (row, col) = if inner_width == 0 {
        (0u16, char_pos as u16)
    } else {
        ((char_pos / inner_width) as u16, (char_pos % inner_width) as u16)
    };
    frame.set_cursor_position((area.x + 1 + col, area.y + 1 + row));
}

/// Word-wrap `text` to `width` columns. Returns one owned `String` per visual row.
///
/// Uses greedy word-fill. Leading spaces are preserved on the first row
/// (matches ratatui `Wrap { trim: false }` semantics). Words exceeding
/// `width` are hard-wrapped at the character boundary.
/// Consecutive interior whitespace is collapsed to a single space.
fn word_wrap_plain(text: &str, width: usize) -> Vec<String> {
    if width == 0 || text.is_empty() {
        return vec![text.to_string()];
    }

    let mut rows: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_w: usize = 0;

    // Preserve leading ASCII spaces on the first row.
    // (Content lines are always "  {text}" — prefix is pure ASCII.)
    let content = text.trim_start_matches(' ');
    let leading = text.len() - content.len(); // byte count == char count for spaces
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

    // `after_leading` is true when we have placed leading spaces but no word yet.
    // The first word is appended directly (no separator space) because the original
    // text has no space between the indentation and the first word.
    let mut after_leading = leading > 0;

    for word in content.split_whitespace() {
        let ww = word.chars().count();
        if after_leading {
            // First word on the indented row — no separator space before it.
            after_leading = false;
            if current_w + ww <= width {
                current.push_str(word);
                current_w += ww;
            } else {
                // Indentation + word overflow: flush indentation, start fresh.
                rows.push(std::mem::take(&mut current));
                current_w = 0;
                place_word_at_row_start(&mut rows, &mut current, &mut current_w, word, ww, width);
            }
        } else if current_w == 0 {
            // Beginning of a new row — hard-wrap if the word exceeds width.
            place_word_at_row_start(&mut rows, &mut current, &mut current_w, word, ww, width);
        } else if current_w + 1 + ww <= width {
            // Word fits on the current row with a leading space.
            current.push(' ');
            current.push_str(word);
            current_w += 1 + ww;
        } else {
            // Word doesn't fit — start a new row.
            rows.push(std::mem::take(&mut current));
            current_w = 0;
            place_word_at_row_start(&mut rows, &mut current, &mut current_w, word, ww, width);
        }
    }

    // Emit whatever remains (always at least one row).
    if !current.is_empty() || rows.is_empty() {
        rows.push(current);
    }
    rows
}

/// Append `word` to the start of an empty row, hard-wrapping at char boundaries
/// when the word is wider than `width`.
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
        // width=20: "aaaa bbbb cccc dddd" = 19 fits; "eeee" wraps → row 2 etc.
        assert_eq!(
            word_wrap_plain(text, 20),
            vec!["aaaa bbbb cccc dddd", "eeee ffff gggg hhhh", "iiii jjjj"]
        );
    }

    #[test]
    fn word_wider_than_width_is_hard_wrapped() {
        // 10-char word at width=4 → ceil(10/4) = 3 rows
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
        // "  ab cd" at width=6: "  ab c" = 7 chars > 6, so "  ab" + "cd" ...
        // Actually: leading=2, then "ab"(2) + "cd"(2).
        // current="  ", w=2; "ab": 2+1+2=5<=6, current="  ab", w=5;
        // "cd": 5+1+2=8>6 → new row → ["  ab", "cd"]
        assert_eq!(word_wrap_plain("  ab cd", 6), vec!["  ab", "cd"]);
    }

    #[test]
    fn zero_width_returns_original() {
        assert_eq!(word_wrap_plain("hello world", 0), vec!["hello world"]);
    }
}
