use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use super::app::{App, Role};
use super::events::{InputSource, PipelineState};

/// Render the entire TUI frame.
pub fn render(frame: &mut Frame, app: &App) {
    let [header_area, conversation_area, input_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(4),
        Constraint::Length(3),
    ])
    .areas(frame.area());

    render_header(frame, app, header_area);
    render_conversation(frame, app, conversation_area);
    render_input(frame, app, input_area);
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
    let mut lines: Vec<Line> = Vec::new();

    for msg in &app.messages {
        lines.push(Line::raw("")); // spacer
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
                    lines.push(Line::raw(format!("  {content_line}")));
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
                    lines.push(Line::raw(format!("  {content_line}")));
                }
            }
            Role::Tool => {
                lines.push(Line::from(vec![Span::styled(
                    format!("  > tool: {}", msg.content),
                    Style::default().fg(Color::DarkGray).italic(),
                )]));
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
                    lines.push(Line::from(vec![Span::styled(
                        format!("  {content_line}"),
                        Style::default().fg(Color::Red),
                    )]));
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
            lines.push(Line::raw(format!("  {content_line}")));
        }
    }

    let text = Text::from(lines);
    let visible_height = area.height.saturating_sub(2) as usize; // minus borders
    let visible_width = area.width.saturating_sub(2) as usize; // minus borders

    // Count wrapped lines accurately using a word-wrap simulation that matches
    // ratatui's Wrap { trim: false } algorithm (greedy word packing per row).
    // The old div_ceil estimate was always too low because word-wrap wastes space
    // at row boundaries, making content_height > char_count / width.
    let content_height: usize = text.lines.iter()
        .map(|line| count_wrapped_lines(line, visible_width))
        .sum();

    let max_scroll = content_height.saturating_sub(visible_height) as u16;

    // app.scroll == 0 → show bottom; app.scroll > 0 → N lines above bottom.
    let scroll = if app.scroll == 0 {
        max_scroll
    } else {
        max_scroll.saturating_sub(app.scroll)
    };

    let conversation = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));

    frame.render_widget(conversation, area);
}

fn render_input(frame: &mut Frame, app: &App, area: Rect) {
    let display = if app.input.is_empty() {
        Line::from(vec![Span::styled(
            "Type a message... (Enter to send)",
            Style::default().fg(Color::DarkGray),
        )])
    } else {
        Line::raw(&app.input)
    };

    let input = Paragraph::new(display)
        .block(Block::default().borders(Borders::ALL).title(" Input "));

    frame.render_widget(input, area);

    // Show cursor at character (not byte) position.
    let char_pos = app.input[..app.cursor].chars().count() as u16;
    let cursor_x = area.x + 1 + char_pos;
    let cursor_y = area.y + 1;
    frame.set_cursor_position((cursor_x, cursor_y));
}

/// Count how many terminal rows `line` occupies when word-wrapped to `width`.
///
/// Simulates ratatui's `Wrap { trim: false }` greedy word-packing algorithm.
/// Key detail: uses `split_whitespace()` (not `split(' ')`) so that leading
/// spaces ("  message content") and consecutive spaces don't produce empty
/// tokens that incorrectly increment the row counter.
/// Leading whitespace is accounted for separately as initial row width.
fn count_wrapped_lines(line: &Line<'_>, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    if text.is_empty() {
        return 1;
    }

    // Leading whitespace consumes space on the first row before any word.
    let trimmed = text.trim_start_matches(' ');
    let leading = text.len() - trimmed.len();

    let mut rows = 1usize; // always at least one row
    let mut row_width = leading.min(width);
    if leading >= width {
        rows += leading / width;
        row_width = leading % width;
    }

    for word in trimmed.split_whitespace() {
        let ww = word.chars().count();

        if row_width == 0 {
            // New row already counted — place the word (hard-wrap if needed).
            let extra = ww.saturating_sub(1) / width;
            rows += extra;
            row_width = ww - extra * width;
            if row_width == 0 {
                row_width = width;
            }
        } else if row_width + 1 + ww <= width {
            row_width += 1 + ww;
        } else {
            // Word doesn't fit — start a new row.
            rows += 1;
            let extra = ww.saturating_sub(1) / width;
            rows += extra;
            row_width = ww - extra * width;
            if row_width == 0 {
                row_width = width;
            }
        }
    }

    rows
}

#[cfg(test)]
mod tests {
    use ratatui::text::Span;
    use super::*;

    fn line(s: &str) -> Line<'static> {
        Line::from(vec![Span::raw(s.to_string())])
    }

    #[test]
    fn empty_line_is_one_row() {
        assert_eq!(count_wrapped_lines(&Line::raw(""), 80), 1);
    }

    #[test]
    fn short_line_fits_in_one_row() {
        assert_eq!(count_wrapped_lines(&line("hello world"), 80), 1);
    }

    #[test]
    fn line_exactly_at_width_is_one_row() {
        // "ab cd" = 5 chars, width 5
        assert_eq!(count_wrapped_lines(&line("ab cd"), 5), 1);
    }

    #[test]
    fn line_one_char_over_wraps_to_two_rows() {
        // "ab cde" = 6 chars, width 5 → "ab" fits, "cde" doesn't fit with space → row 2
        assert_eq!(count_wrapped_lines(&line("ab cde"), 5), 2);
    }

    #[test]
    fn long_line_wraps_correctly() {
        // 10 words of 4 chars each at width 20: "aaaa bbbb cccc dddd" = 19 fits, next word wraps
        let text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj";
        // width=20: "aaaa bbbb cccc dddd" = 19 chars → fits; "eeee" = need 24 → row 2; etc.
        assert_eq!(count_wrapped_lines(&line(text), 20), 3);
    }

    #[test]
    fn word_wider_than_width_is_hard_wrapped() {
        // one 10-char word, width 4 → ceil(10/4) = 3 rows
        assert_eq!(count_wrapped_lines(&line("abcdefghij"), 4), 3);
    }

    #[test]
    fn indented_line_does_not_add_extra_rows() {
        // "  hello" has 2 leading spaces — must NOT produce phantom rows for them.
        assert_eq!(count_wrapped_lines(&line("  hello"), 80), 1);
        assert_eq!(count_wrapped_lines(&line("  hello world"), 80), 1);
    }

    #[test]
    fn indented_line_accounts_for_leading_spaces_in_width() {
        // "  ab cd" = 2 spaces + "ab cd"; width=6: "  ab c" fits (6), "d" wraps → 2 rows
        assert_eq!(count_wrapped_lines(&line("  ab cd"), 6), 2);
    }
}
