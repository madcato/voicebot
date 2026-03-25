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
    let visible_height = area.height.saturating_sub(2); // account for borders

    // With Wrap enabled, actual rendered height can exceed text.lines.len().
    // Estimate wrapped height: sum of ceil(line_width / visible_width) for each line.
    let visible_width = area.width.saturating_sub(2) as usize; // account for borders
    let content_height: u16 = if visible_width == 0 {
        text.lines.len() as u16
    } else {
        text.lines
            .iter()
            .map(|line| {
                let w: usize = line.width();
                if w == 0 { 1 } else { w.div_ceil(visible_width) as u16 }
            })
            .sum()
    };

    let max_scroll = content_height.saturating_sub(visible_height);
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
