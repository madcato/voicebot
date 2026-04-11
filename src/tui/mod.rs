mod app;
pub mod events;
mod input;
mod ui;

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, Viewport, backend::CrosstermBackend, widgets::Paragraph};

use crate::tools::ConversationMode;
use crate::{PipelineEvents, SharedSession};
use app::{Action, App};
use events::TuiEventRx;
use input::KeyReader;
use ui::VIEWPORT_HEIGHT;

const TICK_MS: u64 = 33; // ~30fps

/// Run the TUI event loop. Blocks until the user quits.
pub async fn run(
    mut event_rx: TuiEventRx,
    shared: Arc<SharedSession>,
    events: Arc<PipelineEvents>,
    tts_muted: Arc<AtomicBool>,
    conv_mode: Arc<Mutex<ConversationMode>>,
) -> Result<()> {
    enable_raw_mode()?;
    // No alternate screen — use inline viewport so terminal scrollback works.
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::with_options(
        backend,
        ratatui::TerminalOptions {
            viewport: Viewport::Inline(VIEWPORT_HEIGHT),
        },
    )?;

    let mut app = App::new(conv_mode);
    let mut keys = KeyReader::new();
    let tick = tokio::time::Duration::from_millis(TICK_MS);

    loop {
        // Push any new finalized messages to terminal scrollback via insert_before.
        flush_new_messages(&mut terminal, &mut app)?;

        terminal.draw(|frame| ui::render(frame, &app))?;

        tokio::select! {
            Some(tui_event) = event_rx.recv() => {
                app.handle_tui_event(tui_event);
                while let Ok(ev) = event_rx.try_recv() {
                    app.handle_tui_event(ev);
                }
            }
            key_result = keys.next() => {
                match key_result {
                    Ok(Some(event)) => {
                        if let Some(action) = app.handle_key_event(event) {
                            match action {
                                Action::Quit => {
                                    app.should_quit = true;
                                }
                                Action::Submit(text) => {
                                    shared.text_input_pending.store(true, Ordering::SeqCst);
                                    *shared.transliterated_text.lock().unwrap() = text;
                                    events.vad_finish.notify_one();
                                }
                                Action::ToggleTts => {
                                    let was_muted = tts_muted.load(Ordering::SeqCst);
                                    tts_muted.store(!was_muted, Ordering::SeqCst);
                                    app.tts_enabled = was_muted;
                                }
                            }
                        }
                    }
                    Ok(None) => { app.should_quit = true; }
                    Err(e) => { tracing::error!("Key reader error: {e}"); }
                }
            }
            _ = tokio::time::sleep(tick) => {}
        }

        if app.should_quit {
            break;
        }
    }

    // Flush any remaining messages before exiting.
    flush_new_messages(&mut terminal, &mut app)?;
    terminal.clear()?;
    disable_raw_mode()?;
    // Move cursor to a fresh line so the shell prompt appears cleanly.
    execute!(io::stdout(), crossterm::cursor::MoveToNextLine(1))?;
    Ok(())
}

/// Push messages that haven't been printed yet into the terminal scrollback buffer.
fn flush_new_messages(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    while app.printed_count < app.messages.len() {
        let msg = &app.messages[app.printed_count];
        let width = terminal.size()?.width;
        let lines = ui::message_lines(msg, width);
        let height = lines.len() as u16;
        terminal.insert_before(height, |buf| {
            let area = buf.area();
            use ratatui::widgets::Widget;
            let text = ratatui::text::Text::from(lines);
            Paragraph::new(text).render(*area, buf);
        })?;
        app.printed_count += 1;
    }
    Ok(())
}
