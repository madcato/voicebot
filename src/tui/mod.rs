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
use crossterm::event::{self, Event};
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::sync::mpsc;

use crate::pipeline::PipelineFrame;
use crate::tools::ConversationMode;
use app::{Action, App};
use events::{TuiEvent, TuiEventRx};
use input::KeyReader;

const TICK_MS: u64 = 33; // ~30fps

/// Run the TUI event loop. Blocks until the user quits.
pub async fn run(
    mut event_rx: TuiEventRx,
    transcript_tx: mpsc::Sender<PipelineFrame>,
    tts_muted: Arc<AtomicBool>,
    conv_mode: Arc<Mutex<ConversationMode>>,
) -> Result<()> {
      enable_raw_mode()?;
    execute!(io::stdout(), crossterm::terminal::EnterAlternateScreen)?;
    execute!(io::stdout(), crossterm::cursor::Hide)?;
    execute!(io::stdout(), crossterm::terminal::Clear(crossterm::terminal::ClearType::All))?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(conv_mode);
    let mut keys = KeyReader::new();
    let tick = tokio::time::Duration::from_millis(TICK_MS);

    // Send splash event on startup
    app.handle_tui_event(TuiEvent::Splash);

    loop {
        // Render to terminal - no manual clearing needed with proper viewport
        terminal.draw(|frame| ui::render(frame, &mut app))?;

        tokio::select! {
            Some(tui_event) = event_rx.recv() => {
                app.handle_tui_event(tui_event);
                while let Ok(ev) = event_rx.try_recv() {
                    app.handle_tui_event(ev);
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                // Check for resize events
                if event::poll(std::time::Duration::from_millis(0)).unwrap_or(false) {
                    if let Event::Resize(_, _) = event::read().unwrap_or(Event::Key(crossterm::event::KeyEvent::new(crossterm::event::KeyCode::Enter, crossterm::event::KeyModifiers::empty()))) {
                        // Clear and redraw on resize - re-render everything
                        terminal.clear().unwrap_or_default();
                    }
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
                                    transcript_tx.send(PipelineFrame::TextInput { text }).await.ok();
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
    execute!(io::stdout(), crossterm::terminal::LeaveAlternateScreen)?;
    disable_raw_mode()?;
    // Move cursor to a fresh line so the shell prompt appears cleanly.
    execute!(io::stdout(), crossterm::cursor::MoveToNextLine(1))?;
    Ok(())
}

/// No-op - messages are rendered directly by ui::render now
fn flush_new_messages(
    _terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    _app: &mut App,
) -> Result<()> {
    Ok(())
}
