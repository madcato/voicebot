mod app;
pub mod events;
mod input;
mod ui;

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use crossterm::{
    execute,
    event::{DisableMouseCapture, EnableMouseCapture},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use crate::{PipelineEvents, SharedSession};
use app::{Action, App};
use events::TuiEventRx;
use input::KeyReader;

const TICK_MS: u64 = 33; // ~30fps

/// Run the TUI event loop. Blocks until the user quits.
pub async fn run(
    mut event_rx: TuiEventRx,
    shared: Arc<SharedSession>,
    events: Arc<PipelineEvents>,
    tts_muted: Arc<AtomicBool>,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let mut keys = KeyReader::new();
    let tick = tokio::time::Duration::from_millis(TICK_MS);

    loop {
        terminal.draw(|frame| ui::render(frame, &app))?;

        // Wait for next event: pipeline event, key press, or tick (for redraw).
        tokio::select! {
            Some(tui_event) = event_rx.recv() => {
                app.handle_tui_event(tui_event);
                // Drain any additional buffered events to batch updates before redraw.
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
                                    // Mark as text input so llm_task can label it.
                                    shared.text_input_pending.store(true, Ordering::SeqCst);
                                    *shared.transliterated_text.lock().unwrap() = text;
                                    events.vad_finish.notify_one();
                                }
                                Action::ToggleTts => {
                                    let was_muted = tts_muted.load(Ordering::SeqCst);
                                    tts_muted.store(!was_muted, Ordering::SeqCst);
                                    app.tts_enabled = was_muted; // was muted → now enabled
                                }
                                Action::ScrollUp => {
                                    app.scroll = app.scroll.saturating_add(5);
                                }
                                Action::ScrollDown => {
                                    app.scroll = app.scroll.saturating_sub(5);
                                }
                            }
                        }
                    }
                    Ok(None) => { app.should_quit = true; }
                    Err(e) => { tracing::error!("Key reader error: {e}"); }
                }
            }
            _ = tokio::time::sleep(tick) => {
                // Tick — just redraw.
            }
        }

        if app.should_quit {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    Ok(())
}
