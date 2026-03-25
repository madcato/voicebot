use anyhow::Result;
use crossterm::event::{Event, EventStream};
use futures_util::StreamExt;

/// Async wrapper around crossterm's event stream.
pub struct KeyReader {
    stream: EventStream,
}

impl KeyReader {
    pub fn new() -> Self {
        Self {
            stream: EventStream::new(),
        }
    }

    /// Wait for the next terminal event. Returns `None` on stream exhaustion.
    pub async fn next(&mut self) -> Result<Option<Event>> {
        match self.stream.next().await {
            Some(Ok(ev)) => Ok(Some(ev)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}
