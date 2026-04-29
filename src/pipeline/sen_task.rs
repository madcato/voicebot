use std::sync::Arc;
use std::sync::atomic::Ordering;
use tracing::info;

use crate::tts::SentenceSplitter;
use super::state::{SharedSession, PipelineEvents};

/// SEN task: blocks on LLM_POST_RECEIVED, splits assistant_text into sentences.
pub async fn sen_task(shared: Arc<SharedSession>, events: Arc<PipelineEvents>) {
    let mut cancel_rx = events.cancel_tx.subscribe();
    let mut splitter = SentenceSplitter::new();
    let mut first_sentence_logged = false;

    loop {
        let cancelled = tokio::select! {
            _ = events.llm_post_received.notified() => false,
            _ = cancel_rx.recv() => true,
        };

        if cancelled {
            shared.assistant_text.lock().unwrap().clear();
            splitter = SentenceSplitter::new();
            first_sentence_logged = false;
            while cancel_rx.try_recv().is_ok() {}
            continue;
        }

        let new_text = std::mem::take(&mut *shared.assistant_text.lock().unwrap());

        let mut ready_sentences: Vec<String> = Vec::new();
        if !new_text.is_empty()
            && let Some(s) = splitter.push(&new_text)
        {
            ready_sentences.push(s);
        }

        if shared.llm_post_finished.load(Ordering::SeqCst)
            && let Some(s) = splitter.flush()
        {
            ready_sentences.push(s);
        }

        for sentence in ready_sentences {
            if !first_sentence_logged {
                first_sentence_logged = true;
                if let Some(t0) = shared.t_vad_end.lock().unwrap().as_ref() {
                    let tts_queue_ms = t0.elapsed().as_millis();
                    info!(target: "performance", "[+{}ms] first sentence → TTS queue", tts_queue_ms);
                    if let Some(t_llm_sent) = shared.t_llm_post_send.lock().unwrap().as_ref() {
                        info!(target: "performance", "  └─ LLM processing: {}ms", t_llm_sent.elapsed().as_millis());
                    }
                }
            }
            shared.sentences.lock().unwrap().push_back(sentence);
            events.sentence_ready.notify_one();
        }
    }
}
