use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::sync::atomic::Ordering;
use tracing::{debug, info, warn};

use crate::db::{Database, Memory};
use crate::llm::{OpenAIClient, LlmSession};
use crate::memory::{build_memory_context, extract_memories};
use crate::profile::{build_profile_context, extract_facts, ProfileFact};
use super::state::{SharedSession, PipelineEvents};

/// Assemble the full system prompt from its components.
///
/// Order: base prompt → [USER PROFILE] → [MEMORIES] → tool instructions.
pub fn build_system_prompt(
    base_prompt: &str,
    profile_facts: &[ProfileFact],
    memories: &[Memory],
    tool_section: &str,
) -> String {
    format!(
        "{}{}{}{}",
        base_prompt,
        build_profile_context(profile_facts),
        build_memory_context(memories),
        tool_section,
    )
}

/// Core consolidation work: extract profile facts + memories, summarize old
/// turns, rebuild the system prompt, and apply the compacted session.
///
/// Called both by `consolidation_task` (recurring) and at startup when the
/// context already exceeds `LLM_IDLE_MIN_CONTEXT_PCT`.
#[allow(clippy::too_many_arguments)]
pub async fn run_consolidation_cycle(
    background_client: &OpenAIClient,
    db: &Database,
    session_id: uuid::Uuid,
    llm_session: &Arc<Mutex<LlmSession>>,
    keep_turns: usize,
    base_prompt: &str,
    tool_section: &str,
) {
    let (conversation_text, summary_prompt, turns_to_summarize) = {
        let s = llm_session.lock().unwrap();
        let count = s.summarizable_turn_count(keep_turns);
        let prompt = s.build_summary_prompt(keep_turns);
        let mut conv = String::new();
        for msg in &s.messages[..count.min(s.messages.len())] {
            if let (Some(role), Some(content)) =
                (msg["role"].as_str(), msg["content"].as_str())
                && (role == "user" || role == "assistant")
            {
                conv.push_str(role);
                conv.push_str(": ");
                conv.push_str(content);
                conv.push_str("\n\n");
            }
        }
        (conv, prompt, count)
    };

    // Profile facts.
    if !conversation_text.is_empty() {
        let facts = extract_facts(background_client, &conversation_text, "").await;
        for fact in facts {
            if let Err(e) = db.upsert_profile_fact(&fact.key, &fact.value, fact.confidence).await {
                warn!(target: "profile", "Failed to save profile fact '{}': {}", fact.key, e);
            } else {
                debug!(target: "profile", "Profile: {} = {} ({:.0}%)", fact.key, fact.value, fact.confidence * 100.0);
            }
        }
    }

    // Persistent memories.
    let existing_memories = db.load_active_memories().await.unwrap_or_default();
    let mem_result = extract_memories(background_client, &conversation_text, &existing_memories).await;
    for id in &mem_result.archive_ids {
        if let Err(e) = db.deactivate_memory(*id).await {
            warn!(target: "memory", "Failed to archive memory id={}: {}", id, e);
        }
    }
    if !mem_result.new_memories.is_empty() {
        info!(target: "memory", "Extracted {} new memories", mem_result.new_memories.len());
        if let Err(e) = db.save_memories_batch(&mem_result.new_memories, session_id).await {
            warn!(target: "memory", "Failed to save memories: {}", e);
        }
    }
    if !mem_result.archive_ids.is_empty() {
        info!(target: "memory", "Archived {} outdated memories", mem_result.archive_ids.len());
    }

    // Summarize.
    let summary = if let Some(prompt) = summary_prompt {
        match background_client.complete(&prompt).await {
            Ok(s) if !s.is_empty() => {
                info!(target: "memory", "Summary: {}", s);
                Some(s)
            }
            Ok(_) => {
                warn!(target: "memory", "Summarization returned empty result");
                None
            }
            Err(e) => {
                warn!(target: "memory", "Summarization failed: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Persist summary and rebuild system prompt.
    if let Some(ref summary_text) = summary {
        let prev_through_id = db.get_summary_through_id(session_id).await.unwrap_or(0);
        let through_id = db
            .get_message_id_at_offset(session_id, prev_through_id, turns_to_summarize.saturating_sub(1))
            .await
            .ok()
            .flatten()
            .unwrap_or(0);
        if through_id > 0 {
            if let Err(e) = db.save_summary(session_id, summary_text, through_id).await {
                warn!(target: "db", "Failed to persist summary: {}", e);
            }
        }
    }

    let fresh_profile = db.load_user_profile().await.unwrap_or_default();
    let fresh_profile_facts: Vec<ProfileFact> = fresh_profile
        .into_iter()
        .map(|(key, value, confidence)| ProfileFact { key, value, confidence })
        .collect();
    let fresh_memories = db.load_active_memories().await.unwrap_or_default();
    let new_system_prompt = build_system_prompt(
        base_prompt, &fresh_profile_facts, &fresh_memories, tool_section,
    );

    {
        let mut s = llm_session.lock().unwrap();
        if let Some(ref summary_text) = summary {
            s.apply_summary(summary_text, keep_turns);
        }
        s.set_system_prompt(new_system_prompt);
    }

    info!(
        target: "memory",
        "Consolidation complete — prompt rebuilt ({} profile facts, {} memories, {} recent turns kept)",
        fresh_profile_facts.len(), fresh_memories.len(), keep_turns,
    );
}

/// Context consolidation task: blocks on LLM_POST_FINISHED, runs a full
/// memory consolidation cycle when the context window approaches its limit.
#[allow(clippy::too_many_arguments)]
pub async fn consolidation_task(
    shared: Arc<SharedSession>,
    events: Arc<PipelineEvents>,
    llm_session: Arc<Mutex<LlmSession>>,
    background_client: OpenAIClient,
    db: Database,
    session_id: uuid::Uuid,
    context_tokens: usize,
    keep_turns: usize,
    threshold_pct: usize,
    idle_consolidation_secs: u64,
    idle_min_context_pct: usize,
    base_prompt: String,
    tool_section: String,
) {
    let mut cancel_rx = events.cancel_tx.subscribe();
    let mut last_turn_at = Instant::now();

    loop {
        let triggered_by_idle = loop {
            let idle_wait = if idle_consolidation_secs > 0 {
                let elapsed = last_turn_at.elapsed().as_secs();
                let remaining = idle_consolidation_secs.saturating_sub(elapsed);
                Duration::from_secs(remaining.clamp(1, 60))
            } else {
                Duration::from_secs(3600)
            };

            tokio::select! {
                _ = events.llm_post_finished.notified() => {
                    last_turn_at = Instant::now();
                    break false;
                }
                _ = tokio::time::sleep(idle_wait) => {
                    let elapsed = last_turn_at.elapsed().as_secs();
                    if idle_consolidation_secs > 0
                        && elapsed >= idle_consolidation_secs
                        && !shared.llm_busy.load(Ordering::SeqCst)
                    {
                        break true;
                    }
                }
                _ = cancel_rx.recv() => {}
            }
        };

        let (needs, approx_tokens, current_pct, msg_count, effective_threshold) = {
            let s = llm_session.lock().unwrap();
            let approx = s.approx_tokens();
            let pct = if context_tokens > 0 { approx * 100 / context_tokens } else { 0 };
            let effective = if triggered_by_idle { idle_min_context_pct } else { threshold_pct };
            let needs = s.needs_consolidation(context_tokens, effective);
            (needs, approx, pct, s.messages.len(), effective)
        };
        info!(
            target: "memory",
            "Context check ({}): ~{} tokens / {} max ({}%) — threshold {}% — {} msgs — consolidation {}",
            if triggered_by_idle { "idle" } else { "post-turn" },
            approx_tokens, context_tokens, current_pct, effective_threshold,
            msg_count,
            if needs { "TRIGGERED" } else { "not needed" },
        );
        if !needs {
            while cancel_rx.try_recv().is_ok() {}
            if triggered_by_idle {
                last_turn_at = Instant::now();
            }
            continue;
        }

        if !triggered_by_idle {
            info!(target: "memory", "Context limit approaching — starting announced consolidation");

            while shared.llm_busy.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            shared.consolidation_active.store(false, Ordering::SeqCst);
            *shared.transliterated_text.lock().unwrap() =
                "[Sistema: necesitas reorganizar tu memoria para seguir conversando. \
                 Avisa al usuario de que vuelves en unos minutos.]"
                    .to_string();
            events.vad_finish.notify_one();

            loop {
                tokio::select! {
                    _ = events.llm_post_finished.notified() => { break; }
                    _ = cancel_rx.recv() => {}
                }
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
            shared.consolidation_active.store(true, Ordering::SeqCst);
            info!(target: "memory", "Pipeline paused — running consolidation...");
        } else {
            info!(target: "memory", "Idle timer — running silent consolidation...");
        }

        run_consolidation_cycle(
            &background_client, &db, session_id, &llm_session,
            keep_turns, &base_prompt, &tool_section,
        )
        .await;

        if !triggered_by_idle {
            shared.consolidation_active.store(false, Ordering::SeqCst);
            let now = chrono::Local::now().format("%H:%M").to_string();
            *shared.transliterated_text.lock().unwrap() = format!(
                "[Sistema: has terminado de reorganizar tu memoria. Son las {now}. \
                 Avisa al usuario de que ya estás disponible de nuevo.]"
            );
            events.vad_finish.notify_one();
            info!(target: "memory", "Consolidation cycle finished — pipeline resumed");
        }

        last_turn_at = Instant::now();
        while cancel_rx.try_recv().is_ok() {}
    }
}
