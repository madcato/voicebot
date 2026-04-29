# Jarvis Voicebot — Architecture Analysis & Redesign

## 1. Current Data Flows — Diagnosis

The voicebot has **6 concurrent components** communicating through a mix of mechanisms:

### Existing flows (mapped from code)

```
Thread 1: AUDIO CAPTURE (CPAL thread)
  |
  +--> async_channel::bounded<AudioChunk> (capacity 200)
       |
Thread 2: MAIN LOOP (tokio select)
  |  +--> sttvad.process_audio() --> mpsc<SpeechEvent>
  |  |
  |  +--> SpeechEvent::SpeechStart:
  |  |     - events.cancel_tx.send(())        [broadcast, interrupts EVERYTHING]
  |  |     - play_cancel.store(true)          [AtomicBool, interrupts playback]
  |  |     - speech_buffer.clear()
  |  |     - utterance_epoch.fetch_add()      [AtomicU64]
  |  |
  |  +--> SpeechEvent::SpeechEnd:
  |        - shared.transliterated_text = text  [Mutex<String>, atomic data]
  |        - shared.t_vad_end = Instant         [Mutex<Instant>, timestamp]
  |        - events.vad_finish.notify_one()     [Notify, single-shot signal]
  |
  +--> ProactiveEvent (mpsc channel):
        - AgentResult, AgentQuestion, InferenceDaemon
        - Injected as transliterated_text + vad_finish.notify()
  |
Thread 3: LLM TASK (llm_task)
  |  Blocks on: events.vad_finish.notified()
  |  Reads:   shared.transliterated_text      [Mutex<String>]
  |  Reads:   shared.pending_tool_response    [AtomicBool]
  |  Writes:  shared.assistant_text.push_str  [Mutex<String>, continuous stream]
  |  Writes:  events.llm_post_received.notify [Notify, signal]
  |  Writes:  events.llm_post_finished.notify [Notify, signal]
  |  Listens: cancel_rx.recv()                [broadcast, interruption]
  |
Thread 4: SENTENCE TASK (sen_task)
  |  Blocks on: events.llm_post_received.notified()
  |  Reads:   shared.assistant_text           [Mutex<String>]
  |  Reads:   shared.llm_post_finished        [AtomicBool]
  |  Writes:  shared.sentences.push_back()    [Mutex<VecDeque>, atomic data]
  |  Writes:  events.sentence_ready.notify()  [Notify, signal]
  |  Listens: cancel_rx.recv()                [broadcast, interruption]
  |
Thread 5: TTS TASK (tts_task)
  |  Blocks on: events.sentence_ready.notified()
  |  Reads:   shared.sentences.pop_front()    [Mutex<VecDeque>]
  |  Reads:   tts_muted                       [AtomicBool]
  |  Writes:  latency metrics                 [shared.t_vad_end]
  |  Listens: cancel_rx.recv()                [broadcast, interruption]
  |
Thread 6: CONSOLIDATION TASK (consolidation_task)
  |  Blocks on: events.llm_post_finished.notified()
  |  Reads:   shared.llm_busy                 [AtomicBool, polling]
  |  Writes:  shared.consolidation_active     [AtomicBool]
  |  Writes:  shared.transliterated_text      [reuses the same channel!]
  |  Writes:  events.vad_finish.notify()      [reuses the same trigger!]
  |
Other threads: InferenceDaemon, EyesDaemon, MCP, ACP Agent, WebSocket server
```

### Concrete problems

1. **SharedSession is a battlefield**: 16 fields (`Mutex<String>`, `AtomicBool`,
   `VecDeque`) shared across ALL threads. Every access is a potential race condition,
   deadlock, or data corruption.

2. **Signal reuse without clear semantics**: `transliterated_text` +
   `vad_finish.notify()` is used for 5 different things: user text, system
   notifications, consolidation results, ACP questions, and the initial greeting.
   One communication channel with 5 distinct semantics.

3. **No defined state machine**: The system has implicit states (Idle, Listening,
   Thinking, Speaking) but no code structure declares them. The `AtomicBool` fields
   (`llm_busy`, `consolidation_active`, `stt_result_pending`) are states that should
   be transitions of an explicit FSM.

4. **Cancellation vs interruption mixed**: `cancel_tx` broadcast is used both for
   "the user spoke while you were talking" (barge-in, immediate) and "consolidation
   needs to pause" (planned). There is no priority between interruption signals.

5. **No backpressure**: The audio capture buffer has capacity 200. If VAD/STT is
   delayed (Whisper can take 1–2s), chunks accumulate and memory grows.

---

## 2. Relevant Design Patterns

### A. Hierarchical State Machines (Statecharts) — David Harel, 1987

An FSM where states can contain sub-states. The system state defines which
transitions are valid.

Why it matters: the voicebot has obvious states:

```
IDLE
  --> (vad detected)      --> LISTENING
    --> (vad silence)       --> STT_PROCESSING
      --> (STT ready)         --> THINKING
        --> (first sentence)    --> SPEAKING
          --> (user interrupts)   --> LISTENING   [barge-in]
          --> (finished)          --> IDLE
```

Sub-states also exist: SPEAKING can have TTS_SYNTHESIZING and PLAYING.
THINKING can have STREAMING and TOOL_EXECUTING.

**Key benefit**: replaces 16 `AtomicBool` flags with a single enum. The state is
an enum variant, not 16 independent bits.

### B. Actor Model — Carl Hewitt, 1973

Each component is an "actor" with private state modified only by processing messages.
Actors communicate via async messages and never share state.

Current problem: `SharedSession` is the opposite of the actor model — it is shared
mutable state.

Actor mapping for this project:

```
Actor VADProcessor     -- owns: speech buffer, VAD state machine
Actor SttProcessor     -- owns: Whisper context, partial transcript
Actor LlmProcessor     -- owns: session history, tool registry
Actor SentenceSplitter -- owns: splitter buffer
Actor TtsProcessor     -- owns: audio queue, playback state
```

Each actor is a `tokio::task` with an `mpsc::Receiver<Message>` as its main loop.
Tokio tasks + mpsc channels are already lightweight actors — no external crate needed.

### C. Pipecat Framework — Frame-based Pipeline

Everything is modelled as "frames" flowing through a pipeline. Each frame is a
message (audio, text, control, event). The key insight relevant here:

- **Data frames**: audio chunks, text, tokens — flow forward
- **Control frames**: start, stop, cancel, interrupt — can flow in any direction

The current system mixes data and control in the same channels
(`transliterated_text` carries both user data and system notifications).

### D. Priority Bands (inspired by ARINC 653)

| Priority | Component                | Deadline     | Type             |
|----------|--------------------------|--------------|------------------|
| CRITICAL | VAD detection + barge-in | < 50 ms      | Event-triggered  |
| HIGH     | STT transcription        | 200–500 ms   | Event-triggered  |
| MEDIUM   | LLM inference + TTS      | 500–1000 ms  | Event-triggered  |
| LOW      | Consolidation, Memory    | No deadline  | Time-triggered   |
| MAINT    | DB writes, logging       | No deadline  | Time-triggered   |

High-priority tasks must be able to interrupt low-priority ones, never the reverse.

### E. Tokio Channel Selection

| Channel       | Pattern                      | Use case here                       |
|---------------|------------------------------|-------------------------------------|
| `mpsc`        | 1 producer → 1 consumer      | Audio chunks, SpeechEvents, frames  |
| `broadcast`   | 1 producer → N consumers     | Barge-in (all actors react)         |
| `watch`       | 1 writer → N readers (latest)| Pipeline state (IDLE/SPEAKING/…)    |
| `oneshot`     | 1 message, 1 time            | ACP permission responses            |
| `async-channel`| MPMC bounded                | Audio capture buffer (current)      |

---

## 3. Recommended Architecture

**Hybrid Actor + typed channels + watch-based FSM**

The key insight from the original architecture investigation: the current pipeline
task structure is correct. Actors already communicate in the right direction
(VAD→STT→LLM→SEN→TTS). The problem is **how** they communicate — shared mutable
state instead of typed messages.

### Two planes of communication

**Control plane** (replaces `SharedSession` flags + signal reuse):
```
watch::Sender<PipelineState>   -- one writer, all actors can read current state
broadcast barge_in_tx          -- VAD only; all actors cancel immediately
broadcast pause_tx             -- consolidation only; LLM pauses, others ignore
```

**Data plane** (replaces `SharedSession` data fields):
```
VAD actor      --mpsc<TranscriptReady>-->  LLM actor
LLM actor      --mpsc<LLMToken>-------->  Sentence actor
Sentence actor --mpsc<SentenceReady>--->  TTS actor
```

### Pipeline state machine (without a controller on the hot path)

```rust
// src/pipeline/fsm.rs

#[derive(Clone, Debug, PartialEq)]
pub enum PipelineState {
    Idle,
    Listening { utterance_id: u64 },
    Thinking  { utterance_id: u64 },
    Speaking  { utterance_id: u64 },
    Paused    { reason: PauseReason },
}

#[derive(Clone, Debug, PartialEq)]
pub enum PauseReason { Consolidation }
```

State is held in a `watch::Sender<PipelineState>`. **Each actor that owns a
transition writes it directly** — the VAD actor transitions Idle→Listening, the
LLM actor transitions Listening→Thinking, etc. No central coordinator sits in the
path between them.

This is the critical difference from the original proposal: a `PipelineSupervisor`
that receives all events and re-emits commands would be a single-threaded bottleneck
processing hundreds of `LLMToken` frames per second. The `watch` channel gives every
actor read access to global state without any coordinator.

### Supervisor as observer only (off the hot path)

A supervisor task can still exist for monitoring, logging, and diagnostics — it
subscribes to the `watch` channel and the `ContextLens` bus. It never sits between
actors:

```rust
// observer — reads state changes, logs transitions, feeds TUI/dashboard
async fn supervisor_observer(mut state_rx: watch::Receiver<PipelineState>) {
    loop {
        state_rx.changed().await.ok();
        let state = state_rx.borrow().clone();
        tracing::info!(target: "fsm", "State → {:?}", state);
        // update TUI, metrics, etc.
    }
}
```

### Actor diagram

```
                   watch<PipelineState>  (readable by all, written by owner)
                   broadcast barge_in_tx (VAD → all)
                   broadcast pause_tx    (consolidation → LLM)

[CPAL thread] --async_channel--> [VAD Actor]
                                      |
                             mpsc<TranscriptReady>
                                      |
                                 [LLM Actor] <--mpsc<ProactiveEvent>-- [agents/tools]
                                      |
                              mpsc<LLMToken>
                                      |
                            [Sentence Actor]
                                      |
                           mpsc<SentenceReady>
                                      |
                              [TTS Actor] --> speaker

                   [Supervisor Observer] -- reads watch + ContextLens bus (no hot path)
                   [Analysis Ring]       -- reads ContextLens bus (identity, emotion, video)
```

### Frame/Message enum

```rust
// src/pipeline/frames.rs

pub enum PipelineFrame {
    // STT output
    TranscriptReady { utterance_id: u64, text: String },

    // LLM output
    LLMToken         { utterance_id: u64, token: String },
    LLMToolCall      { name: String, args: String },
    LLMResponseDone  { utterance_id: u64, full_text: String },

    // Sentence splitter output
    SentenceReady    { utterance_id: u64, sentence: String },

    // TTS output
    PlaybackDone     { utterance_id: u64 },

    // Proactive / system input to LLM
    SystemNotification { text: String },
    AgentResult        { task: String, result: String, tool_call_id: Option<String> },
    TextInput          { text: String },
}
```

`AudioChunk` is not in this enum — audio flows through its own `async_channel` directly
to the VAD actor, bypassing the frame routing entirely (it would be too high-volume).

### SharedSession field migration

| Current field           | Replacement                                        |
|-------------------------|----------------------------------------------------|
| `transliterated_text`   | `mpsc<PipelineFrame::TranscriptReady>`             |
| `assistant_text`        | `mpsc<PipelineFrame::LLMToken>` (streamed)         |
| `sentences`             | `mpsc<PipelineFrame::SentenceReady>`               |
| `llm_post_finished`     | `mpsc<PipelineFrame::LLMResponseDone>`             |
| `llm_post_received`     | implicit in `LLMToken` flow                        |
| `vad_finish`            | implicit in `TranscriptReady` flow                 |
| `sentence_ready`        | implicit in `SentenceReady` flow                   |
| `cancel_tx`             | `broadcast barge_in_tx` (barge-in only)            |
| `llm_busy`              | `PipelineState::Thinking`                          |
| `consolidation_active`  | `PipelineState::Paused { Consolidation }`          |
| `stt_result_pending`    | removed (implicit in pipeline flow)                |
| `pending_tool_response` | `mpsc<PipelineFrame::AgentResult>`                 |
| `pending_system_injection` | `mpsc<PipelineFrame::SystemNotification>`       |
| `text_input_pending`    | `mpsc<PipelineFrame::TextInput>`                   |
| `t_vad_end`, `t_llm_post_send`, `first_speech_played` | latency fields on `PipelineState` variants |

---

## 4. Key Decision: Rewrite vs Refactor

The current code has the right high-level structure — tasks separated, channels used
for communication. The problem is the communication mechanism, not the structure.
An incremental refactor is the right call.

### High-impact refactor (3–5 days) — recommended

1. Define `PipelineFrame` enum in `src/pipeline/frames.rs`
2. Add `PipelineState` enum + `watch::Sender` in `src/pipeline/fsm.rs`
3. Replace `SharedSession` fields one by one with typed channels
4. Split `cancel_tx` into `barge_in_tx` and `pause_tx`

No behavioral changes. Each step compiles and passes tests before the next begins.

### Full rewrite (2–3 weeks) — not recommended

Full actor framework with supervision trees, SCXML-defined HSM, priority scheduling.
Appropriate for a product with multiple engineers and strict latency SLAs.
Overhead not justified for a single-user bot on a single machine.

---

## 5. Step-by-Step Migration Plan

### Step 1 — Define PipelineFrame (day 1)

Create `src/pipeline/frames.rs`. Add `utterance_id: u64` to every event variant so
the full lifetime of an utterance (VAD → STT → LLM → TTS → done) can be correlated
in logs and metrics.

### Step 2 — Add PipelineState + watch channel (day 1–2)

Create `src/pipeline/fsm.rs`. Wire a `watch::Sender<PipelineState>` through main.rs.
Replace all `AtomicBool` state flags (`llm_busy`, `consolidation_active`,
`stt_result_pending`) with reads/writes on the watch channel. Each actor that owns
a transition calls `state_tx.send(PipelineState::Thinking { utterance_id })` directly.

Add the supervisor observer as a `tokio::spawn` that logs state changes.

### Step 3 — Migrate SharedSession fields to typed channels (days 2–4)

Field by field, in this order (least risky first):

1. `pending_system_injection` + `pending_tool_response` → `mpsc<PipelineFrame>`
   (these already have clear single-writer semantics)
2. `sentences` + `sentence_ready` → `mpsc<SentenceReady>` between sen_task and tts_task
3. `assistant_text` + `llm_post_received` → `mpsc<LLMToken>` between llm_task and sen_task
4. `transliterated_text` + `vad_finish` → `mpsc<TranscriptReady>` from audio loop to llm_task
5. Latency timestamps → fields on `PipelineState` variants (carried with state transitions)

After each field migration: `cargo build && cargo test`.

### Step 4 — Split cancellation signals (day 4–5)

Replace the single `cancel_tx broadcast` with two dedicated broadcasts:

```rust
let (barge_in_tx, _) = broadcast::channel::<u64>(4); // payload = utterance_id
let (pause_tx, _)    = broadcast::channel::<()>(2);
```

- `barge_in_tx`: only the VAD actor sends this on `SpeechStart`. All pipeline actors
  cancel their current work immediately.
- `pause_tx`: only `consolidation_task` sends this. Only `llm_task` listens.

This makes barge-in and consolidation pause independent — they can no longer
interfere with each other.

### Step 5 — Remove SharedSession

At this point `SharedSession` should be empty (or contain only fields that have
no typed-channel equivalent yet). Delete the struct. Each actor holds its own
state privately.

### Step 6 — Testing and validation (ongoing)

- Unit test each actor in isolation by injecting frames directly into its receiver
- Integration test: inject a `TranscriptReady` frame, assert `PlaybackDone` fires
- Latency benchmark: confirm end-to-end VAD→first-audio latency did not increase
- Barge-in stress test: fire `barge_in_tx` while TTS is mid-sentence, assert clean stop

---

## 6. Relationship to the Analysis Ring (ContextLens)

The Analysis Ring (`src/analysis/`) already implemented in this codebase is
compatible with this architecture and does not need to change:

- `ContextLens` continues to receive writes from `IdentityAnalyzer` and future analyzers
- The `ContextLens` broadcast bus (planned Step 5 in the analysis plan) receives
  `"stt_transcript"` and `"llm_response"` entries — exactly the `TranscriptReady`
  and `LLMResponseDone` frame payloads
- The supervisor observer can subscribe to both the `watch<PipelineState>` and the
  `ContextLens` bus for a complete view of system activity

---

## 7. References

- David Harel, "Statecharts: A Visual Formalism for Complex Systems" (1987)
  https://www.inf.ed.ac.uk/teaching/courses/seoc/2002_2003/statecharts.html

- Carl Hewitt, "Actor Models of Computation" (1973)
  https://en.wikipedia.org/wiki/Actor_model

- Pipecat AI — Frame Pipeline Architecture
  https://docs.pipecat.ai/pipecat/learn/pipeline

- Tokio Channels Guide
  https://tokio.rs/tokio/tutorial/channels

- Tokio `sync::watch` docs
  https://docs.rs/tokio/latest/tokio/sync/watch/index.html

- ARINC 653 — Time and Space Partitioning
  https://en.wikipedia.org/wiki/ARINC_653

- Barr Group — Introduction to Hierarchical State Machines
  https://barrgroup.com/blog/introduction-hierarchical-state-machines

---

## 8. Summary

The current project is not a disaster. Task separation, channel-based communication,
the ACP Body/Brain architecture — all correct. The specific problem is `SharedSession`
as a shared mutable battlefield and the abuse of `transliterated_text`/`vad_finish`
as a multi-purpose channel.

**What to do:**
1. Give every data flow its own typed channel (`PipelineFrame` variants)
2. Replace 16 `AtomicBool` flags with one `watch<PipelineState>` enum
3. Split cancellation into `barge_in_tx` (VAD → all) and `pause_tx` (consolidation → LLM)
4. Keep actors communicating directly — no coordinator on the hot path
5. Add a supervisor *observer* that reads `watch` + `ContextLens` bus for monitoring

Once each data flow has its own typed channel, the system will be significantly
easier to reason about, debug, and extend — without rewriting from scratch.
