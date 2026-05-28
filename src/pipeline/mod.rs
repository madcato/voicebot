pub mod consolidation;
pub mod frames;
pub mod fsm;
pub mod llm_task;
pub mod sen_task;
pub mod state;
pub mod tts_task;

pub use consolidation::{build_system_prompt, consolidation_task, run_consolidation_cycle};
pub use frames::PipelineFrame;
pub use fsm::{PauseReason, PipelineState};
pub use llm_task::llm_task;
pub use sen_task::sen_task;
pub use state::PipelineEvents;
pub use tts_task::tts_task;
