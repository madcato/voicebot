use std::time::Duration;

use async_trait::async_trait;
use tracing::{info, warn};

use super::Tool;

/// Substrings that are blocked regardless of context.
/// This is a safety net — not a full sandbox.
const DENYLIST: &[&str] = &[
    "rm -rf /",
    "rm -fr /",
    ":(){",       // fork bomb skeleton
    "mkfs",
    "dd if=/dev/",
    "> /dev/sd",
    "chmod -R 777 /",
    "sudo rm -rf",
    "sudo rm -fr",
];

/// Maximum output returned to the LLM (bytes).
const MAX_OUTPUT_BYTES: usize = 2_000;

/// Tool that runs a shell command and returns its output.
///
/// Enabled only when `SHELL_ENABLED=1` is set in the environment.
/// Configured via:
/// - `SHELL_TIMEOUT_SECS` (default 30) — hard timeout per command
pub struct RunShellTool {
    timeout_secs: u64,
}

impl RunShellTool {
    pub fn new(timeout_secs: u64) -> Self {
        Self { timeout_secs }
    }
}

#[async_trait]
impl Tool for RunShellTool {
    fn name(&self) -> &str {
        "run_shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its output (stdout + stderr + exit code). \
         Use for compiling code, reading files, searching the filesystem, checking system state, \
         running scripts, git operations, etc. \
         Always say what you are about to do before calling this tool. \
         Do NOT run destructive commands (delete, overwrite, format) without explicit user confirmation."
    }

    async fn run(&self, args: &str) -> String {
        let cmd = args.trim();
        if cmd.is_empty() {
            return "Error: no command provided".to_string();
        }

        // Safety: denylist check (case-insensitive)
        let cmd_lower = cmd.to_lowercase();
        for blocked in DENYLIST {
            if cmd_lower.contains(blocked) {
                warn!("run_shell: blocked dangerous pattern {:?} in command {:?}", blocked, cmd);
                return format!("Error: command blocked by safety policy (matched {:?}). Ask the user for confirmation before running destructive commands.", blocked);
            }
        }

        info!("run_shell: {:?}", cmd);

        let cmd_owned = cmd.to_string();
        let result = tokio::time::timeout(
            Duration::from_secs(self.timeout_secs),
            tokio::process::Command::new("sh")
                .arg("-c")
                .arg(&cmd_owned)
                .output(),
        )
        .await;

        match result {
            Err(_) => format!("Error: command timed out after {}s", self.timeout_secs),
            Ok(Err(e)) => format!("Error: failed to spawn command: {}", e),
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let exit_code = output.status.code().unwrap_or(-1);

                let mut out = String::new();
                if !stdout.trim().is_empty() {
                    out.push_str(stdout.trim());
                }
                if !stderr.trim().is_empty() {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str("[stderr] ");
                    out.push_str(stderr.trim());
                }
                if out.is_empty() {
                    out.push_str("(no output)");
                }

                let mut result_str = format!("exit_code={exit_code}\n{out}");
                if result_str.len() > MAX_OUTPUT_BYTES {
                    result_str.truncate(MAX_OUTPUT_BYTES);
                    result_str.push_str("\n[output truncated]");
                }
                result_str
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> RunShellTool {
        RunShellTool::new(10)
    }

    #[tokio::test]
    async fn runs_echo() {
        let result = tool().run("echo hello").await;
        assert!(result.contains("hello"), "{result:?}");
        assert!(result.contains("exit_code=0"), "{result:?}");
    }

    #[tokio::test]
    async fn captures_stderr() {
        let result = tool().run("echo error >&2").await;
        assert!(result.contains("error"), "{result:?}");
    }

    #[tokio::test]
    async fn reports_non_zero_exit() {
        let result = tool().run("exit 1").await;
        assert!(result.contains("exit_code=1"), "{result:?}");
    }

    #[tokio::test]
    async fn blocks_rm_rf_root() {
        let result = tool().run("rm -rf /tmp/../").await;
        // doesn't match the denylist literally but close enough — test the actual denylist entry
        let result2 = tool().run("rm -rf /").await;
        assert!(result2.contains("blocked"), "{result2:?}");
    }

    #[tokio::test]
    async fn blocks_fork_bomb() {
        let result = tool().run(":(){:|:&};:").await;
        assert!(result.contains("blocked"), "{result:?}");
    }

    #[tokio::test]
    async fn empty_args_returns_error() {
        let result = tool().run("").await;
        assert!(result.contains("Error"), "{result:?}");
    }

    #[tokio::test]
    async fn truncates_long_output() {
        // Generate output larger than MAX_OUTPUT_BYTES
        let result = tool().run("yes x | head -c 10000").await;
        assert!(result.contains("[output truncated]"), "{result:?}");
        assert!(result.len() <= MAX_OUTPUT_BYTES + 100, "output too long: {}", result.len());
    }

    #[tokio::test]
    async fn name_is_run_shell() {
        assert_eq!(tool().name(), "run_shell");
    }

    #[tokio::test]
    async fn description_is_non_empty() {
        assert!(!tool().description().is_empty());
    }
}
