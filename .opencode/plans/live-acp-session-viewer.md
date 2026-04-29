# Live ACP Session Viewer via SQLite

## Problem

`hermes --resume {sid}` opens a new interactive Hermes session instead of viewing the active ACP session. ACP sessions are process-local, so a second process can't attach to the running session.

## Solution

Replace the `hermes --resume` command with a self-contained bash loop in a new Terminal window. The loop polls `~/.hermes/state.db` every 2 seconds for the current session's messages, color-coding user/assistant/tool messages. Messages are written to SQLite in real-time by the AIAgent loop during ACP execution, so this approach provides a near-live view.

## Files to Modify

- `src/tools/run_agent.rs` — `open_session_in_terminal` method (lines 763-782)

## Implementation Details

### New `open_session_in_terminal` logic

1. **Resolve Hermes DB path**: Read `HERMES_HOME` env var (default `~/.hermes`), construct `state.db` path.

2. **Build a bash polling script**: Uses `clear`, `sqlite3` to query messages, and ANSI color escapes for roles. Pipe-delimited output parsed with `read` in a while loop.

3. **Launch via osascript**: Same mechanism as current code (`tell application "Terminal" to do script`).

### Key script content

```bash
clear
sqlite3 ~/.hermes/state.db \
  "SELECT role || '|' || COALESCE(content, '[tool_call]') \
   FROM messages \
   WHERE session_id = '{sid}' \
   ORDER BY timestamp, rowid" | \
while IFS='|' read -r ROLE CONTENT; do
  case "$ROLE" in
    user)      echo -e "[$(tput setaf 1)USER$(tput sgr0)] $CONTENT" ;;
    assistant) echo -e "[$(tput setaf 2)JAVAS$(tput sgr0)] $CONTENT" ;;
    tool)      echo -e "[$(tput setaf 3)TOOL$(tput sgr0)] $CONTENT" ;;
    *)         echo -e "[$(tput setaf 6)$ROLE$(tput sgr0)] $CONTENT" ;;
  esac
  echo "---"
done
sleep 2
```

### Rust code structure

```rust
pub async fn open_session_in_terminal(&self) {
    let sid = match &self.session_id {
        Some(s) => s,
        None => { warn!(...); return; }
    };

    // Locate Hermes state.db
    let hermes_home = std::env::var("HERMES_HOME")
        .ok()
        .unwrap_or_else(|| format!("{}/.hermes", std::env::var("HOME").unwrap_or_default()));
    let db_path = format!("{}/state.db", hermes_home);

    if !std::path::Path::new(&db_path).exists() {
        warn!(target: "agent", "Hermes state.db not found at {}", db_path);
        return;
    }

    // Build escaped bash script for osascript
    // Use \x1b for Terminal escape sequence prefix
    // Use sqlite3 pipe to format messages with colors
    // Run in an infinite loop with sleep 2

    // Spawn via osascript -e "tell application \"Terminal\" to do script ..."
}
```

### Osascript escaping rules

The `do script` command requires proper escaping:
- Use `\x1b` for the escape character (Terminal interprets this)
- Double quotes inside the script must be escaped as `\"`
- Backslashes must be double-escaped for osascript
- The script is wrapped in `while true; do ... sleep 2; done`

### Edge cases
- **Missing `state.db`**: Log warning, skip
- **Session not yet in DB**: Brief empty output, resolves on first message save
- **Missing `sqlite3`**: Check with `sqlite3 --version`, skip if unavailable
- **Multiple HERMES profiles**: Uses the same `HERMES_HOME` as the ACP subprocess

## Testing

After implementation:
1. Run voicebot with ACP mode enabled
2. Trigger `run_agent` with a task
3. Verify a Terminal window opens showing live session messages
4. Verify messages appear in real-time as the agent works
5. Verify color coding works (user=red, assistant=green, tool=yellow)
