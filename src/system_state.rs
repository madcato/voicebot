/// Builds a compact `[SYSTEM STATE]` string with ambient context for the LLM.
///
/// Injected as a prefix to the current user message before each LLM call.
/// Enabled via `INJECT_SYSTEM_DATA=true`. Never stored in the session or DB.
pub async fn build() -> String {
    let time = chrono::Local::now().format("%Y-%m-%d %H:%M").to_string();

    let (app, battery) = tokio::join!(active_app(), battery_level());

    let mut parts = vec![time];
    if let Some(a) = app {
        parts.push(format!("App: {a}"));
    }
    if let Some(b) = battery {
        parts.push(format!("Battery: {b}"));
    }

    format!("[SYSTEM STATE] {}", parts.join(" | "))
}

/// Returns the name of the frontmost application (macOS only).
async fn active_app() -> Option<String> {
    let out = tokio::process::Command::new("osascript")
        .args([
            "-e",
            "tell application \"System Events\" \
             to name of first application process whose frontmost is true",
        ])
        .output()
        .await
        .ok()?;

    if out.status.success() {
        let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if name.is_empty() { None } else { Some(name) }
    } else {
        None
    }
}

/// Returns battery percentage + charging status (macOS only).
async fn battery_level() -> Option<String> {
    let out = tokio::process::Command::new("pmset")
        .args(["-g", "batt"])
        .output()
        .await
        .ok()?;

    if out.status.success() {
        parse_battery(&String::from_utf8_lossy(&out.stdout))
    } else {
        None
    }
}

/// Parses `pmset -g batt` output for a percentage + status string.
/// Example line: `\t-InternalBattery-0 (id=...)	73%; charging; 1:23 remaining`
fn parse_battery(text: &str) -> Option<String> {
    let line = text.lines().find(|l| l.contains('%'))?;
    let pct_end = line.find('%')?;
    let pct_start = line[..pct_end].rfind(|c: char| !c.is_ascii_digit())?;
    let pct = &line[pct_start + 1..=pct_end];

    let status = if line.contains("charging") && !line.contains("discharging") {
        " charging"
    } else {
        ""
    };

    Some(format!("{pct}{status}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_battery_charging() {
        let text = "Now drawing from 'AC Power'\n\
                    \t-InternalBattery-0 (id=12345678)\t73%; charging; 1:23 remaining\n";
        assert_eq!(parse_battery(text), Some("73% charging".to_string()));
    }

    #[test]
    fn parse_battery_discharging() {
        let text = "Now drawing from 'Battery Power'\n\
                    \t-InternalBattery-0 (id=12345678)\t45%; discharging; 2:10 remaining\n";
        assert_eq!(parse_battery(text), Some("45%".to_string()));
    }

    #[test]
    fn parse_battery_full() {
        let text = "Now drawing from 'AC Power'\n\
                    \t-InternalBattery-0 (id=12345678)\t100%; charged; 0:00 remaining\n";
        // "charged" contains neither "charging" standalone nor "discharging"
        assert_eq!(parse_battery(text), Some("100%".to_string()));
    }

    #[test]
    fn parse_battery_no_battery_line() {
        assert_eq!(parse_battery("Now drawing from 'AC Power'\n"), None);
    }

    #[tokio::test]
    async fn build_contains_system_state_prefix() {
        let state = build().await;
        assert!(
            state.starts_with("[SYSTEM STATE]"),
            "should start with marker: {state:?}"
        );
    }

    #[tokio::test]
    async fn build_contains_date_and_time() {
        let state = build().await;
        // Date is formatted as YYYY-MM-DD HH:MM
        assert!(
            state.contains('-') && state.contains(':'),
            "should contain date/time: {state:?}"
        );
    }
}
