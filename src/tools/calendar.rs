use async_trait::async_trait;
use chrono::{Datelike, Duration, Local, NaiveDate, NaiveTime, Timelike};
use tokio::process::Command;

use super::Tool;

// ── Date helpers ──────────────────────────────────────────────────────────────

/// Parses a date string into a `NaiveDate`.
///
/// Accepts:
/// - `"today"` / `"hoy"`
/// - `"tomorrow"` / `"mañana"`
/// - ISO `"2026-03-10"`
/// - European `"10/03/2026"` (DD/MM/YYYY)
fn parse_date(input: &str) -> Option<NaiveDate> {
    let s = input.trim().to_lowercase();
    let today = Local::now().date_naive();
    match s.as_str() {
        "today" | "hoy" => return Some(today),
        "tomorrow" | "mañana" => return Some(today + Duration::days(1)),
        _ => {}
    }
    NaiveDate::parse_from_str(input.trim(), "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(input.trim(), "%d/%m/%Y"))
        .ok()
}

/// Parses "HH:MM" into (hours, minutes). Returns None if unparseable.
fn parse_time(input: &str) -> Option<NaiveTime> {
    NaiveTime::parse_from_str(input.trim(), "%H:%M").ok()
}

/// Escapes double-quotes in a string for safe embedding in an AppleScript
/// string literal (surrounded by `"…"`).
fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ── AppleScript builders ──────────────────────────────────────────────────────

/// Builds the `set date …` AppleScript fragment that constructs a date from
/// components without relying on locale-sensitive date parsing.
///
/// `var` is the AppleScript variable name (e.g. `"d"`).
/// `seconds_since_midnight` is the time-of-day in seconds (0 for midnight).
fn as_date_fragment(var: &str, date: NaiveDate, seconds_since_midnight: i64) -> String {
    format!(
        "set {var} to current date\n\
         set year of {var} to {y}\n\
         set month of {var} to {m}\n\
         set day of {var} to {d}\n\
         set time of {var} to {t}",
        var = var,
        y = date.year(),
        m = date.month(),
        d = date.day(),
        t = seconds_since_midnight,
    )
}

/// Builds the AppleScript that queries Calendar.app for events on a given day.
fn build_get_events_script(date: NaiveDate) -> String {
    // We query events whose start date falls within [00:00, 00:00 + 1 day).
    format!(
        r#"tell application "Calendar"
    {start_frag}
    set endD to startD + (1 * days)
    set lf to ASCII character 10
    set output to ""
    repeat with c in every calendar
        try
            set es to (every event of c whose start date >= startD and start date < endD)
            repeat with e in es
                set eStart to start date of e
                set eEnd to end date of e
                set hh1 to text -2 thru -1 of ("0" & ((hours of eStart) as integer) as string)
                set mm1 to text -2 thru -1 of ("0" & ((minutes of eStart) as integer) as string)
                set hh2 to text -2 thru -1 of ("0" & ((hours of eEnd) as integer) as string)
                set mm2 to text -2 thru -1 of ("0" & ((minutes of eEnd) as integer) as string)
                set output to output & (name of c) & "|" & (summary of e) & "|" & hh1 & ":" & mm1 & "|" & hh2 & ":" & mm2 & lf
            end repeat
        end try
    end repeat
    return output
end tell"#,
        start_frag = as_date_fragment("startD", date, 0),
    )
}

/// Builds the AppleScript that creates a Calendar.app event.
fn build_create_event_script(
    title: &str,
    date: NaiveDate,
    start_secs: i64,
    end_secs: i64,
    calendar_name: Option<&str>,
    notes: Option<&str>,
) -> String {
    let cal = esc(calendar_name.unwrap_or(""));
    let title_esc = esc(title);
    let notes_prop = notes
        .map(|n| format!(", description:\"{}\"", esc(n)))
        .unwrap_or_default();

    // If no calendar name is given, pick the first available calendar.
    let cal_block = if cal.is_empty() {
        "set targetCal to item 1 of every calendar".to_string()
    } else {
        format!("set targetCal to (first calendar whose name is \"{cal}\")")
    };

    format!(
        r#"tell application "Calendar"
    {start_frag}
    set endD to startD
    set time of endD to {end_secs}
    {cal_block}
    tell targetCal
        make new event at end with properties {{summary:"{title_esc}", start date:startD, end date:endD{notes_prop}}}
    end tell
    reload calendars
end tell
return "ok""#,
        start_frag = as_date_fragment("startD", date, start_secs),
        end_secs = end_secs,
        cal_block = cal_block,
        title_esc = title_esc,
        notes_prop = notes_prop,
    )
}

/// Builds the AppleScript that creates a Reminders.app reminder.
fn build_create_reminder_script(title: &str, date: NaiveDate, time_secs: Option<i64>) -> String {
    let title_esc = esc(title);
    let due_block = if let Some(secs) = time_secs {
        format!(
            "{frag}\n    set due date of r to dueD",
            frag = as_date_fragment("dueD", date, secs)
        )
    } else {
        // Due at midnight of the given day
        format!(
            "{frag}\n    set due date of r to dueD",
            frag = as_date_fragment("dueD", date, 0)
        )
    };

    format!(
        r#"tell application "Reminders"
    set r to make new reminder at end of default list with properties {{name:"{title_esc}"}}
    {due_block}
end tell
return "ok""#,
        title_esc = title_esc,
        due_block = due_block,
    )
}

/// Runs an AppleScript string via `osascript -e` and returns stdout on success.
async fn run_script(script: &str) -> Result<String, String> {
    let out = Command::new("osascript")
        .arg("-e")
        .arg(script)
        .output()
        .await
        .map_err(|e| format!("osascript launch failed: {e}"))?;

    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        Err(if err.is_empty() {
            format!("osascript exited with status {}", out.status)
        } else {
            err
        })
    }
}

// ── CalendarGetEventsTool ─────────────────────────────────────────────────────

/// Queries Calendar.app for all events on a specific date.
pub struct CalendarGetEventsTool;

#[async_trait]
impl Tool for CalendarGetEventsTool {
    fn name(&self) -> &str {
        "get_calendar_events"
    }

    fn description(&self) -> &str {
        "Returns all calendar events for a specific date from Calendar.app. \
         Use when the user asks about their schedule, agenda, meetings, or \
         what they have planned for a day."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "The date to query. Accepts 'today', 'tomorrow', \
                                    'hoy', 'mañana', or ISO format 'YYYY-MM-DD'."
                }
            },
            "required": ["date"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let date_str = serde_json::from_str::<serde_json::Value>(args)
            .ok()
            .and_then(|v| v["date"].as_str().map(String::from))
            .unwrap_or_else(|| "today".to_string());

        let date = match parse_date(&date_str) {
            Some(d) => d,
            None => return format!("No entiendo la fecha '{date_str}'. Usa 'today', 'mañana' o formato YYYY-MM-DD."),
        };

        let script = build_get_events_script(date);

        match run_script(&script).await {
            Ok(output) if output.is_empty() => {
                format!(
                    "No hay eventos en el calendario para el {}.",
                    date.format("%d/%m/%Y")
                )
            }
            Ok(output) => format_events_output(&output, date),
            Err(e) => format!("Error al consultar Calendar.app: {e}"),
        }
    }
}

/// Formats the raw `name|title|HH:MM|HH:MM\n…` output into a readable string.
fn format_events_output(raw: &str, date: NaiveDate) -> String {
    let mut lines: Vec<String> = raw
        .split(['\n', '\r'])
        .filter(|l| !l.is_empty())
        .map(|line| {
            let parts: Vec<&str> = line.splitn(4, '|').collect();
            match parts.as_slice() {
                [cal, title, start, end] => format!("{start}–{end}  {title}  ({cal})"),
                _ => line.to_string(),
            }
        })
        .collect();

    lines.sort(); // sort by start time (lexicographic on HH:MM works)

    format!(
        "Eventos del {}:\n{}",
        date.format("%d/%m/%Y"),
        lines.join("\n")
    )
}

// ── CalendarCreateTool ────────────────────────────────────────────────────────

/// Creates a Calendar.app event or a Reminders.app reminder.
pub struct CalendarCreateTool;

#[async_trait]
impl Tool for CalendarCreateTool {
    fn name(&self) -> &str {
        "create_calendar_event"
    }

    fn description(&self) -> &str {
        "Creates a new event in Calendar.app or a reminder in Reminders.app. \
         Use when the user asks to add, schedule, or create an appointment, \
         meeting, reminder, or task."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the event or reminder."
                },
                "date": {
                    "type": "string",
                    "description": "Date. Accepts 'today', 'tomorrow', 'hoy', 'mañana', or 'YYYY-MM-DD'."
                },
                "type": {
                    "type": "string",
                    "enum": ["event", "reminder"],
                    "description": "Whether to create a calendar event (default) or a Reminders reminder."
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in 'HH:MM' (24h). Omit for all-day events."
                },
                "end_time": {
                    "type": "string",
                    "description": "End time in 'HH:MM' (24h). Defaults to start_time + 1 hour."
                },
                "calendar": {
                    "type": "string",
                    "description": "Calendar name (e.g. 'Work', 'Personal'). Uses first available if omitted."
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes or description for the event."
                }
            },
            "required": ["title", "date"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let v = match serde_json::from_str::<serde_json::Value>(args) {
            Ok(v) => v,
            Err(_) => return "Argumentos inválidos para create_calendar_event.".to_string(),
        };

        let title = match v["title"].as_str().filter(|s| !s.is_empty()) {
            Some(t) => t.to_string(),
            None => return "Se requiere un título para el evento.".to_string(),
        };

        let date_str = v["date"].as_str().unwrap_or("today");
        let date = match parse_date(date_str) {
            Some(d) => d,
            None => return format!("No entiendo la fecha '{date_str}'."),
        };

        let kind = v["type"].as_str().unwrap_or("event");
        let start_time_str = v["start_time"].as_str();
        let end_time_str = v["end_time"].as_str();
        let calendar = v["calendar"].as_str();
        let notes = v["notes"].as_str();

        if kind == "reminder" {
            let time_secs = start_time_str
                .and_then(|s| parse_time(s))
                .map(|t| t.hour() as i64 * 3600 + t.minute() as i64 * 60);

            let script = build_create_reminder_script(&title, date, time_secs);
            return match run_script(&script).await {
                Ok(_) => format!(
                    "Recordatorio '{}' creado para el {}.",
                    title,
                    date.format("%d/%m/%Y")
                ),
                Err(e) => format!("Error al crear el recordatorio: {e}"),
            };
        }

        // ── Calendar event ────────────────────────────────────────────────────
        let (start_secs, end_secs) = match start_time_str {
            Some(st) => {
                let start = match parse_time(st) {
                    Some(t) => t.hour() as i64 * 3600 + t.minute() as i64 * 60,
                    None => return format!("No entiendo la hora de inicio '{st}'."),
                };
                let end = end_time_str
                    .and_then(|et| parse_time(et))
                    .map(|t| t.hour() as i64 * 3600 + t.minute() as i64 * 60)
                    .unwrap_or(start + 3600); // default: +1 hour
                (start, end)
            }
            // All-day event: midnight to midnight+1
            None => (0i64, 86400i64),
        };

        let script = build_create_event_script(
            &title,
            date,
            start_secs,
            end_secs,
            calendar,
            notes,
        );

        match run_script(&script).await {
            Ok(_) => {
                if let Some(st) = start_time_str {
                    let end_label = end_time_str.unwrap_or("—");
                    format!(
                        "Evento '{}' creado el {} de {} a {}.",
                        title,
                        date.format("%d/%m/%Y"),
                        st,
                        end_label
                    )
                } else {
                    format!(
                        "Evento de día completo '{}' creado para el {}.",
                        title,
                        date.format("%d/%m/%Y")
                    )
                }
            }
            Err(e) => format!("Error al crear el evento: {e}"),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_date ────────────────────────────────────────────────────────────

    #[test]
    fn parse_today_and_hoy() {
        let today = Local::now().date_naive();
        assert_eq!(parse_date("today"), Some(today));
        assert_eq!(parse_date("hoy"), Some(today));
    }

    #[test]
    fn parse_today_case_insensitive() {
        let today = Local::now().date_naive();
        assert_eq!(parse_date("Today"), Some(today));
        assert_eq!(parse_date("HOY"), Some(today));
    }

    #[test]
    fn parse_tomorrow_and_manana() {
        let tomorrow = Local::now().date_naive() + Duration::days(1);
        assert_eq!(parse_date("tomorrow"), Some(tomorrow));
        assert_eq!(parse_date("mañana"), Some(tomorrow));
    }

    #[test]
    fn parse_iso_date() {
        assert_eq!(
            parse_date("2026-03-10"),
            Some(NaiveDate::from_ymd_opt(2026, 3, 10).unwrap())
        );
    }

    #[test]
    fn parse_european_date() {
        assert_eq!(
            parse_date("10/03/2026"),
            Some(NaiveDate::from_ymd_opt(2026, 3, 10).unwrap())
        );
    }

    #[test]
    fn parse_invalid_date_returns_none() {
        assert_eq!(parse_date("not a date"), None);
        assert_eq!(parse_date(""), None);
    }

    // ── parse_time ────────────────────────────────────────────────────────────

    #[test]
    fn parse_time_24h() {
        let t = parse_time("14:30").unwrap();
        assert_eq!(t.hour(), 14);
        assert_eq!(t.minute(), 30);
    }

    #[test]
    fn parse_time_midnight() {
        let t = parse_time("00:00").unwrap();
        assert_eq!(t.hour(), 0);
        assert_eq!(t.minute(), 0);
    }

    #[test]
    fn parse_time_invalid_returns_none() {
        assert_eq!(parse_time("25:00"), None);
        assert_eq!(parse_time("not a time"), None);
    }

    // ── esc ───────────────────────────────────────────────────────────────────

    #[test]
    fn esc_handles_double_quotes() {
        assert_eq!(esc("say \"hello\""), "say \\\"hello\\\"");
    }

    #[test]
    fn esc_leaves_plain_strings_unchanged() {
        assert_eq!(esc("Team standup"), "Team standup");
    }

    // ── as_date_fragment ──────────────────────────────────────────────────────

    #[test]
    fn date_fragment_contains_all_components() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let frag = as_date_fragment("d", date, 3600);
        assert!(frag.contains("year of d to 2026"));
        assert!(frag.contains("month of d to 3"));
        assert!(frag.contains("day of d to 10"));
        assert!(frag.contains("time of d to 3600"));
    }

    // ── build_get_events_script ───────────────────────────────────────────────

    #[test]
    fn get_events_script_targets_calendar_app() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let script = build_get_events_script(date);
        assert!(script.contains("tell application \"Calendar\""));
        assert!(script.contains("year of startD to 2026"));
        assert!(script.contains("month of startD to 3"));
        assert!(script.contains("day of startD to 10"));
        assert!(script.contains("time of startD to 0"));
    }

    // ── build_create_event_script ─────────────────────────────────────────────

    #[test]
    fn create_event_script_contains_title_and_dates() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let script = build_create_event_script("Team meeting", date, 50400, 54000, None, None);
        assert!(script.contains("Team meeting"));
        assert!(script.contains("year of startD to 2026"));
        assert!(script.contains("time of startD to 50400"));
        assert!(script.contains("time of endD to 54000"));
    }

    #[test]
    fn create_event_script_with_calendar_name() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let script = build_create_event_script("Sprint review", date, 0, 3600, Some("Work"), None);
        assert!(script.contains("\"Work\""));
        assert!(!script.contains("item 1 of every calendar"));
    }

    #[test]
    fn create_event_script_without_calendar_uses_first() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let script = build_create_event_script("Dinner", date, 72000, 79200, None, None);
        assert!(script.contains("item 1 of every calendar"));
    }

    #[test]
    fn create_event_script_with_notes() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let script =
            build_create_event_script("Doctor", date, 36000, 39600, None, Some("Bring results"));
        assert!(script.contains("Bring results"));
        assert!(script.contains("description"));
    }

    #[test]
    fn create_event_script_escapes_quotes_in_title() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let script =
            build_create_event_script("Say \"hello\"", date, 0, 3600, None, None);
        assert!(script.contains("Say \\\"hello\\\""));
    }

    // ── build_create_reminder_script ──────────────────────────────────────────

    #[test]
    fn create_reminder_script_targets_reminders_app() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let script = build_create_reminder_script("Buy milk", date, Some(36000));
        assert!(script.contains("tell application \"Reminders\""));
        assert!(script.contains("Buy milk"));
        assert!(script.contains("time of dueD to 36000"));
    }

    #[test]
    fn create_reminder_script_without_time_uses_midnight() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let script = build_create_reminder_script("Take pills", date, None);
        assert!(script.contains("time of dueD to 0"));
    }

    // ── format_events_output ──────────────────────────────────────────────────

    #[test]
    fn format_events_parses_pipe_separated_lines() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let raw = "Work|Standup|09:00|09:30\nPersonal|Doctor|14:00|15:00\n";
        let result = format_events_output(raw, date);
        assert!(result.contains("09:00–09:30"));
        assert!(result.contains("Standup"));
        assert!(result.contains("14:00–15:00"));
        assert!(result.contains("Doctor"));
        assert!(result.contains("10/03/2026"));
    }

    #[test]
    fn format_events_sorts_by_time() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let raw = "Cal|Late meeting|15:00|16:00\nCal|Early standup|09:00|09:30\n";
        let result = format_events_output(raw, date);
        let early_pos = result.find("09:00").unwrap();
        let late_pos = result.find("15:00").unwrap();
        assert!(early_pos < late_pos, "events should be sorted by start time");
    }

    // ── tool metadata ─────────────────────────────────────────────────────────

    #[test]
    fn get_events_name_and_params() {
        let t = CalendarGetEventsTool;
        assert_eq!(t.name(), "get_calendar_events");
        assert!(t.parameters()["properties"]["date"].is_object());
    }

    #[test]
    fn create_event_name_and_params() {
        let t = CalendarCreateTool;
        assert_eq!(t.name(), "create_calendar_event");
        let p = t.parameters();
        assert!(p["properties"]["title"].is_object());
        assert!(p["properties"]["date"].is_object());
        assert!(p["properties"]["type"].is_object());
        assert!(p["properties"]["start_time"].is_object());
    }
}
