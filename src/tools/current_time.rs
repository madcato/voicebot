use async_trait::async_trait;
use chrono::Local;

use super::Tool;

pub struct CurrentTimeTool;

#[async_trait]
impl Tool for CurrentTimeTool {
    fn name(&self) -> &str {
        "current_time"
    }

    fn description(&self) -> &str {
        "Returns the current local date and time."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn run(&self, _args: &str) -> String {
        Local::now().format("%H:%M:%S, %A %d %B %Y").to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_current_time() {
        assert_eq!(CurrentTimeTool.name(), "current_time");
    }

    #[test]
    fn description_is_non_empty() {
        assert!(!CurrentTimeTool.description().is_empty());
    }

    #[tokio::test]
    async fn run_returns_non_empty_string() {
        let result = CurrentTimeTool.run("").await;
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn run_output_matches_format() {
        // Expected: "HH:MM:SS, Weekday DD Month YYYY"
        // Example:  "14:05:32, Saturday 08 March 2025"
        let result = CurrentTimeTool.run("").await;
        let parts: Vec<&str> = result.splitn(2, ", ").collect();
        assert_eq!(parts.len(), 2, "output must contain ', ' separator: {result:?}");

        // Time part: HH:MM:SS
        let time = parts[0];
        let time_parts: Vec<&str> = time.split(':').collect();
        assert_eq!(time_parts.len(), 3, "time must be HH:MM:SS: {time:?}");
        let h: u32 = time_parts[0].parse().expect("hours must be numeric");
        let m: u32 = time_parts[1].parse().expect("minutes must be numeric");
        let s: u32 = time_parts[2].parse().expect("seconds must be numeric");
        assert!(h < 24, "hours out of range: {h}");
        assert!(m < 60, "minutes out of range: {m}");
        assert!(s < 60, "seconds out of range: {s}");

        // Date part: "Weekday DD Month YYYY"
        let date = parts[1];
        let date_parts: Vec<&str> = date.split_whitespace().collect();
        assert_eq!(date_parts.len(), 4, "date must have 4 parts: {date:?}");
        let day: u32 = date_parts[1].parse().expect("day must be numeric");
        let year: u32 = date_parts[3].parse().expect("year must be numeric");
        assert!((1..=31).contains(&day), "day out of range: {day}");
        assert!(year >= 2024, "year seems wrong: {year}");
    }

    #[tokio::test]
    async fn run_output_is_consistent_within_same_second() {
        // Two consecutive calls should return the same second (or differ by at most 1s).
        let before = Local::now();
        let result = CurrentTimeTool.run("").await;
        let after = Local::now();

        let result_time_str = result.splitn(2, ", ").next().unwrap();
        let b = before.format("%H:%M:%S").to_string();
        let a = after.format("%H:%M:%S").to_string();
        assert!(
            result_time_str >= b.as_str() || result_time_str <= a.as_str(),
            "time {result_time_str:?} should be within [{b}, {a}]"
        );
    }

    #[tokio::test]
    async fn run_ignores_args() {
        // current_time does not use args — any input should still return a valid time
        let result = CurrentTimeTool.run("ignored args").await;
        assert!(result.contains(':'), "should still return a time: {result:?}");
    }
}
