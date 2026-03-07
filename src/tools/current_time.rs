use chrono::Local;

use super::Tool;

pub struct CurrentTimeTool;

impl Tool for CurrentTimeTool {
    fn name(&self) -> &str {
        "current_time"
    }

    fn description(&self) -> &str {
        "Returns the current local date and time."
    }

    fn run(&self) -> String {
        Local::now().format("%H:%M:%S, %A %d %B %Y").to_string()
    }
}
