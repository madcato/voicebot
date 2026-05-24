use std::env;

/// Single external agent definition loaded from environment variables.
///
/// Each agent gets its own tool (`run_{name}`), its own system prompt
/// section (when-to-use + instructions), and independent mode (cli/acp).
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Unique name used as tool suffix: `run_{name}`.
    pub name: String,
    /// Communication mode: `"cli"` (one-shot subprocess) or `"acp"` (persistent JSON-RPC stdio).
    pub mode: String,
    /// CLI command (e.g. `"hermes chat"`). Used only when `mode = "cli"`.
    pub command: Option<String>,
    /// ACP command (e.g. `"hermes acp"`). Used only when `mode = "acp"`.
    pub acp_command: String,
    /// When true, send a warmup prompt at startup to force model load.
    pub acp_warmup: bool,
    /// LLM-facing text: when to delegate to this agent.
    /// Appended to the system prompt so the primary LLM knows which agent to pick.
    pub when_to_use: String,
    /// Agent-facing instructions. Prepended to the query sent to the agent subprocess
    /// so it knows how to behave (role, style, capabilities).
    pub instructions: String,
}

/// Registry of all configured external agents.
///
/// Created once at startup from environment variables. Supports both the
/// new multi-agent format (`AGENTS=hermes,<nombre_agente>`) and the legacy
/// single-agent format (`AGENT_COMMAND` / `AGENT_MODE`).
#[derive(Debug, Clone)]
pub struct AgentRegistry {
    pub agents: Vec<AgentConfig>,
}

impl AgentRegistry {
    /// Load agents from environment variables.
    ///
    /// Priority:
    /// 1. If `AGENTS` is set → parse comma-separated names, load each via `AGENT_<NAME>_*`.
    /// 2. If `AGENT_COMMAND` or `AGENT_MODE` is set → create single `"hermes"` agent from legacy vars.
    /// 3. Otherwise → empty registry (no agent tools).
    pub fn from_env() -> Self {
        // ── New multi-agent format ──────────────────────────────────────
        if let Ok(raw) = env::var("AGENTS") {
            let names: Vec<&str> = raw
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            if !names.is_empty() {
                let agents = names.into_iter().filter_map(load_agent_from_env).collect();
                return Self { agents };
            }
        }

        // ── Legacy single-agent format (backward compatibility) ─────────
        let has_command = env::var("AGENT_COMMAND").is_ok();
        let has_mode = env::var("AGENT_MODE").is_ok();
        let has_acp = env::var("AGENT_ACP_COMMAND").is_ok();

        if (has_command || has_mode || has_acp)
            && let Some(agent) = load_legacy_agent()
        {
            return Self {
                agents: vec![agent],
            };
        }

        // ── No agents configured ────────────────────────────────────────
        Self { agents: Vec::new() }
    }

    /// Build the system prompt section describing all available agents.
    ///
    /// Returns an empty string if no agents are configured.
    /// The result is meant to be inserted between memory context and tool
    /// instructions in the full system prompt.
    pub fn system_prompt_section(&self) -> String {
        if self.agents.is_empty() {
            return String::new();
        }

        let mut section = String::from(
            "\n\n## AGENTES EXTERNOS DISPONIBLES\n\n\
             Puedes delegar tareas complejas a los siguientes agentes externos.\n\
             Cada agente tiene herramientas propias y especialización.\n\
             Para delegar, llama a la herramienta correspondiente (run_<nombre>) \n\
             con task=\"descripción de la tarea\". El resultado llega de forma proactiva.\n",
        );

        for agent in &self.agents {
            section.push_str(&format!(
                "\n### {display_name} (run_{name})\n\
                 Cuándo usar: {when}\n\
                 Instrucciones para el agente: {instructions}\n",
                display_name = capitalize(&agent.name),
                name = agent.name,
                when = agent.when_to_use,
                instructions = agent.instructions,
            ));
        }

        section
    }
}

/// Load a single agent from env vars using the `AGENT_<NAME>_*` convention.
///
/// Returns `None` if the agent has no valid command configured (neither CLI nor ACP).
fn load_agent_from_env(name: &str) -> Option<AgentConfig> {
    let upper = name.to_uppercase().replace('-', "_");

    let mode = env::var(format!("AGENT_{}_MODE", upper)).unwrap_or_else(|_| "acp".to_string());

    let command = env::var(format!("AGENT_{}_COMMAND", upper)).ok();
    let acp_command = env::var(format!("AGENT_{}_ACP_COMMAND", upper))
        .unwrap_or_else(|_| default_acp_command(name));

    // Validate: at least one command must be available for the chosen mode.
    if mode == "cli" && command.is_none() {
        return None;
    }
    if mode == "acp" && acp_command.is_empty() {
        return None;
    }

    let acp_warmup = env::var(format!("AGENT_{}_ACP_WARMUP", upper)).as_deref() == Ok("1");

    let when_to_use = env::var(format!("AGENT_{}_WHEN_TO_USE", upper))
        .unwrap_or_else(|_| default_when_to_use(name));

    let instructions = env::var(format!("AGENT_{}_INSTRUCTIONS", upper))
        .unwrap_or_else(|_| default_instructions(name));

    Some(AgentConfig {
        name: name.to_string(),
        mode,
        command,
        acp_command,
        acp_warmup,
        when_to_use,
        instructions,
    })
}

/// Load a single agent from legacy env vars (backward compatibility).
fn load_legacy_agent() -> Option<AgentConfig> {
    let command = env::var("AGENT_COMMAND").ok();
    let mode = env::var("AGENT_MODE").unwrap_or_else(|_| "cli".to_string());
    let acp_command = env::var("AGENT_ACP_COMMAND").unwrap_or_else(|_| "hermes acp".to_string());
    let acp_warmup = env::var("AGENT_ACP_WARMUP").as_deref() == Ok("1");

    // Legacy agents use built-in defaults for when_to_use and instructions.
    let when_to_use = default_when_to_use("hermes");
    let instructions =
        env::var("AGENT_PROMPT_INSTRUCTIONS").unwrap_or_else(|_| default_instructions("hermes"));

    Some(AgentConfig {
        name: "hermes".to_string(),
        mode,
        command,
        acp_command,
        acp_warmup,
        when_to_use,
        instructions,
    })
}

/// Default ACP command for known agent names.
fn default_acp_command(name: &str) -> String {
    match name {
        "hermes" => "hermes acp".to_string(),
        _ => format!("{} acp", name),
    }
}

/// Default "when to use" text for known agent names.
fn default_when_to_use(name: &str) -> String {
    match name {
        "hermes" => "Único agente externo disponible. Punto de contacto para lo que Voicebot \
                      no puede resolver con sus propias herramientas: programación y código, \
                      investigación profunda, gestión de calendario, flujos de múltiples pasos, \
                      y cualquier tarea que requiera razonamiento extendido o herramientas \
                      del sistema."
            .to_string(),
        _ => format!("Tareas delegables al agente externo {name}."),
    }
}

/// Default instructions for known agent names.
fn default_instructions(name: &str) -> String {
    match name {
        "hermes" => "Eres Hermes, el gateway de agentes externos. Puedes redirigir \
                      internamente a especialistas (programadores, investigadores, etc.). \
                      Recibe la consulta de Voicebot, coordina los recursos necesarios y devuelves \
                      una respuesta clara y ejecutable."
            .to_string(),
        _ => format!(
            "Eres un agente externo ({name}). Resuelve la tarea delegada de forma autónoma."
        ),
    }
}

/// Capitalize first letter of a string.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use temp_env::with_vars;

    #[test]
    fn empty_registry_when_no_env() {
        with_vars(&[] as &[(&str, Option<&str>); 0], || {
            let reg = AgentRegistry::from_env();
            assert!(reg.agents.is_empty());
        });
    }

    #[test]
    fn legacy_single_agent_fallback() {
        with_vars(
            [
                ("AGENT_COMMAND", Some("hermes chat")),
                ("AGENT_MODE", Some("cli")),
            ],
            || {
                let reg = AgentRegistry::from_env();
                assert_eq!(reg.agents.len(), 1);
                assert_eq!(reg.agents[0].name, "hermes");
                assert_eq!(reg.agents[0].mode, "cli");
                assert_eq!(reg.agents[0].command.as_deref(), Some("hermes chat"));
            },
        );
    }

    #[test]
    fn legacy_acp_mode_fallback() {
        with_vars(
            [
                ("AGENT_MODE", Some("acp")),
                ("AGENT_ACP_COMMAND", Some("hermes acp")),
            ],
            || {
                let reg = AgentRegistry::from_env();
                assert_eq!(reg.agents.len(), 1);
                assert_eq!(reg.agents[0].name, "hermes");
                assert_eq!(reg.agents[0].mode, "acp");
                assert_eq!(reg.agents[0].acp_command, "hermes acp");
            },
        );
    }

    #[test]
    fn multi_agent_from_env() {
        with_vars(
            [
                ("AGENTS", Some("hermes,generic_test")),
                ("AGENT_HERMES_MODE", Some("acp")),
                ("AGENT_HERMES_ACP_COMMAND", Some("hermes acp")),
                ("AGENT_GENERIC_TEST_MODE", Some("acp")),
                ("AGENT_GENERIC_TEST_ACP_COMMAND", Some("generic_test acp")),
            ],
            || {
                let reg = AgentRegistry::from_env();
                assert_eq!(reg.agents.len(), 2);
                assert_eq!(reg.agents[0].name, "hermes");
                assert_eq!(reg.agents[1].name, "generic_test");
            },
        );
    }

    #[test]
    fn system_prompt_section_empty_for_no_agents() {
        let reg = AgentRegistry { agents: vec![] };
        assert!(reg.system_prompt_section().is_empty());
    }

    #[test]
    fn system_prompt_section_non_empty_for_agents() {
        let reg = AgentRegistry {
            agents: vec![AgentConfig {
                name: "hermes".to_string(),
                mode: "acp".to_string(),
                command: None,
                acp_command: "hermes acp".to_string(),
                acp_warmup: false,
                when_to_use: "Test when to use".to_string(),
                instructions: "Test instructions".to_string(),
            }],
        };
        let section = reg.system_prompt_section();
        assert!(section.contains("AGENTES EXTERNOS"));
        assert!(section.contains("run_hermes"));
        assert!(section.contains("Test when to use"));
        assert!(section.contains("Test instructions"));
    }

    #[test]
    fn default_acp_command_known_agents() {
        assert_eq!(default_acp_command("hermes"), "hermes acp");
        assert_eq!(default_acp_command("unknown"), "unknown acp");
    }

    #[test]
    fn capitalize_works() {
        assert_eq!(capitalize("hermes"), "Hermes");
        assert_eq!(capitalize("my_agent"), "My_agent");
        assert_eq!(capitalize(""), "");
    }
}
