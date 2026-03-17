# Voicebot Project Structure

## Overview

This document describes the complete project structure created for the Voicebot application.

## Total Files Created: 32 Rust modules

## Directory Structure

```
src/
├── main.rs                                    # Entry point
├── lib.rs                                     # Library exports
├── config.rs                                  # Configuration (existing)
├── websocket_client.rs                        # WebSocket (existing)
│
├── audio/                                     # Audio Processing (6 files)
│   ├── mod.rs                                 # Module exports
│   ├── audio_capture.rs                       # Microphone capture (existing)
│   ├── audio_transform.rs                     # Audio transform (existing)
│   ├── vad.rs                                 # Voice Activity Detection ✨ NEW
│   ├── buffer.rs                              # Audio buffering ✨ NEW
│   └── output.rs                              # Speaker output ✨ NEW
│
├── session/                                   # Session Management (3 files) ✨ NEW
│   ├── mod.rs                                 # Module exports
│   ├── manager.rs                             # Session manager
│   └── context.rs                             # Conversation context
│
├── s2s/                                       # S2S Models (7 files) ✨ NEW
│   ├── mod.rs                                 # Module exports
│   ├── adapter.rs                             # Model adapter (abstraction layer)
│   └── models/
│       ├── mod.rs                             # Model types and config
│       ├── llama_omni.rs                      # LLaMA-Omni implementation
│       ├── moshi.rs                           # Moshi implementation
│       ├── ultravox.rs                        # Ultravox implementation
│       └── lfm.rs                             # LFM2.5-Audio implementation
│
├── tools/                                     # Tool System (7 files) ✨ NEW
│   ├── mod.rs                                 # Module exports
│   ├── router.rs                              # Tool router
│   ├── registry.rs                            # Tool registry
│   └── builtin/
│       ├── mod.rs                             # Built-in tools exports
│       ├── file_operations.rs                 # File I/O tool
│       ├── web_search.rs                      # Web search tool
│       └── system_info.rs                     # System info tool
│
├── mcp/                                       # MCP Protocol (3 files) ✨ NEW
│   ├── mod.rs                                 # Module exports
│   ├── server.rs                              # MCP server
│   └── protocol.rs                            # Protocol types
│
├── agents/                                    # External Agents (3 files) ✨ NEW
│   ├── mod.rs                                 # Module exports
│   ├── manager.rs                             # Agent manager
│   └── openclaw.rs                            # OpenClaw integration
│
└── db/                                        # Database (3 files) ✨ NEW
    ├── mod.rs                                 # Module exports
    ├── database.rs                            # SQLite operations
    └── schema.rs                              # Schema definitions
```

## Key Components and Their Responsibilities

### 1. Audio Layer (audio/)

#### VoiceActivityDetector (vad.rs)
- Detects speech vs silence in audio streams
- Configurable thresholds and durations
- Returns: `VadResult::Speech`, `VadResult::Silence`, `VadResult::SpeechStart`, `VadResult::SpeechEnd`

#### AudioBuffer (buffer.rs)
- Accumulates audio chunks
- Fixed-size circular buffer
- Provides duration tracking

#### AudioOutput (output.rs)
- Plays audio through speakers using `cpal`
- Manages output streams
- Handles audio playback

### 2. Session Layer (session/)

#### SessionManager (manager.rs)
- Creates and manages conversation sessions
- Handles message history
- Integrates with database for persistence
- Key methods:
  - `create_session()`
  - `add_message()`
  - `get_history()`
  - `load_session()`

#### ConversationContext (context.rs)
- Holds session state and messages
- Message types: `User`, `Assistant`, `System`, `Tool`
- Content types: `Text`, `Audio`, `ToolCall`, `ToolResult`

### 3. S2S Model Layer (s2s/)

#### S2SAdapter (adapter.rs)
- **Key Design**: Adapter pattern for interchangeable models
- Unified interface for all S2S models
- Supports: LlamaOmni, Moshi, Ultravox, LFM
- Methods:
  - `new(model_type, config)` - Create adapter with specific model
  - `process(request)` - Process audio and generate response
  - `supports_streaming()` - Check streaming support
  - `supports_tools()` - Check tool calling support

#### S2SModel Trait
- Interface all models must implement
- Methods:
  - `process(request) -> Result<S2SResponse>`
  - `supports_streaming() -> bool`
  - `supports_tools() -> bool`
  - `name() -> &str`

#### Model Implementations
Each model (LlamaOmni, Moshi, Ultravox, LFM) implements the S2SModel trait:
- **LlamaOmni**: Low-latency, supports streaming and tools
- **Moshi**: Full-duplex, streaming, no native tool support
- **Ultravox**: Whisper + Llama hybrid, supports tools
- **LFM**: Liquid AI audio model, supports streaming and tools

### 4. Tool Layer (tools/)

#### ToolRouter (router.rs)
- Routes tool calls between S2S model, MCP, and agents
- Integration points:
  - Built-in tools (via ToolRegistry)
  - MCP tools (via McpServer)
  - External agents (via AgentManager)
- Methods:
  - `execute_tool(name, args) -> Result<ToolResult>`
  - `list_tools() -> Vec<String>`

#### ToolRegistry (registry.rs)
- Manages built-in tools
- Tool trait with methods:
  - `name()`, `description()`, `definition()`
  - `execute(arguments) -> Result<ToolResult>`

#### Built-in Tools
1. **FileOperationsTool**: Read, write, list, delete files
2. **WebSearchTool**: Search the web (placeholder for API integration)
3. **SystemInfoTool**: CPU, memory, disk information

### 5. MCP Layer (mcp/)

#### McpServer (server.rs)
- Implements Model Context Protocol
- Manages MCP tool registration and execution
- Methods:
  - `connect(endpoint)`
  - `register_tool(tool)`
  - `execute_tool(name, args)`
  - `list_tools()`

### 6. Agent Layer (agents/)

#### AgentManager (manager.rs)
- Manages external AI agents
- Agent trait interface
- Methods:
  - `register_agent(agent)`
  - `call_agent(name, request)`
  - `list_agents()`

#### OpenClawAgent (openclaw.rs)
- Integration with OpenClaw external agent
- Implements Agent trait
- Supports complex task delegation

### 7. Database Layer (db/)

#### Database (database.rs)
- SQLite operations using `sqlx`
- Tables:
  - `sessions`: Session metadata
  - `messages`: Conversation messages
  - `config`: Configuration key-value store
- Methods:
  - `create_session()`
  - `save_message()`
  - `get_session_messages()`
  - `list_sessions()`
  - `save_config()`, `get_config()`

## Data Structures

### S2S Request/Response
```rust
struct S2SRequest {
    audio: Vec<f32>,
    sample_rate: u32,
    context: Vec<String>,
    tools: Option<Vec<ToolDefinition>>,
    stream: bool,
}

struct S2SResponse {
    audio: Vec<f32>,
    sample_rate: u32,
    input_text: Option<String>,
    output_text: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}
```

### Message Types
```rust
enum MessageRole { User, Assistant, System, Tool }

enum MessageContent {
    Text(String),
    Audio(Vec<f32>),
    ToolCall { name: String, args: String },
    ToolResult { name: String, result: String },
}
```

### Tool Types
```rust
struct ToolDefinition {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

struct ToolResult {
    success: bool,
    output: String,
    error: Option<String>,
}
```

## Design Patterns Used

1. **Adapter Pattern**: S2SAdapter abstracts different S2S models
2. **Registry Pattern**: ToolRegistry manages available tools
3. **Router Pattern**: ToolRouter routes calls to handlers
4. **Manager Pattern**: SessionManager, AgentManager
5. **Trait-based Polymorphism**: S2SModel, Tool, Agent traits

## Dependencies Required

Add these to `Cargo.toml`:

```toml
[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }
async-trait = "0.1"
async-channel = "2"

# Error handling
anyhow = "1"
thiserror = "1"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Database
sqlx = { version = "0.7", features = ["runtime-tokio-native-tls", "sqlite"] }

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# Audio (existing)
cpal = "0.15"

# System info
sysinfo = "0.30"

# UUID for sessions
uuid = { version = "1", features = ["v4", "serde"] }

# Time
chrono = { version = "0.4", features = ["serde"] }
```

## Next Steps

### Implementation Priorities

1. **Complete Audio Pipeline**
   - Integrate VAD with existing audio capture
   - Implement audio buffering strategy
   - Test audio output

2. **Database Setup**
   - Test SQLite migrations
   - Implement session persistence
   - Add indexing for performance

3. **S2S Model Integration**
   - Choose primary S2S model
   - Implement actual inference (currently placeholders)
   - Add model loading from disk
   - Test audio-to-audio pipeline

4. **Tool System**
   - Implement actual web search API
   - Add more built-in tools
   - Test tool execution pipeline

5. **MCP Integration**
   - Implement MCP protocol client
   - Test with MCP servers
   - Add error handling

6. **Agent Integration**
   - Complete OpenClaw integration
   - Test agent communication
   - Add more agents as needed

7. **Main Application Loop**
   - Integrate all components in main.rs
   - Add proper error handling
   - Implement graceful shutdown

8. **Testing**
   - Unit tests for each component
   - Integration tests for pipelines
   - End-to-end testing

## Architecture Benefits

✅ **Modularity**: Each component is independent and testable
✅ **Extensibility**: Easy to add new models, tools, and agents
✅ **Flexibility**: Adapter pattern allows swapping S2S models
✅ **Maintainability**: Clear separation of concerns
✅ **Scalability**: Async architecture supports high throughput
✅ **Type Safety**: Strong Rust types prevent many bugs

## Notes

- All implementations include TODO comments for actual inference/API integration
- Placeholder implementations return mock data for testing structure
- Error handling uses `anyhow::Result` consistently
- All async operations use Tokio runtime
- Database uses SQLx for compile-time query checking
