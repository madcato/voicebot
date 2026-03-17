# 📋 Integration Testing Strategy for Voicebot

## 🎯 Test Layers Overview

Your project has 5 distinct integration layers that need testing:

1. **Audio Pipeline** - Microphone → VAD → Audio → Speaker
2. **Session Management** - State persistence with SQLite
3. **S2S Model Integration** - Model adapter and tool routing
4. **Tool System** - Built-in, MCP, and agent tools
5. **Agent Integration** - External agent interactions

---

## 🧪 Recommended Testing Approach

### **1. Audio Pipeline Tests**

**Target components**: `src/audio/vad.rs`, `src/audio/buffer.rs`, `src/audio/output.rs`

```rust
// tests/audio/vad_integration.rs
use voicebot::audio::vad::{VoiceActivityDetector, VadResult};

#[tokio::test]
async fn test_vad_detects_speech_from_buffer() {
    let mut vad = VoiceActivityDetector::new(16000);
    
    // Generate synthetic speech audio
    let speech_audio = generate_speech_samples(16000); // 1 second at 16kHz
    
    let result = vad.process(&speech_audio);
    
    assert!(matches!(result, VadResult::SpeechStart | VadResult::Speech));
}

#[tokio::test]
async fn test_audio_buffer_accumulates_chunks() {
    let mut buffer = voicebot::audio::buffer::AudioBuffer::new(16000, 5); // 5 seconds max
    
    let chunk1 = vec![0.1f32; 800]; // 50ms
    let chunk2 = vec![0.2f32; 800];
    
    buffer.push(&chunk1);
    buffer.push(&chunk2);
    
    assert_eq!(buffer.len(), 1600);
}
```

**Key concerns**:
- Use synthetic/silent audio to avoid hardware dependency
- Mock `cpal` for output testing
- Test VAD state machines thoroughly

---

### **2. Session & Database Tests**

**Target components**: `src/session/manager.rs`, `src/db/database.rs`

```rust
// tests/session/integration_tests.rs
use voicebot::db::Database;
use voicebot::session::{SessionManager, Message};

#[tokio::test]
async fn test_session_lifecycle() {
    // Use in-memory SQLite for testing
    let db = Database::new("file:memdb?mode=memory&cache=shared").await.unwrap();
    let mut manager = SessionManager::new(db);
    
    // Create session
    let session_id = manager.create_session().await.unwrap();
    
    // Add messages
    let msg = Message::user_text("Hello");
    manager.add_message(msg).await.unwrap();
    
    // Verify persistence
    let history = manager.get_history();
    assert_eq!(history.len(), 1);
    
    // Close session
    manager.close_session().await.unwrap();
}

#[tokio::test]
async fn test_session_persists_messages() {
    let db = Database::new("file:memdb?mode=memory&cache=shared").await.unwrap();
    let mut manager = SessionManager::new(db);
    
    let session_id = manager.create_session().await.unwrap();
    
    // Add multiple messages
    for i in 0..5 {
        let msg = Message::assistant_text(format!("Response {}", i));
        manager.add_message(msg).await.unwrap();
    }
    
    // Verify all messages persisted
    let history = manager.get_history();
    assert_eq!(history.len(), 5);
    
    // Load session from DB and verify messages
    manager.load_session(session_id).await.unwrap();
    let loaded_history = manager.get_history();
    assert_eq!(loaded_history.len(), 5);
}
```

**Key concerns**:
- Use SQLite in-memory mode for fast tests
- Test transaction rollbacks
- Verify foreign key constraints

---

### **3. S2S Model & Tool Integration Tests**

**Target components**: `src/s2s/adapter.rs`, `src/tools/router.rs`

```rust
// tests/s2s/tool_routing_integration.rs
use voicebot::tools::{ToolRouter, registry::{Tool, ToolResult}};
use voicebot::s2s::{S2SAdapter, ModelConfig, ModelType};

#[tokio::test]
async fn test_tool_router_selects_built_in_tool() {
    let router = ToolRouter::new();
    
    // Register mock tool
    struct MockTool;
    
    impl Tool for MockTool {
        fn name(&self) -> &str { "test_tool" }
        fn description(&self) -> &str { "Mock tool for testing" }
        
        async fn execute(&self, _args: &str) -> anyhow::Result<ToolResult> {
            Ok(ToolResult::success("Mock executed".to_string()))
        }
    }
    
    router.register_tool(Box::new(MockTool)).await.unwrap();
    
    // Execute tool
    let result = router.execute_tool("test_tool", "{\"param\": \"value\"}").await;
    
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_s2s_adapter_with_mock_model() {
    use voicebot::s2s::adapter::{S2SModel, S2SRequest, S2SResponse};
    use async_trait::async_trait;
    
    // Create mock model
    struct MockS2SModel;
    
    #[async_trait]
    impl S2SModel for MockS2SModel {
        async fn process(&mut self, _request: S2SRequest) -> anyhow::Result<S2SResponse> {
            Ok(S2SResponse {
                audio: vec![0.0; 16000],
                sample_rate: 16000,
                input_text: Some("hello".to_string()),
                output_text: Some("world".to_string()),
                tool_calls: None,
            })
        }
        
        fn supports_streaming(&self) -> bool { false }
        fn supports_tools(&self) -> bool { true }
        fn name(&self) -> &str { "MockModel" }
    }
    
    // Test the adapter with mock
    let config = ModelConfig::default();
    let mut adapter = S2SAdapter::new(config, Box::new(MockS2SModel)).await.unwrap();
    
    let request = S2SRequest {
        audio: vec![0.0; 8000],
        sample_rate: 16000,
        context: Vec::new(),
        tools: None,
        stream: false,
    };
    
    let response = adapter.process(request).await.unwrap();
    assert!(response.output_text.is_some());
}
```

**Key concerns**:
- Use `mockall` for mocking S2SModel trait
- Test tool priority (built-in → MCP → agent)
- Verify error handling

---

### **4. Tool System Integration Tests**

**Target components**: `src/tools/builtin/`

```rust
// tests/tools/file_operations_test.rs
use voicebot::tools::builtin::file_operations::FileOperationsTool;
use tempfile::NamedTempFile;

#[tokio::test]
async fn test_file_tool_operations() {
    let tool = FileOperationsTool::new();
    
    // Create temp file
    let tmp_file = NamedTempFile::new().unwrap();
    let test_path = tmp_file.path().to_string_lossy();
    let content = "test content for voicebot";
    
    // Write operation
    let args = serde_json::json!({
        "operation": "write",
        "path": test_path.as_ref(),
        "content": content
    }).to_string();
    
    let result = tool.execute(&args).await.unwrap();
    assert!(result.success);
    assert!(result.output.contains("written successfully"));
    
    // Read operation
    let read_args = serde_json::json!({
        "operation": "read",
        "path": test_path.as_ref()
    }).to_string();
    
    let read_result = tool.execute(&read_args).await.unwrap();
    assert!(read_result.output.contains(content));
    
    // Cleanup is automatic with NamedTempFile
}

#[tokio::test]
async fn test_file_tool_error_handling() {
    let tool = FileOperationsTool::new();
    
    // Test non-existent file read
    let args = serde_json::json!({
        "operation": "read",
        "path": "/nonexistent/path/file.txt"
    }).to_string();
    
    let result = tool.execute(&args).await.unwrap();
    assert!(!result.success);
    assert!(result.error.is_some());
}

#[tokio::test]
async fn test_file_tool_list_directory() {
    let tool = FileOperationsTool::new();
    
    // Create temp directory with files
    let tmp_dir = tempfile::tempdir().unwrap();
    std::fs::write(tmp_dir.path().join("file1.txt"), "content1").unwrap();
    std::fs::write(tmp_dir.path().join("file2.txt"), "content2").unwrap();
    
    let args = serde_json::json!({
        "operation": "list",
        "path": tmp_dir.path().to_string_lossy()
    }).to_string();
    
    let result = tool.execute(&args).await.unwrap();
    assert!(result.success);
    assert!(result.output.contains("file1.txt"));
    assert!(result.output.contains("file2.txt"));
}
```

**Key concerns**:
- Test with temporary files using `tempfile` crate
- Test async operations properly
- Verify error cases (file not found, permissions)

---

### **5. Agent Integration Tests**

**Target components**: `src/agents/openclaw.rs`

```rust
// tests/agents/openclaw_test.rs
use voicebot::agents::openclaw::OpenClawAgent;

#[tokio::test]
async fn test_openclay_agent_call() {
    let mut agent = OpenClawAgent::new("http://localhost:8080".to_string());
    
    let result = agent.call_agent("test_task", "test arguments").await;
    
    // Since OpenClaw integration is a TODO, test the structure
    assert!(result.is_ok());
}
```

**Key concerns**:
- Use `mockito` or similar for HTTP mocks
- Test connection failures gracefully

---

## 📦 Testing Dependencies to Add

Add these to your `Cargo.toml`:

```toml
[dev-dependencies]
# Mocking framework
mockall = "0.14"
tokio-test = "0.4"

# Async testing utilities
async-channel = "2"
futures = "0.3"

# HTTP mocking
mockito = "1.3"
reqwest = { version = "0.12", features = ["json"] }

# Temporary file handling
tempfile = "3.8"

# Async process testing
async-process = "2.5"
```

---

## 📁 Test File Structure

```
tests/
├── mod.rs                              # Test module declarations
│
├── audio/
│   ├── vad_tests.rs                    # VAD integration tests
│   └── buffer_tests.rs                 # Audio buffer tests
│
├── session/
│   ├── lifecycle_tests.rs              # Session create/persist/close
│   └── message_flow_tests.rs           # Message storage/retrieval
│
├── s2s/
│   ├── adapter_tests.rs                # Adapter pattern tests
│   └── model_integration_tests.rs      # Full S2S pipeline
│
├── tools/
│   ├── router_tests.rs                 # Tool routing integration
│   ├── builtin/
│   │   ├── file_ops_integration.rs     # File tool tests
│   │   ├── web_search_tests.rs         # Web search (mocked)
│   │   └── system_info_tests.rs        # System tool tests
│   └── registry_tests.rs               # Tool registry tests
│
├── mcp/
│   ├── server_integration.rs           # MCP server tests
│   └── protocol_tests.rs               # Protocol encoding/decoding
│
├── agents/
│   └── openclaw_tests.rs               # OpenClaw integration
│
└── integration/
    ├── full_conversation_test.rs       # End-to-end conversation flow
    └── tool_agent_combo_tests.rs       # Tools + agents together

common/
└── mod.rs                              # Common test utilities
```

---

## 🛠️ Recommended Testing Patterns

### **1. Test Database Setup**

```rust
// tests/common/db.rs
use voicebot::db::Database;

pub async fn init_test_db() -> Database {
    let db_path = create_unique_db_path();
    Database::new(&db_path).await.unwrap()
}

fn create_unique_db_path() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros();
    format!("/tmp/voicebot_test_{}.db", timestamp)
}
```

### **2. Mock Tool Implementation**

```rust
// tests/common/mock_tool.rs
use async_trait::async_trait;
use voicebot::tools::registry::{Tool, ToolResult};

pub struct MockTestTool;

#[async_trait]
impl Tool for MockTestTool {
    fn name(&self) -> &str { "mock_tool" }
    
    fn description(&self) -> &str { "Mock tool for testing" }
    
    async fn execute(&self, args: &str) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success(format!("Mock executed: {}", args)))
    }
}
```

### **3. Test Audio Generation**

```rust
// tests/common/audio.rs
pub fn generate_silent_audio(sample_rate: u32, duration_ms: u32) -> Vec<f32> {
    let num_samples = (sample_rate as u32 * duration_ms) / 1000;
    vec![0.0f32; num_samples as usize]
}

pub fn generate_speech_like_audio(sample_rate: u32, duration_ms: u32) -> Vec<f32> {
    let num_samples = (sample_rate as u32 * duration_ms) / 1000;
    let mut audio = Vec::with_capacity(num_samples as usize);
    
    // Generate pseudo-random speech-like signal
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        // Mixed frequencies for speech-like waveform
        let sample = (t * 100.0).sin() * 0.5 + (t * 250.0).sin() * 0.3;
        audio.push(sample.clamp(-1.0, 1.0));
    }
    
    audio
}
```

### **4. Mock S2S Model**

```rust
// tests/common/mock_s2s_model.rs
use async_trait::async_trait;
use voicebot::s2s::adapter::{S2SModel, S2SRequest, S2SResponse};

pub struct MockS2SModel {
    pub expect_input_text: Option<String>,
    pub expect_output_text: Option<String>,
}

impl MockS2SModel {
    pub fn new() -> Self {
        Self {
            expect_input_text: None,
            expect_output_text: None,
        }
    }
}

#[async_trait]
impl S2SModel for MockS2SModel {
    async fn process(&mut self, request: S2SRequest) -> anyhow::Result<S2SResponse> {
        Ok(S2SResponse {
            audio: vec![0.0; 16000],
            sample_rate: 16000,
            input_text: request.context.first().cloned(),
            output_text: Some("Mock response".to_string()),
            tool_calls: None,
        })
    }
    
    fn supports_streaming(&self) -> bool { false }
    fn supports_tools(&self) -> bool { true }
    fn name(&self) -> &str { "MockModel" }
}
```

---

## 🎨 Test Organization by Priority

### **Tier 1: Core Integration (High Priority)**
- [ ] Session creation and message persistence
- [ ] VAD detection from audio input
- [ ] Tool router basic functionality

### **Tier 2: Tool Integration (Medium Priority)**
- [ ] File operations with temp files
- [ ] Built-in tool error handling
- [ ] MCP tool registration and execution

### **Tier 3: End-to-End (Lower Priority)**
- [ ] Full audio pipeline
- [ ] Agent integration with mocking
- [ ] Complete conversation flow tests

---

## 📋 Testing Best Practices for Your Project

1. **Use in-memory SQLite** for database tests (faster, no cleanup)
2. **Mock external dependencies** (agents, MCP) with `mockall`
3. **Use temporary files** for file operation tests
4. **Generate synthetic audio** - don't require real hardware
5. **Test async flows** with `tokio::test` and proper timeouts
6. **Use fixtures** for common test data (messages, sessions)
7. **Test error paths** thoroughly - Voicebot will face real-world errors
8. **Parallel tests** where possible with `cargo test -- --test-threads=4`

---

## 🚀 Sample Test Command Structure

```bash
# Run all tests
cargo test

# Run specific test module
cargo test -- session::lifecycle_tests

# Run with coverage (requires grcov)
cargo test && grcov . --source-dir ./src -t html --output-path target/coverage/

# Run only integration tests
cargo test --test '*integration*'

# Test with specific model adapter
cargo test -- s2s::adapter --nocapture

# Run tests with verbose output
cargo test -- --test-threads=1 --nocapture
```

---

## 📊 Coverage Goals

- **Unit tests**: 80%+ coverage for core modules
- **Integration tests**: Test all user-facing flows
- **E2E tests**: Critical path only (session → tool → agent)

---

## 🎯 Implementation Roadmap

### Phase 1: Foundation (Week 1)
- Setup test infrastructure
- Write database integration tests
- Implement VAD unit tests

### Phase 2: Core Flows (Week 2)
- Session lifecycle tests
- Tool router integration tests
- Built-in tools integration

### Phase 3: Advanced Integration (Week 3)
- S2S model adapter tests
- MCP server integration
- Agent endpoint mocks

### Phase 4: End-to-End (Week 4)
- Full conversation flow tests
- Tool + agent combination tests
- Performance and stress tests

---

## 🔧 Example Complete Integration Test

```rust
// tests/integration/full_conversation_test.rs
use voicebot::db::Database;
use voicebot::session::{SessionManager, Message};
use voicebot::tools::{ToolRouter, registry::{Tool, ToolResult}};
use async_trait::async_trait;

#[tokio::test]
async fn test_full_conversation_flow() {
    // Setup
    let db_path = "/tmp/test_conversation.db";
    std::fs::remove_file(db_path).ok(); // Clean up from previous run
    
    let db = Database::new(db_path).await.unwrap();
    let mut manager = SessionManager::new(db);
    
    // Create tool router with mock tool
    let mut router = ToolRouter::new();
    
    struct TestTool;
    
    #[async_trait]
    impl Tool for TestTool {
        fn name(&self) -> &str { "test_tool" }
        fn description(&self) -> &str { "Test tool" }
        
        async fn execute(&self, _args: &str) -> anyhow::Result<ToolResult> {
            Ok(ToolResult::success("Tool result".to_string()))
        }
    }
    
    router.register_tool(Box::new(TestTool)).await.unwrap();
    
    // Create session
    let session_id = manager.create_session().await.unwrap();
    
    // Simulate user message
    let user_msg = Message::user_text("Hello, how are you?");
    manager.add_message(user_msg).await.unwrap();
    
    // Simulate tool usage
    let result = router.execute_tool("test_tool", "{}").await.unwrap();
    assert!(result.success);
    
    // Simulate assistant response
    let assistant_msg = Message::assistant_text("I'm doing well, thanks for asking!");
    manager.add_message(assistant_msg).await.unwrap();
    
    // Verify conversation history
    let history = manager.get_history();
    assert_eq!(history.len(), 2);
    
    // Cleanup
    std::fs::remove_file(db_path).ok();
}
```

---

This testing strategy covers all the critical integration points in your Voicebot project. Start with Tier 1 tests, then move to Tier 2 and Tier 3 as you build out more complex features.

Would you like me to provide starter code for any specific test module, or help you set up the initial test structure?
