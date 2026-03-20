# ACP Protocol Information

This document contains essential information about the Agent Communication Protocol (ACP) and how Hermes implements it.

## 🔗 Official URLs & Documentation

### Primary Resources

| Resource | URL |
|----------|-----|
| **Official GitHub Repository** | https://github.com/vilukes/agent-client-protocol |
| **Protocol Specification** | https://github.com/vilukes/agent-client-protocol/blob/main/PROTOCOL.md |
| **Python Package (PyPI)** | https://pypi.org/project/agent-client-protocol/ |
| **TypeScript Reference** | https://github.com/vilukes/agent-client-protocol/tree/main/packages/acp |

### Additional Resources

- **Protocol README**: https://github.com/vilukes/agent-client-protocol/blob/main/README.md
- **Examples Repository**: Check the GitHub repo for TypeScript examples
- **Discussions**: https://github.com/vilukes/agent-client-protocol/discussions

## 📋 Protocol Summary

### Core Concepts

1. **Transport Layer**: JSON-RPC 2.0 over stdio (stdin/stdout) or WebSocket
2. **Message Format**: Newline-delimited JSON objects
3. **Session Model**: Each conversation has a unique session_id
4. **Streaming**: Server sends `sessionUpdate` notifications for real-time progress

### Protocol Methods

#### Client → Server Requests

| Method | Purpose | Required Params |
|--------|---------|-----------------|
| `initialize` | Handshake & capability negotiation | protocol_version, client_info |
| `new_session` | Create new conversation context | cwd |
| `prompt` | Send message to agent | session_id, prompt[] |
| `fork_session` | Clone existing session | session_id, cwd? |
| `load_session` | Resume previous session | session_id, cwd |
| `resume_session` | Continue suspended session | session_id, cwd |
| `cancel` | Stop current operation | session_id |
| `list_sessions` | Enumerate active sessions | cursor?, cwd? |

#### Server → Client Notifications

| Notification | Purpose |
|--------------|---------|
| `sessionUpdate` | Streaming updates (messages, tool calls, progress) |

### Message Formats

**Text Content Block:**
```json
{"type": "text", "text": "Your message here"}
```

**Image Content Block:**
```json
{
  "type": "image",
  "data": "...base64...",
  "mimeType": "image/png"
}
```

### Stop Reasons (Response Status)

- `end_turn` - Normal completion
- `cancelled` - Operation was cancelled
- `refusal` - Request was refused

## 🏗️ Hermes Implementation Details

### Location in Hermes Codebase

```
~/.hermes/hermes-agent/acp_adapter/
├── __init__.py        # Package init: "ACP (Agent Communication Protocol) adapter"
├── __main__.py        # Allows: python -m acp_adapter
├── entry.py           # CLI entry point → hermes-acp command
├── server.py          # HermesACPAgent class (main implementation)
├── session.py         # SessionManager, SessionState classes
├── events.py          # Callback factories for streaming
├── permissions.py     # Permission bridging to editor/IDE
├── tools.py           # Tool definition mapping
└── auth.py            # Provider detection (optional)
```

### Key Classes in Hermes

#### `HermesACPAgent` (server.py:78)

Main ACP server implementation. Inherits from `acp.Agent`.

**Key Methods:**
- `initialize()` - Protocol handshake
- `new_session(cwd)` - Create session
- `prompt(prompt[], session_id)` - Process user message
- `fork_session(session_id, cwd)` - Fork context
- `_handle_slash_command()` - Internal commands (/model, /tools, etc.)

#### `SessionManager` (session.py:49)

Manages all active sessions and their state.

**Features:**
- Per-session conversation history
- Model configuration per session
- Working directory tracking
- Thread-safe operations

### Installation in Hermes

```toml
# From ~/.hermes/hermes-agent/pyproject.toml

[project.optional-dependencies]
acp = ["agent-client-protocol>=0.8.1,<1.0"]

[project.scripts]
hermes-acp = "acp_adapter.entry:main"
```

Install with:
```bash
pip install 'hermes-agent[acp]'
```

## 🔄 Communication Flow Example

### Complete Interaction Sequence

```
┌─────────────┐                    ┌──────────────┐
│   Client    │                    │  hermes-acp  │
└──────┬──────┘                    └──────┬───────┘
       │                                  │
       │  1. initialize request          │
       │────────────────────────────────>│
       │  2. initialize response         │
       │<────────────────────────────────│
       │                                  │
       │  3. new_session(cwd="/project") │
       │────────────────────────────────>│
       │  4. NewSessionResponse          │
       │<────────────────────────────────│
       │     session_id: "abc123"        │
       │                                  │
       │  5. prompt(session_id, msg)     │
       │────────────────────────────────>│
       │                                  │
       │  6. sessionUpdate (streaming)   │
       │<────────────────────────────────│
       │     "Thinking..."               │
       │                                  │
       │  7. sessionUpdate (more...)     │
       │<────────────────────────────────│
       │     "Analyzing files..."        │
       │                                  │
       │  8. PromptResponse              │
       │<────────────────────────────────│
       │     stop_reason: "end_turn"     │
       │                                  │
```

## 🛠️ Usage in Your Applications

### Minimal Integration (5 lines)

```python
from acp_client import ACPClient

async with ACPClient() as client:
    async with client.new_session(".") as session:
        response = await session.prompt("Hello Hermes!")
        print(response)
```

### With Error Handling

```python
try:
    async with ACPClient() as client:
        async with client.new_session("/path") as session:
            result = await session.prompt(task)
except FileNotFoundError:
    print("hermes-acp not installed!")
except Exception as e:
    print(f"Error: {e}")
```

## 📚 Related Protocols

| Protocol | Purpose | Relationship to ACP |
|----------|---------|---------------------|
| **MCP** (Model Context Protocol) | Tool discovery & invocation | Different focus (tools vs conversations) |
| **JSON-RPC 2.0** | Generic RPC protocol | Transport layer used by ACP |
| **OpenAI Tools** | Function calling | Conceptually similar tool invocations |

## 🐛 Troubleshooting

### Common Issues

1. **"hermes-acp not found"**
   - Install: `pip install 'hermes-agent[acp]'`
   - Verify: `which hermes-acp`

2. **JSON parsing errors**
   - Check Hermes logs (stderr from hermes-acp)
   - Enable verbose mode: `acp-client -v`

3. **Connection closed unexpectedly**
   - Ensure Hermes is not crashing (check stderr)
   - Verify protocol version compatibility

## 📞 Support & Contributing

- **Hermes Issues**: https://github.com/NousResearch/hermes-agent/issues
- **ACP Protocol Issues**: https://github.com/vilukes/agent-client-protocol/issues
- **Discord**: Check Hermes Discord for real-time help
