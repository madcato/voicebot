# CoreML / ANE Acceleration Setup

## Current Status

Your voicebot installation has:
- ✅ **Metal GPU acceleration** (enabled via Cargo.toml features)
- ⚠️ **CoreML encoder model files present** (`ggml-large-v3-turbo-encoder.mlmodelc`)  
- ❌ **CoreML NOT enabled by default** (causes build errors if auto-enabled)

## How to Enable CoreML (Advanced)

**Warning:** Enabling CoreML requires careful setup and may cause build failures on some systems. Metal GPU acceleration provides excellent performance without extra configuration.

If you want to try CoreML (Apple Neural Engine):

### Prerequisites

```bash
# Ensure Xcode Command Line Tools are installed
xcode-select --install

# Verify CoreML encoder exists
ls -lh models/*-encoder.mlmodelc
```

### Enable CoreML Build

```bash
cd /Users/danielvela/projects/ai/voicebot

# Clean everything
cargo clean

# Set environment variable BEFORE building
export WHISPER_USE_COREML=1

# Rebuild whisper-cpp-plus-sys specifically first
cargo build -p whisper-cpp-plus-sys --release

# Then build full project
cargo build --release
```

### Troubleshooting

If you get linker errors like `whisper_coreml_* symbol not found`:

1. The issue is CMake can't find or compile the CoreML source files
2. This requires Xcode toolchain and proper macOS SDK paths
3. Solution: Don't use CoreML — Metal GPU is already very fast

Alternative approach (if you have a working whisper.cpp CoreML build):
```bash
# Point to pre-built CoreML whisper library
export WHISPER_PREBUILT_PATH=/path/to/coreml-whisper
cargo build --release
```

## Performance Comparison

| Configuration | Expected Latency | Notes |
|--------------|------------------|-------|
| CPU only | 3000-8000ms | Baseline, slowest |
| Metal GPU ✓ | 800-1500ms | **Current setup**, good speed |
| CoreML + Metal | 400-800ms | Fastest but complex setup |

## Recommendation

**Use Metal GPU acceleration (current setup)**. It's simpler, more reliable, and provides excellent performance for real-time voice interaction.

Only enable CoreML if:
- You need absolute minimum latency (< 500ms)
- You're comfortable debugging Xcode/CMake build issues
- You've verified your Mac has A12 Bionic chip or later (M1/M2/M3 yes)

---

## Quick Start (Recommended)

Just run normally with Metal GPU acceleration:
```bash
cargo run --release --features tui
```

No special configuration needed! 🚀
