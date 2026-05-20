#!/bin/bash
# Voicebot installer smoke-test harness
#
# Tests install.sh and install-gitea.sh without real network access by
# placing a mock curl on PATH that serves local fixture files.
#
# Usage:
#   bash scripts/test-installers.sh          # normal (exit 0)
#   SIMULATE_MISSING_VAD=1 bash scripts/test-installers.sh   # expected failure
set -e

# ── Setup ───────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DIR=$(mktemp -d)
FIXTURE_DIR="$TEST_DIR/fixtures"
MOCK_BIN="$TEST_DIR/mock-bin"

cleanup() { rm -rf "$TEST_DIR"; }
trap cleanup EXIT

mkdir -p "$FIXTURE_DIR" "$MOCK_BIN"

echo "=== Setting up test fixtures ==="

# Stub voicebot binary + tarball (what install.sh / install-gitea.sh expect)
cat > "$FIXTURE_DIR/voicebot" << 'EOF'
#!/bin/sh
echo "Voicebot stub"
EOF
chmod +x "$FIXTURE_DIR/voicebot"
tar -czf "$FIXTURE_DIR/voicebot-stub.tar.gz" -C "$FIXTURE_DIR" voicebot

# Stub model files (tiny placeholders)
echo "STUB_WHISPER" > "$FIXTURE_DIR/ggml-large-v3-turbo.bin"
echo "STUB_VAD"     > "$FIXTURE_DIR/silero_vad.onnx"
echo "STUB_KOKORO"  > "$FIXTURE_DIR/kokoro-v1.0.onnx"
echo "STUB_VOICES"  > "$FIXTURE_DIR/voices-v1.0.bin"

# ── Mock curl ───────────────────────────────────────────────────────────────
# Both installers call `curl` internally.  We intercept every call and serve
# fixture files instead of hitting real networks.
cat > "$MOCK_BIN/curl" << 'MOCKEOF'
#!/bin/bash
MOCK_FIXTURE_DIR="${MOCK_FIXTURE_DIR:?}"
OUT_FILE=""
URL=""

while [ $# -gt 0 ]; do
    case "$1" in
        -o)
            shift
            OUT_FILE="$1"
            ;;
        --progress-bar|-fsSL|-S|-s|-f|-L|-k)
            # Flags consumed silently
            ;;
        *)
            URL="$1"
            ;;
    esac
    shift
done

[ -n "$URL" ] || { echo "Mock curl: no URL" >&2; exit 1; }

# ── API call (no -o flag → output to stdout) ────────────────────────────────
if [ -z "$OUT_FILE" ]; then
    case "$URL" in
        */api/v1/repos/*/releases/latest*)
            # For install-gitea.sh resolve_version()
            echo '{"tag_name":"v0.0.0-test"}'
            exit 0
            ;;
        *)
            echo "Mock curl: unknown API call: $URL" >&2
            exit 1
            ;;
    esac
fi

# ── File download — serve from fixtures based on URL pattern ────────────────
case "$URL" in
    *voicebot-*.tar.gz*)
        cp "$MOCK_FIXTURE_DIR/voicebot-stub.tar.gz" "$OUT_FILE" ;;
    *ggml-large-v3-turbo.bin*)
        cp "$MOCK_FIXTURE_DIR/ggml-large-v3-turbo.bin" "$OUT_FILE" ;;
    *silero_vad.onnx*)
        cp "$MOCK_FIXTURE_DIR/silero_vad.onnx" "$OUT_FILE" ;;
    *kokoro-v1.0.onnx*)
        cp "$MOCK_FIXTURE_DIR/kokoro-v1.0.onnx" "$OUT_FILE" ;;
    *voices-v1.0.bin*)
        cp "$MOCK_FIXTURE_DIR/voices-v1.0.bin" "$OUT_FILE" ;;
    *)
        echo "Mock curl: unknown download URL: $URL" >&2
        exit 1
        ;;
esac

if [ -f "$OUT_FILE" ]; then
    exit 0
else
    echo "Mock curl: failed to serve: $URL" >&2
    exit 1
fi
MOCKEOF
chmod +x "$MOCK_BIN/curl"

export MOCK_FIXTURE_DIR="$FIXTURE_DIR"
export PATH="$MOCK_BIN:$PATH"

# ── Helper: verify installed files ──────────────────────────────────────────
check_file() {
    if [ -f "$1" ]; then
        echo "  [+] $2"
        return 0
    else
        echo "  [!!] $2 — MISSING"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — install.sh
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════"
echo "  Test 1: install.sh (GitHub release installer)"
echo "═══════════════════════════════════════════════"
echo ""

INSTALL_DIR="$TEST_DIR/install-test"

VOICEBOT_HOME="$INSTALL_DIR"                      \
BIN_DIR="$INSTALL_DIR/launcher"                   \
GITHUB_REPO="localhost:9876"                      \
VOICEBOT_VERSION=""                               \
WHISPER_MODEL_URL="http://localhost:9876/ggml-large-v3-turbo.bin" \
VAD_MODEL_URL="http://localhost:9876/silero_vad.onnx"           \
KOKORO_MODEL_URL="http://localhost:9876/kokoro-v1.0.onnx"       \
KOKORO_VOICES_URL="http://localhost:9876/voices-v1.0.bin"       \
sh "$PROJECT_ROOT/install.sh"

echo ""
echo "--- Verifying install.sh ---"
ERR=0
check_file "$INSTALL_DIR/bin/voicebot"                 "Binary installed"        || ERR=1
check_file "$INSTALL_DIR/models/ggml-large-v3-turbo.bin" "Whisper model"        || ERR=1
check_file "$INSTALL_DIR/models/ggml-silero-vad.bin"   "VAD model"              || ERR=1
check_file "$INSTALL_DIR/.env"                         "Default config"         || ERR=1

if [ "$(uname -s)" = "Linux" ]; then
    check_file "$INSTALL_DIR/models/kokoro-v1.0.onnx"  "Kokoro model (Linux)"   || ERR=1
    check_file "$INSTALL_DIR/models/voices-v1.0.bin"   "Kokoro voices (Linux)"  || ERR=1
fi

if [ "$ERR" = "1" ]; then
    echo ""
    echo "FAILED: install.sh test — missing files"
    exit 1
fi
echo ""
echo "  [+] install.sh test passed"

# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — install-gitea.sh
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════"
echo "  Test 2: install-gitea.sh (Gitea release installer)"
echo "═══════════════════════════════════════════════"
echo ""

GITEA_INSTALL_DIR="$TEST_DIR/install-gitea-test"

VOICEBOT_HOME="$GITEA_INSTALL_DIR"                \
BIN_DIR="$GITEA_INSTALL_DIR/launcher"             \
GITEA_URL="http://localhost:9876"                  \
GITEA_REPO="danielvela/voicebot"                   \
WHISPER_MODEL_URL="http://localhost:9876/ggml-large-v3-turbo.bin" \
VAD_MODEL_URL="http://localhost:9876/silero_vad.onnx"           \
KOKORO_MODEL_URL="http://localhost:9876/kokoro-v1.0.onnx"       \
KOKORO_VOICES_URL="http://localhost:9876/voices-v1.0.bin"       \
sh "$PROJECT_ROOT/install-gitea.sh"

echo ""
echo "--- Verifying install-gitea.sh ---"
ERR=0
check_file "$GITEA_INSTALL_DIR/bin/voicebot"                 "Binary installed"        || ERR=1
check_file "$GITEA_INSTALL_DIR/models/ggml-large-v3-turbo.bin" "Whisper model"        || ERR=1
check_file "$GITEA_INSTALL_DIR/models/ggml-silero-vad.bin"   "VAD model"              || ERR=1
check_file "$GITEA_INSTALL_DIR/.env"                         "Default config"         || ERR=1

if [ "$(uname -s)" = "Linux" ]; then
    check_file "$GITEA_INSTALL_DIR/models/kokoro-v1.0.onnx"  "Kokoro model (Linux)"   || ERR=1
    check_file "$GITEA_INSTALL_DIR/models/voices-v1.0.bin"   "Kokoro voices (Linux)"  || ERR=1
fi

if [ "$ERR" = "1" ]; then
    echo ""
    echo "FAILED: install-gitea.sh test — missing files"
    exit 1
fi
echo ""
echo "  [+] install-gitea.sh test passed"

# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — SIMULATE_MISSING_VAD (expected failure)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SIMULATE_MISSING_VAD" = "1" ]; then
    echo ""
    echo "═══════════════════════════════════════════════"
    echo "  Test 3: SIMULATE_MISSING_VAD"
    echo "═══════════════════════════════════════════════"
    echo ""

    echo "  Removing VAD model to simulate missing asset..."
    rm -f "$INSTALL_DIR/models/ggml-silero-vad.bin"

    if [ ! -f "$INSTALL_DIR/models/ggml-silero-vad.bin" ]; then
        echo ""
        echo "  VAD model missing as expected"
        echo "  Test 3 FAILED intentionally — missing VAD detected correctly."
        exit 1
    fi
fi

# ── Success ──────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  All installer smoke tests passed"
echo "═══════════════════════════════════════════════"
exit 0
