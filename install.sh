#!/bin/sh
# Voicebot installer
# Usage: curl -fsSL https://github.com/OWNER/REPO/releases/latest/download/install.sh | sh
#
# Environment overrides:
#   GITHUB_REPO    — GitHub owner/repo (default: set at release time)
#   VOICEBOT_HOME  — where models/data/config live (default: ~/.voicebot)
#   BIN_DIR        — where to place the `voicebot` launcher (default: ~/.local/bin)
#   VOICEBOT_VERSION — pin a release tag, e.g. v1.2.0 (default: latest)
set -e

# ── Configurable defaults ─────────────────────────────────────────────────────
GITHUB_REPO="${GITHUB_REPO:-OWNER/voicebot}"
VOICEBOT_HOME="${VOICEBOT_HOME:-$HOME/.voicebot}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
VOICEBOT_VERSION="${VOICEBOT_VERSION:-latest}"

# Model download URLs
WHISPER_MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
KOKORO_MODEL_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# ── Output helpers ────────────────────────────────────────────────────────────
# Use ANSI colors only when stdout is a terminal (not when piped)
if [ -t 1 ]; then
    _GREEN='\033[0;32m'; _YELLOW='\033[1;33m'; _RED='\033[0;31m'; _NC='\033[0m'
else
    _GREEN=''; _YELLOW=''; _RED=''; _NC=''
fi

info()  { printf "${_GREEN}[voicebot]${_NC} %s\n" "$1"; }
warn()  { printf "${_YELLOW}[voicebot]${_NC} %s\n" "$1"; }
error() { printf "${_RED}[voicebot] ERROR:${_NC} %s\n" "$1" >&2; exit 1; }
step()  { printf "\n${_GREEN}▶ %s${_NC}\n" "$1"; }

# ── Platform detection ────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Darwin) OS_NAME="macOS" ;;
    Linux)  OS_NAME="Linux" ;;
    *)      error "Unsupported operating system: $OS. Only macOS and Linux are supported." ;;
esac

case "$ARCH" in
    x86_64)          ARCH_TRIPLE="x86_64" ;;
    arm64 | aarch64) ARCH_TRIPLE="aarch64" ;;
    *)               error "Unsupported architecture: $ARCH. Only x86_64 and arm64/aarch64 are supported." ;;
esac

case "$OS" in
    Darwin) PLATFORM="apple-darwin" ;;
    Linux)  PLATFORM="unknown-linux-gnu" ;;
esac

TARGET="${ARCH_TRIPLE}-${PLATFORM}"
TARBALL="voicebot-${TARGET}.tar.gz"

if [ "$VOICEBOT_VERSION" = "latest" ]; then
    RELEASE_BASE="https://github.com/${GITHUB_REPO}/releases/latest/download"
else
    RELEASE_BASE="https://github.com/${GITHUB_REPO}/releases/download/${VOICEBOT_VERSION}"
fi
BINARY_URL="${RELEASE_BASE}/${TARBALL}"

# Derived paths
VOICEBOT_BIN_DIR="$VOICEBOT_HOME/bin"
VOICEBOT_MODELS_DIR="$VOICEBOT_HOME/models"
VOICEBOT_DATA_DIR="$VOICEBOT_HOME/data"
VOICEBOT_ENV="$VOICEBOT_HOME/.env"

# ── Utility: download a file ──────────────────────────────────────────────────
download() {
    local url="$1" dest="$2" label="$3"
    info "  Downloading: $label"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --progress-bar -o "$dest" "$url" 2>&1 || {
            rm -f "$dest"
            error "Download failed: $url"
        }
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$dest" "$url" 2>&1 || {
            rm -f "$dest"
            error "Download failed: $url"
        }
    else
        error "Neither curl nor wget found. Please install one and re-run."
    fi
}

# ── Step 1: System dependency check ──────────────────────────────────────────
check_dependencies() {
    step "Checking system dependencies"

    # Both platforms need a working C runtime — already guaranteed if we got here.

    if [ "$OS" = "Linux" ]; then
        local missing=""

        # ALSA — required by CPAL for audio I/O
        if ! ldconfig -p 2>/dev/null | grep -q "libasound\.so" && \
           ! find /usr/lib /usr/local/lib 2>/dev/null | grep -q "libasound"; then
            missing="$missing libasound2"
        fi

        # espeak-ng — required by Kokoro TTS for phonemization
        if ! command -v espeak-ng >/dev/null 2>&1; then
            missing="$missing espeak-ng"
        fi

        if [ -n "$missing" ]; then
            warn "The following runtime dependencies are missing:$missing"
            warn ""
            warn "Install them before running voicebot:"
            warn "  Debian/Ubuntu:  sudo apt-get install -y$missing"
            warn "  Fedora/RHEL:    sudo dnf install -y$missing"
            warn "  Arch Linux:     sudo pacman -S$missing"
            warn ""
            warn "Installation will continue, but voicebot may fail to start."
        else
            info "  All Linux runtime dependencies found."
        fi
    fi

    if [ "$OS" = "Darwin" ]; then
        # macOS: AVSpeechSynthesizer is a system framework — no extra deps.
        # Microphone permission must be granted at first run (macOS will prompt).
        info "  macOS detected — TTS: AVSpeechSynthesizer (built-in)."
        info "  Microphone access will be requested on first run."
    fi
}

# ── Step 2: Create directory layout ──────────────────────────────────────────
setup_directories() {
    step "Setting up directories"
    mkdir -p "$VOICEBOT_BIN_DIR"
    mkdir -p "$VOICEBOT_MODELS_DIR"
    mkdir -p "$VOICEBOT_DATA_DIR"
    mkdir -p "$BIN_DIR"
    info "  Install home : $VOICEBOT_HOME"
    info "  Launcher dir : $BIN_DIR"
}

# ── Step 3: Download and install the pre-compiled binary ──────────────────────
install_binary() {
    step "Downloading voicebot binary ($TARGET)"

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    # Ensure temp dir is cleaned up even on error
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp_dir'" EXIT

    download "$BINARY_URL" "$tmp_dir/$TARBALL" "voicebot ($TARGET)"

    info "  Extracting binary..."
    tar -xzf "$tmp_dir/$TARBALL" -C "$tmp_dir"

    if [ ! -f "$tmp_dir/voicebot" ]; then
        error "Binary not found inside tarball. Expected: voicebot"
    fi

    mv "$tmp_dir/voicebot" "$VOICEBOT_BIN_DIR/voicebot"
    chmod +x "$VOICEBOT_BIN_DIR/voicebot"
    info "  Binary installed: $VOICEBOT_BIN_DIR/voicebot"
}

# ── Step 4: Download Whisper STT model ───────────────────────────────────────
install_whisper_model() {
    step "Installing Whisper STT model"
    local dest="$VOICEBOT_MODELS_DIR/ggml-large-v3-turbo.bin"

    if [ -f "$dest" ]; then
        info "  Already present — skipping (delete to re-download)."
        return
    fi

    warn "  Downloading ggml-large-v3-turbo.bin (~1.6 GB). This may take several minutes..."
    download "$WHISPER_MODEL_URL" "$dest" "Whisper large-v3-turbo"
    info "  Whisper model installed."
}

# ── Step 5: Download Kokoro TTS models (Linux only) ──────────────────────────
install_kokoro_models() {
    step "Installing Kokoro TTS models (Linux)"

    local kokoro_model="$VOICEBOT_MODELS_DIR/kokoro-v1.0.onnx"
    local kokoro_voices="$VOICEBOT_MODELS_DIR/voices-v1.0.bin"

    if [ -f "$kokoro_model" ]; then
        info "  kokoro-v1.0.onnx already present — skipping."
    else
        warn "  Downloading kokoro-v1.0.onnx (~305 MB)..."
        download "$KOKORO_MODEL_URL" "$kokoro_model" "Kokoro ONNX model"
        info "  Kokoro model installed."
    fi

    if [ -f "$kokoro_voices" ]; then
        info "  voices-v1.0.bin already present — skipping."
    else
        warn "  Downloading voices-v1.0.bin (~28 MB)..."
        download "$KOKORO_VOICES_URL" "$kokoro_voices" "Kokoro voice embeddings"
        info "  Kokoro voices installed."
    fi
}

# ── Step 6: Write default .env if absent ─────────────────────────────────────
create_default_env() {
    step "Writing default configuration"

    if [ -f "$VOICEBOT_ENV" ]; then
        info "  Config already exists at $VOICEBOT_ENV — skipping."
        return
    fi

    # TTS settings differ per platform
    if [ "$OS" = "Darwin" ]; then
        TTS_PROVIDER_DEFAULT="avspeech"
        TTS_VOICE_LINE="AVSPEECH_VOICE=Marisol (Enhanced)"
        TTS_RATE_LINE="AVSPEECH_RATE=0.55"
    else
        TTS_PROVIDER_DEFAULT="kokoro"
        TTS_VOICE_LINE="KOKORO_VOICE=es_xb"
        TTS_RATE_LINE="KOKORO_LANGUAGE=es"
    fi

    cat > "$VOICEBOT_ENV" << ENVEOF
# ── Voicebot configuration ────────────────────────────────────────────────────
# Edit this file to customize your setup.
# Full list of options: see .env.example in the source repo.

# Language: es (Spanish) or en (English)
VOICEBOT_LANGUAGE=es

# ── LLM server ────────────────────────────────────────────────────────────────
# Start with: mlx_lm.server --model mlx-community/Qwen3-8B-4bit --port 8000
#         or: omlx serve --model-dir ~/models --port 8001
LLM_URL=http://localhost:8000
LLM_MAX_TOKENS=400
LLM_TEMPERATURE=0.3
# LLM_SYSTEM_PROMPT=You are a helpful voice assistant.
# LLM_MODEL=local-model

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_PROVIDER=$TTS_PROVIDER_DEFAULT
$TTS_VOICE_LINE
$TTS_RATE_LINE

# ── Audio devices ─────────────────────────────────────────────────────────────
# Uncomment and set to a substring of your device name.
# Run: voicebot --list-devices  to see available devices.
# AUDIO_INPUT_DEVICE=
# AUDIO_OUTPUT_DEVICE=
ENVEOF

    info "  Config written: $VOICEBOT_ENV"
}

# ── Step 7: Install the launcher wrapper script ───────────────────────────────
install_launcher() {
    step "Installing launcher"

    local launcher="$BIN_DIR/voicebot"

    # Determine default TTS provider for this platform
    if [ "$OS" = "Darwin" ]; then
        DEFAULT_TTS="avspeech"
    else
        DEFAULT_TTS="kokoro"
    fi

    cat > "$launcher" << LAUNCHEOF
#!/bin/sh
# voicebot launcher — generated by install.sh
# Edit $VOICEBOT_ENV to configure your setup.
VOICEBOT_HOME="\${VOICEBOT_HOME:-$VOICEBOT_HOME}"

# Point to installed models (can be overridden by env or .env file)
export WHISPER_MODEL="\${WHISPER_MODEL:-\$VOICEBOT_HOME/models/ggml-large-v3-turbo.bin}"
export DB_PATH="\${DB_PATH:-\$VOICEBOT_HOME/data/voicebot.db}"
export KOKORO_MODEL="\${KOKORO_MODEL:-\$VOICEBOT_HOME/models/kokoro-v1.0.onnx}"
export KOKORO_VOICES="\${KOKORO_VOICES:-\$VOICEBOT_HOME/models/voices-v1.0.bin}"
export TTS_PROVIDER="\${TTS_PROVIDER:-$DEFAULT_TTS}"

# Load user configuration (values here override defaults above)
if [ -f "\$VOICEBOT_HOME/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "\$VOICEBOT_HOME/.env"
    set +a
fi

exec "\$VOICEBOT_HOME/bin/voicebot" "\$@"
LAUNCHEOF

    chmod +x "$launcher"
    info "  Launcher installed: $launcher"
}

# ── Step 8: PATH check ────────────────────────────────────────────────────────
check_path() {
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *)
            warn ""
            warn "  $BIN_DIR is not in your PATH."
            warn "  Add this line to your shell config (~/.bashrc, ~/.zshrc, etc.):"
            warn ""
            warn "    export PATH=\"\$HOME/.local/bin:\$PATH\""
            warn ""
            warn "  Then reload your shell:  source ~/.bashrc  (or restart your terminal)"
            ;;
    esac
}

# ── Main ──────────────────────────────────────────────────────────────────────
printf "\n"
info "╔══════════════════════════════════════════════╗"
info "║          Voicebot Installer                  ║"
info "╚══════════════════════════════════════════════╝"
printf "\n"
info "Platform : $OS_NAME ($TARGET)"
info "Home     : $VOICEBOT_HOME"
info "Launcher : $BIN_DIR/voicebot"
printf "\n"

check_dependencies
setup_directories
install_binary
install_whisper_model

if [ "$OS" = "Linux" ]; then
    install_kokoro_models
fi

create_default_env
install_launcher
check_path

printf "\n"
info "══════════════════════════════════════════════════"
info "  Installation complete!"
info "══════════════════════════════════════════════════"
printf "\n"
info "Before starting voicebot, make sure your LLM server is running."
info ""
info "  mlx-lm:  mlx_lm.server --model mlx-community/Qwen3-8B-4bit --port 8000"
info "  omlx:    omlx serve --model-dir ~/models --port 8001"
printf "\n"
info "Configure voicebot:"
info "  \$EDITOR $VOICEBOT_ENV"
printf "\n"
info "Then start:"
info "  voicebot"
printf "\n"
info "List audio devices:"
info "  voicebot --list-devices"
printf "\n"
