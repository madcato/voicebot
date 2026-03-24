# Publishing Voicebot Releases

This document explains how to build and publish a new voicebot release so that
users can install it with:

```sh
curl -fsSL https://github.com/OWNER/REPO/releases/latest/download/install.sh | sh
```

---

## What gets published

Each release contains:

| File | Description |
|------|-------------|
| `voicebot-aarch64-apple-darwin.tar.gz` | macOS Apple Silicon (M1/M2/M3) |
| `voicebot-x86_64-apple-darwin.tar.gz` | macOS Intel |
| `voicebot-x86_64-unknown-linux-gnu.tar.gz` | Linux x86\_64 |
| `voicebot-aarch64-unknown-linux-gnu.tar.gz` | Linux ARM64 |
| `*.sha256` | SHA-256 checksums for each tarball |
| `install.sh` | The installer script |

Each tarball contains a single stripped executable named `voicebot`.

### Compiled features per platform

| Platform | Feature flags | TTS backend |
|----------|--------------|-------------|
| macOS | `--features avspeech` | AVSpeechSynthesizer (system, no models needed) |
| Linux | `--features kokoro` | Kokoro via ONNX (models downloaded by installer) |

LLM models and the LLM server (`llama-server` or `mlx_lm.server`) are **never
bundled**. Users are expected to have them installed separately.

---

## Automated release via GitHub Actions

The workflow at `.github/workflows/release.yml` runs automatically whenever you
push a version tag.  It builds all four targets in parallel using native GitHub
runners (no cross-compilation, no Docker).

### Step-by-step

1. **Update the version** in `Cargo.toml`:
   ```toml
   [package]
   version = "1.2.0"
   ```

2. **Commit** the version bump:
   ```sh
   git add Cargo.toml Cargo.lock
   git commit -m "chore: bump version to v1.2.0"
   ```

3. **Tag the commit**.  Use an annotated tag — the tag message becomes the
   release description on GitHub:
   ```sh
   git tag -a v1.2.0 -m "$(cat <<'MSG'
   Short summary of what changed.

   - Fix: describe a bug fix
   - Feature: describe a new feature
   MSG
   )"
   ```

4. **Push tag** to trigger the workflow:
   ```sh
   git push origin v1.2.0
   ```

5. Open the **Actions** tab in your GitHub repository to watch the build
   progress.  Four jobs run in parallel (~10–15 min on GitHub's free runners).

6. Once all jobs succeed, a GitHub Release is created automatically at
   `https://github.com/OWNER/REPO/releases/tag/v1.2.0`.

### Pre-releases

Tags that contain a hyphen (e.g. `v1.2.0-beta.1`, `v1.2.0-rc.1`) are
automatically marked as **pre-release** on GitHub and are **not** served by the
`/releases/latest/download/` URL.  Users must opt in explicitly:

```sh
VOICEBOT_VERSION=v1.2.0-beta.1 \
  curl -fsSL https://github.com/OWNER/REPO/releases/latest/download/install.sh | sh
```

---

## Manual builds (without GitHub Actions)

Use these commands when you need to build locally or on a CI system that is not
GitHub Actions.

### macOS — Apple Silicon

```sh
# Requires: Xcode Command Line Tools (xcode-select --install)
cargo build --release --bin voicebot --features avspeech
strip target/release/voicebot
tar -czf voicebot-aarch64-apple-darwin.tar.gz -C target/release voicebot
```

### macOS — Intel

Same as above, but run on an Intel Mac (or use Rosetta only for local testing —
do **not** publish a Rosetta binary as the ARM64 release):

```sh
cargo build --release --bin voicebot --features avspeech
strip target/release/voicebot
tar -czf voicebot-x86_64-apple-darwin.tar.gz -C target/release voicebot
```

### Linux x86\_64

```sh
# Build-time dependencies
sudo apt-get install -y libasound2-dev espeak-ng pkg-config build-essential cmake

cargo build --release --bin voicebot --features kokoro
strip target/release/voicebot
tar -czf voicebot-x86_64-unknown-linux-gnu.tar.gz -C target/release voicebot
```

### Linux ARM64

Run the same Linux commands on an `aarch64` host (e.g. a Raspberry Pi 4/5,
AWS Graviton, or an OrbStack/QEMU VM):

```sh
sudo apt-get install -y libasound2-dev espeak-ng pkg-config build-essential cmake
cargo build --release --bin voicebot --features kokoro
strip target/release/voicebot
tar -czf voicebot-aarch64-unknown-linux-gnu.tar.gz -C target/release voicebot
```

---

## Build dependencies

### macOS (both architectures)

| Dependency | Source | Notes |
|-----------|--------|-------|
| Xcode CLT | `xcode-select --install` | Provides `cc`, Metal SDK, CoreML framework |
| Rust stable | rustup | 1.80+ recommended |

`whisper-rs` compiles in Metal + CoreML acceleration automatically on macOS.
No extra libraries need to be installed.

### Linux (both architectures)

| Dependency | Package (Debian/Ubuntu) | Purpose |
|-----------|------------------------|---------|
| `libasound2-dev` | `apt-get install libasound2-dev` | CPAL audio I/O (build-time + runtime) |
| `espeak-ng` | `apt-get install espeak-ng` | Kokoro TTS phonemization (runtime) |
| `pkg-config` | `apt-get install pkg-config` | Used by several crates |
| `build-essential` | `apt-get install build-essential` | C toolchain for native crates |
| `cmake` | `apt-get install cmake` | Required by whisper-rs build scripts |
| `libssl-dev` | `apt-get install libssl-dev` | reqwest TLS (build-time) |

#### Runtime dependencies (must be present on the user's machine)

The installer warns users if these are missing, but does not install them:

- **`libasound2`** — ALSA audio library
- **`espeak-ng`** — Kokoro TTS phonemizer

---

## What the installer downloads

When a user runs `install.sh`, the following files are downloaded automatically:

| File | Source | Size | Platform |
|------|--------|------|----------|
| `voicebot` binary | GitHub Releases | ~30–50 MB | both |
| `ggml-large-v3-turbo.bin` | Hugging Face (ggerganov/whisper.cpp) | ~1.6 GB | both |
| `kokoro-v1.0.onnx` | GitHub (thewh1teagle/kokoro-onnx) | ~305 MB | Linux only |
| `voices-v1.0.bin` | GitHub (thewh1teagle/kokoro-onnx) | ~28 MB | Linux only |

On macOS, Kokoro models are not needed — AVSpeechSynthesizer is a system
framework that requires no additional files.

---

## Testing the installer locally

You can test `install.sh` without publishing a release by serving the tarball
locally:

```sh
# 1. Build the binary for your current platform
cargo build --release --bin voicebot --features avspeech   # macOS
# cargo build --release --bin voicebot --features kokoro   # Linux

# 2. Package it
strip target/release/voicebot
tar -czf voicebot-$(rustc -vV | grep host | cut -d' ' -f2).tar.gz \
    -C target/release voicebot

# 3. Serve locally
python3 -m http.server 9000 &

# 4. Run the installer pointing at your local server
GITHUB_REPO="localhost:9000"  \
  VOICEBOT_VERSION=""          \
  sh install.sh
```

Or simply test with the binary already in place by running install.sh directly:

```sh
# Override VOICEBOT_HOME to avoid touching ~/.voicebot during testing
VOICEBOT_HOME=/tmp/voicebot-test sh install.sh
```

---

## Updating the install.sh URL in documentation

After you confirm the first release works, update `readme.md` to use the real
GitHub URL (replace `OWNER/REPO` with the actual repository path):

```sh
sed -i 's|OWNER/REPO|yourname/voicebot|g' install.sh readme.md
```

Also update the `GITHUB_REPO` default at the top of `install.sh`.

---

---

## Publishing to Gitea (tesla.local)

The project has a second workflow and installer for the private Gitea instance.

### Gitea release workflow

File: `.gitea/workflows/release.yml`

Triggered the same way as the GitHub workflow — push a version tag:

```sh
git tag -a v1.2.0 -m "Release notes..."
git push gitea v1.2.0          # push to the Gitea remote
```

Where `gitea` is a remote pointing at:
```
ssh://git@tesla.local:222/danielvela/voicebot.git
```

Add it if not already configured:
```sh
git remote add gitea ssh://git@tesla.local:222/danielvela/voicebot.git
```

### Required act_runner labels

Register runners on the machines below.  Get the registration token from
`http://tesla.local:3000/danielvela/voicebot/settings/runners`.

| Label | Machine | Notes |
|-------|---------|-------|
| `ubuntu-latest` | Linux x86\_64 host (e.g. tesla.local) | builds Linux x86\_64 + release upload |
| `linux-arm64` | Linux aarch64 host (Pi 5, Graviton…) | builds Linux ARM64 |
| `macos-latest` | Your Mac | builds both macOS targets |

Register a runner (run on each machine):
```sh
# Download act_runner from http://tesla.local:3000/-/admin/runners
act_runner register \
  --instance http://tesla.local:3000 \
  --token    <token-from-gitea-settings> \
  --labels   ubuntu-latest,linux    # adjust per machine
```

For the macOS runner:
```sh
act_runner register \
  --instance http://tesla.local:3000 \
  --token    <token> \
  --labels   macos-latest,macos
```

### Gitea installer usage

```sh
# Latest release
curl -fsSL http://tesla.local:3000/danielvela/voicebot/releases/download/latest/install-gitea.sh | sh

# Pin a version
VOICEBOT_VERSION=v1.2.0 \
  curl -fsSL http://tesla.local:3000/danielvela/voicebot/releases/download/v1.2.0/install-gitea.sh | sh
```

> **Note:** `tesla.local` is an mDNS hostname — the machine running the installer
> must be on the same local network.  mDNS is available by default on macOS and
> on Linux via `avahi-daemon` (`sudo apt-get install avahi-daemon`).

### Difference between install.sh and install-gitea.sh

| | `install.sh` | `install-gitea.sh` |
|-|------------|------------------|
| Binary source | GitHub Releases | `http://tesla.local:3000` Gitea Releases |
| Latest version | GitHub API (`/releases/latest`) | Gitea API (`/api/v1/repos/.../releases/latest`) |
| Whisper model | Hugging Face (public) | Hugging Face (public) |
| Kokoro models | GitHub (public) | GitHub (public) |
| Network access | Internet | Local network only |

---

## Checklist for a new release

- [ ] Version bumped in `Cargo.toml`
- [ ] `Cargo.lock` updated (`cargo build` or `cargo update`)
- [ ] All tests pass: `cargo test`
- [ ] `readme.md` updated for new features / config vars
- [ ] `GITHUB_REPO` at the top of `install.sh` points to the real repo
- [ ] Annotated tag created: `git tag -a vX.Y.Z -m "..."`
- [ ] Tag pushed to GitHub: `git push origin vX.Y.Z`
- [ ] Tag pushed to Gitea: `git push gitea vX.Y.Z`
- [ ] GitHub Actions build passes (check the Actions tab)
- [ ] Gitea Actions build passes (`http://tesla.local:3000/danielvela/voicebot/actions`)
- [ ] Test `install.sh` on at least one macOS and one Linux machine
- [ ] Test `install-gitea.sh` on at least one machine on the local network
