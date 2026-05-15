# Apple Watch Client for Voicebot

Guide for building a watchOS app that connects to the Voicebot WebSocket server, streams microphone audio, and plays back TTS responses.

## Prerequisites

- Xcode 15+ with watchOS 10+ SDK
- Voicebot running with `--features remote` and `WS_PORT` set
- Apple Watch paired with iPhone (or Simulator)

## Project Setup

1. Create a new watchOS App project in Xcode (SwiftUI lifecycle)
2. Set deployment target to **watchOS 10.0** minimum (`URLSessionWebSocketTask` is stable from watchOS 6, but audio improvements landed in watchOS 10)
3. No third-party dependencies needed -- Foundation and AVFAudio are sufficient

### Info.plist / Capabilities

- Add `NSMicrophoneUsageDescription` to Info.plist: `"Voicebot needs microphone access to hear your voice."`
- Enable **Background Modes** capability with **Audio, AirPlay, and Picture in Picture** checked

## Wire Protocol

The voicebot WebSocket server expects:

| Direction | Frame type | Format |
|-----------|-----------|--------|
| Watch -> Server | Binary | PCM i16 little-endian, mono, 16 kHz |
| Server -> Watch | Binary | PCM i16 little-endian, mono, 16 kHz |
| Watch -> Server | Text | JSON control messages |
| Server -> Watch | Text | JSON control messages |

### Control Messages

```
Watch -> Server:
  {"type": "session.start", "sample_rate": 16000}
  {"type": "barge_in"}

Server -> Watch:
  {"type": "session.ready"}
  {"type": "transcript", "text": "..."}
  {"type": "response.text", "text": "..."}
  {"type": "response.end"}
  {"type": "audio.start"}
  {"type": "audio.end"}
  {"type": "error", "message": "..."}
```

## Architecture

```
┌─────────────────────────────────────┐
│           Apple Watch App           │
│                                     │
│  ┌──────────┐    ┌───────────────┐  │
│  │ AudioMgr │───>│ WebSocketMgr  │──────> voicebot:WS_PORT/ws
│  │ (capture)│    │  (send audio) │  │
│  └──────────┘    └───────────────┘  │
│                                     │
│  ┌──────────┐    ┌───────────────┐  │
│  │ AudioMgr │<───│ WebSocketMgr  │<────── voicebot TTS audio
│  │ (play)   │    │ (recv audio)  │  │
│  └──────────┘    └───────────────┘  │
└─────────────────────────────────────┘
```

Three classes:
- **`ContentView`** -- SwiftUI view with a talk button
- **`AudioManager`** -- owns `AVAudioEngine`, handles mic capture + speaker playback
- **`WebSocketManager`** -- owns `URLSessionWebSocketTask`, handles send/receive

## Audio Capture

Use `AVAudioEngine` to tap the microphone at 16 kHz mono Int16 -- this matches the voicebot wire format exactly, so no conversion is needed.

```swift
import AVFAudio

class AudioManager {
    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()

    // 16kHz mono Int16 -- matches voicebot wire protocol
    private let captureFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: 16000,
        channels: 1,
        interleaved: true
    )!

    func startCapture(onAudio: @escaping (Data) -> Void) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, options: [.defaultToSpeaker])
        try session.setActive(true)

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Convert from device format to 16kHz Int16
        guard let converter = AVAudioConverter(from: inputFormat, to: captureFormat) else {
            throw AudioError.converterFailed
        }

        // 100ms chunks at 16kHz = 1600 samples
        let bufferSize: AVAudioFrameCount = AVAudioFrameCount(inputFormat.sampleRate * 0.1)

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) {
            [captureFormat] buffer, _ in
            // Convert to 16kHz Int16
            let frameCapacity: AVAudioFrameCount = AVAudioFrameCount(
                Double(buffer.frameLength) * 16000.0 / inputFormat.sampleRate
            )
            guard let converted = AVAudioPCMBuffer(
                pcmFormat: captureFormat,
                frameCapacity: frameCapacity
            ) else { return }

            var error: NSError?
            converter.convert(to: converted, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }

            if let error { return }

            // Extract raw bytes (i16 LE on Apple silicon)
            guard let int16Data = converted.int16ChannelData else { return }
            let byteCount = Int(converted.frameLength) * 2
            let data = Data(bytes: int16Data[0], count: byteCount)
            onAudio(data)
        }

        engine.prepare()
        try engine.start()
    }

    func stopCapture() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}
```

## WebSocket Connection

```swift
class WebSocketManager: NSObject {
    private var task: URLSessionWebSocketTask?
    var onAudioReceived: ((Data) -> Void)?
    var onTranscript: ((String) -> Void)?
    var onResponseText: ((String) -> Void)?

    func connect(to url: URL) {
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        task = session.webSocketTask(with: url)
        task?.resume()

        // Send session.start
        let startMsg = #"{"type": "session.start", "sample_rate": 16000}"#
        task?.send(.string(startMsg)) { error in
            if let error { print("Send error: \(error)") }
        }

        receiveLoop()
    }

    func sendAudio(_ data: Data) {
        task?.send(.data(data)) { _ in }
    }

    func sendBargeIn() {
        let msg = #"{"type": "barge_in"}"#
        task?.send(.string(msg)) { _ in }
    }

    func disconnect() {
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
    }

    private func receiveLoop() {
        task?.receive { [weak self] result in
            switch result {
            case .success(.data(let data)):
                // Binary frame = TTS audio (i16 LE mono 16kHz)
                self?.onAudioReceived?(data)
            case .success(.string(let text)):
                self?.handleControlMessage(text)
            case .failure(let error):
                print("WS receive error: \(error)")
                return // Stop loop on error
            default:
                break
            }
            // Continue receiving
            self?.receiveLoop()
        }
    }

    private func handleControlMessage(_ json: String) {
        guard let data = json.data(using: .utf8),
              let msg = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = msg["type"] as? String else { return }

        switch type {
        case "session.ready":
            print("Session ready")
        case "transcript":
            if let text = msg["text"] as? String {
                onTranscript?(text)
            }
        case "response.text":
            if let text = msg["text"] as? String {
                onResponseText?(text)
            }
        case "audio.start":
            break // Audio frames incoming
        case "audio.end":
            break // Audio for this sentence done
        case "error":
            print("Server error: \(msg["message"] ?? "unknown")")
        default:
            break
        }
    }
}
```

## Audio Playback

Receive i16 LE binary frames from the WebSocket and play them through `AVAudioPlayerNode`.

```swift
extension AudioManager {
    private static let playbackFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: 16000,
        channels: 1,
        interleaved: true
    )!

    func setupPlayback() {
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: Self.playbackFormat)
    }

    func playAudio(_ data: Data) {
        let frameCount = AVAudioFrameCount(data.count / 2) // 2 bytes per Int16 sample
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: Self.playbackFormat,
            frameCapacity: frameCount
        ) else { return }

        buffer.frameLength = frameCount

        // Copy i16 bytes into the buffer
        data.withUnsafeBytes { rawPtr in
            guard let src = rawPtr.baseAddress else { return }
            memcpy(buffer.int16ChannelData![0], src, data.count)
        }

        if !playerNode.isPlaying {
            playerNode.play()
        }
        playerNode.scheduleBuffer(buffer)
    }

    func stopPlayback() {
        playerNode.stop()
    }
}
```

## Barge-in

Two options:

**Option A: Server-side VAD (recommended)**
Just keep streaming microphone audio to the server. The voicebot's VAD will detect when the user starts speaking and automatically cancel the current TTS playback. No extra work needed on the watch.

**Option B: Client-side barge-in**
If you want faster response, detect audio input locally and send an explicit barge-in signal:

```swift
// When user starts speaking while TTS is playing:
webSocketManager.sendBargeIn()
audioManager.stopPlayback()
```

## Minimal SwiftUI View

```swift
import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = VoiceViewModel()

    var body: some View {
        VStack(spacing: 16) {
            Text(viewModel.statusText)
                .font(.caption)

            if let transcript = viewModel.lastTranscript {
                Text(transcript)
                    .font(.footnote)
                    .foregroundColor(.secondary)
            }

            Button(action: { viewModel.toggleListening() }) {
                Image(systemName: viewModel.isListening ? "mic.fill" : "mic")
                    .font(.title)
                    .foregroundColor(viewModel.isListening ? .red : .blue)
            }
            .buttonStyle(.plain)
        }
        .onAppear { viewModel.connect() }
        .onDisappear { viewModel.disconnect() }
    }
}
```

## watchOS Considerations

### Battery
- 16 kHz mono Int16 = ~32 KB/s upstream + downstream = ~64 KB/s total
- Manageable over Bluetooth relay or WiFi
- Avoid keeping the microphone open when not needed (use push-to-talk or VAD)

### Extended Runtime
For conversations longer than ~30 seconds, use `WKExtendedRuntimeSession`:

```swift
let session = WKExtendedRuntimeSession()
session.start()
// ... conversation ...
session.invalidate()
```

### Network
- Watch on WiFi: direct connection to voicebot server
- Watch on Bluetooth only: routes through paired iPhone
- Cellular models: can connect directly if on cellular data
- Ensure the voicebot server is reachable from the watch's network

### Audio Session
Always configure before starting audio:

```swift
let session = AVAudioSession.sharedInstance()
try session.setCategory(.playAndRecord, options: [.defaultToSpeaker, .allowBluetooth])
try session.setActive(true)
```

## Testing

1. Start the voicebot with remote support:
   ```bash
   WS_PORT=9090 cargo run --features remote --release
   ```

2. In the watch app, connect to `ws://<your-mac-ip>:9090/ws`

3. Tap the mic button and speak -- you should see the transcript appear and hear the TTS response

4. Test barge-in by speaking while the assistant is responding
