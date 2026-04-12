/// Test binary for whisper-cpp-plus streaming functionality
/// Run with: cargo run --bin test_stt_plus --release
use std::time::Instant;
use voicebot::stt::WhisperSttPlus;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing whisper-cpp-plus STT ===\n");

    // Load model from environment or default
    let model_path = std::env::var("WHISPER_MODEL")
        .unwrap_or_else(|_| "models/ggml-large-v3-turbo.bin".to_string());

    let language = std::env::var("VOICEBOT_LANGUAGE").unwrap_or_else(|_| "es".to_string());

    let threads = std::env::var("WHISPER_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let threads_display: String = if threads == 0 {
        "auto".to_string()
    } else {
        threads.to_string()
    };

    println!(
        "Loading model: {} (language={}, threads={})",
        model_path, language, threads_display
    );

    let start = Instant::now();
    let stt = WhisperSttPlus::new(&model_path, &language, threads)?;
    println!("Model loaded in {:?}\n", start.elapsed());

    // Test 1: Single transcription
    println!("Test 1: Single complete transcription");
    // Generate synthetic audio (silence + simple tone)
    let audio_len = 48_000; // 3 seconds @ 16kHz
    let mut audio = vec![0.0f32; audio_len];

    // Add some simple sine wave pattern (not real speech but tests the pipeline)
    for i in 0..audio_len {
        audio[i] = 0.1 * (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16000.0).sin();
    }

    let start = Instant::now();
    match stt.transcribe_complete(&audio) {
        Ok(text) => println!("  Transcription: '{}'", text),
        Err(e) => println!("  Error: {}", e),
    }
    println!("  Time: {:?}\n", start.elapsed());

    // Test 2: Streaming with chunks
    println!("Test 2: Incremental streaming");
    let mut streamer = stt.create_streamer();

    let chunk_size = 8000; // 500ms chunks @ 16kHz
    let num_chunks = 6; // 3 seconds total

    for i in 0..num_chunks {
        let chunk_start = i * chunk_size;
        let chunk_end = chunk_start + chunk_size;
        let chunk = &audio[chunk_start..chunk_end.min(audio.len())];

        match streamer.feed_chunk(chunk)? {
            Some(partial) => {
                println!("  Chunk {}: partial = '{}'", i + 1, partial);
            }
            None => {
                println!("  Chunk {}: accumulating...", i + 1);
            }
        }
    }

    let final_text = streamer.finalize()?;
    println!("  Final result: '{}'\n", final_text);

    // Test 3: Transcription with prompt
    println!("Test 3: With initial prompt");
    let prompt = "hola";
    match stt.transcribe_with_prompt(&audio, prompt) {
        Ok(text) => println!("  Result with prompt '{}': '{}'", prompt, text),
        Err(e) => println!("  Error: {}", e),
    }

    println!("\n=== All tests completed ===");
    Ok(())
}
