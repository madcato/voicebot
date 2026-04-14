/// Build script for voicebot.

fn main() {
    // Re-run if environment changes
    println!("cargo:rerun-if-env-changed=WHISPER_MODEL");
    
    // Note: To enable CoreML acceleration on macOS, users must:
    // 1. Set WHISPER_USE_COREML=1 before building
    // 2. Have the corresponding *-encoder.mlmodelc file present  
    // 3. Ensure Xcode command line tools are installed
    // Example: WHISPER_USE_COREML=1 cargo clean && cargo build --release
    
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    if target_os == "macos" {
        println!("cargo:warning=Building for macOS with Metal GPU acceleration");
        
        // Only pass CoreML flag if explicitly requested by user
        if std::env::var("WHISPER_USE_COREML").is_ok() {
            println!("cargo:rustc-env=CMAKE_WHISPER_USE_COREML=ON");
            println!("cargo:rustc-env=WHISPER_USE_COREML=1");
            println!("cargo:warning=CoreML requested — ensure Xcode CLI tools are installed");
        } else {
            println!("cargo:warning=Use WHISPER_USE_COREML=1 to enable ANE acceleration");
        }
    }
}
