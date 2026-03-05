use anyhow::Result;
use async_trait::async_trait;

use super::models::{ModelConfig, ModelType};

/// S2S Model Adapter - Abstraction layer for interchangeable S2S models
pub struct S2SAdapter {
    model: Box<dyn S2SModel + Send + Sync>,
    config: ModelConfig,
}

impl S2SAdapter {
    /// Create a new adapter with the specified model type
    pub async fn new(model_type: ModelType, config: ModelConfig) -> Result<Self> {
        let model = Self::create_model(model_type, &config).await?;
        Ok(Self { model, config })
    }

    /// Create an adapter from a pre-built model instance (useful for testing).
    pub fn with_model(model: Box<dyn S2SModel + Send + Sync>, config: ModelConfig) -> Self {
        Self { model, config }
    }

    /// Create a model instance based on type
    async fn create_model(
        model_type: ModelType,
        config: &ModelConfig,
    ) -> Result<Box<dyn S2SModel + Send + Sync>> {
        match model_type {
            ModelType::LlamaOmni => {
                Ok(Box::new(super::models::llama_omni::LlamaOmniModel::new(config).await?))
            }
            ModelType::Moshi => {
                Ok(Box::new(super::models::moshi::MoshiModel::new(config).await?))
            }
            ModelType::Ultravox => {
                Ok(Box::new(super::models::ultravox::UltravoxModel::new(config).await?))
            }
            ModelType::LFM => {
                Ok(Box::new(super::models::lfm::LFMModel::new(config).await?))
            }
        }
    }

    /// Process audio input and generate audio response
    pub async fn process(&mut self, request: S2SRequest) -> Result<S2SResponse> {
        self.model.process(request).await
    }

    /// Get model information
    pub fn model_info(&self) -> &ModelConfig {
        &self.config
    }

    /// Check if model supports streaming
    pub fn supports_streaming(&self) -> bool {
        self.model.supports_streaming()
    }

    /// Check if model supports tool calls
    pub fn supports_tools(&self) -> bool {
        self.model.supports_tools()
    }
}

/// S2S Model trait - Interface that all models must implement
#[async_trait]
pub trait S2SModel {
    /// Process audio input and generate response
    async fn process(&mut self, request: S2SRequest) -> Result<S2SResponse>;

    /// Check if model supports streaming output
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Check if model supports tool calling
    fn supports_tools(&self) -> bool {
        false
    }

    /// Get model name
    fn name(&self) -> &str;
}

/// Request to S2S model
#[derive(Debug, Clone)]
pub struct S2SRequest {
    /// Input audio samples (mono, f32)
    pub audio: Vec<f32>,
    /// Sample rate of input audio
    pub sample_rate: u32,
    /// Conversation context/history
    pub context: Vec<String>,
    /// Optional tool definitions
    pub tools: Option<Vec<ToolDefinition>>,
    /// Whether to stream the response
    pub stream: bool,
}

/// Response from S2S model
#[derive(Debug, Clone)]
pub struct S2SResponse {
    /// Output audio samples (mono, f32)
    pub audio: Vec<f32>,
    /// Sample rate of output audio
    pub sample_rate: u32,
    /// Transcription of user input (if available)
    pub input_text: Option<String>,
    /// Text version of response (if available)
    pub output_text: Option<String>,
    /// Tool calls requested by model (if any)
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}
