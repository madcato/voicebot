use std::collections::VecDeque;

/// Audio buffer for accumulating audio chunks
pub struct AudioBuffer {
    buffer: VecDeque<f32>,
    max_size: usize,
    sample_rate: u32,
}

impl AudioBuffer {
    pub fn new(sample_rate: u32, max_duration_secs: u32) -> Self {
        let max_size = (sample_rate * max_duration_secs) as usize;
        Self {
            buffer: VecDeque::with_capacity(max_size),
            max_size,
            sample_rate,
        }
    }

    /// Add audio samples to the buffer
    pub fn push(&mut self, samples: &[f32]) {
        for &sample in samples {
            if self.buffer.len() >= self.max_size {
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }
    }

    /// Get all buffered samples
    pub fn get_samples(&self) -> Vec<f32> {
        self.buffer.iter().copied().collect()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get buffer length in samples
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Get buffer duration in milliseconds
    pub fn duration_ms(&self) -> u32 {
        ((self.buffer.len() as f32 / self.sample_rate as f32) * 1000.0) as u32
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get samples starting from `offset` (number of samples to skip from the front).
    /// If `offset >= self.buffer.len()` returns an empty Vec.
    pub fn get_samples_from(&self, offset: usize) -> Vec<f32> {
        self.buffer.iter().skip(offset).copied().collect()
    }

    /// Number of samples currently in the buffer.
    pub fn sample_count(&self) -> usize {
        self.buffer.len()
    }
}
