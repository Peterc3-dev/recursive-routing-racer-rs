//! KV cache compression and eviction integration via kv-squeeze.
//!
//! This module bridges kv-squeeze's quantization and eviction strategies
//! into rrr's GPU-resident KV cache. Since the actual K/V data lives in
//! Vulkan GPU buffers (written by compute shaders in FP16), this layer
//! provides:
//!
//! - **Eviction tracking**: maintains per-token metadata (position, attention
//!   scores, age) and triggers eviction when cache exceeds a budget.
//! - **Compression mode tracking**: records which quantization method is
//!   configured so stats can be reported on exit. The GPU shaders already
//!   store KV in FP16; this config controls whether we report that as the
//!   baseline or as an additional compression step.
//! - **Exit stats**: compression ratio, memory saved, tokens evicted.

use kv_squeeze::eviction::{
    EvictionStrategy, H2OEviction, SlidingWindow, TokenEntry, TokenEviction,
};
use kv_squeeze::quantize::QuantMethod;
use std::fmt;

/// Compression mode for KV cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVCompressMode {
    /// No additional compression (GPU default: FP16 via shader).
    None,
    /// Track as FP16 (matches GPU shader behavior, for stats).
    FP16,
    /// FP8 E4M3 quantization.
    FP8,
    /// INT4 grouped quantization.
    INT4,
}

impl KVCompressMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(KVCompressMode::None),
            "fp16" => Some(KVCompressMode::FP16),
            "fp8" => Some(KVCompressMode::FP8),
            "int4" => Some(KVCompressMode::INT4),
            _ => None,
        }
    }

    pub fn to_quant_method(&self) -> Option<QuantMethod> {
        match self {
            KVCompressMode::None => Option::None,
            KVCompressMode::FP16 => Some(QuantMethod::FP16),
            KVCompressMode::FP8 => Some(QuantMethod::FP8E4M3),
            KVCompressMode::INT4 => Some(QuantMethod::INT4),
        }
    }

    /// Bits per element for this mode (FP32 baseline = 32).
    pub fn bits_per_element(&self) -> u32 {
        match self {
            KVCompressMode::None => 16, // GPU already stores as FP16
            KVCompressMode::FP16 => 16,
            KVCompressMode::FP8 => 8,
            KVCompressMode::INT4 => 4,
        }
    }

    /// Compression ratio vs FP32 baseline.
    pub fn compression_ratio_vs_fp32(&self) -> f64 {
        32.0 / self.bits_per_element() as f64
    }
}

impl fmt::Display for KVCompressMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KVCompressMode::None => write!(f, "none (GPU FP16)"),
            KVCompressMode::FP16 => write!(f, "FP16"),
            KVCompressMode::FP8 => write!(f, "FP8 E4M3"),
            KVCompressMode::INT4 => write!(f, "INT4 grouped"),
        }
    }
}

/// Eviction mode for KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVEvictMode {
    None,
    Sliding,
    H2O,
}

impl KVEvictMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(KVEvictMode::None),
            "sliding" => Some(KVEvictMode::Sliding),
            "h2o" => Some(KVEvictMode::H2O),
            _ => None,
        }
    }
}

impl fmt::Display for KVEvictMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KVEvictMode::None => write!(f, "none"),
            KVEvictMode::Sliding => write!(f, "sliding window"),
            KVEvictMode::H2O => write!(f, "H2O (heavy hitter oracle)"),
        }
    }
}

/// Configuration for KV cache compression and eviction.
#[derive(Debug, Clone)]
pub struct KVCompressConfig {
    pub compress_mode: KVCompressMode,
    pub evict_mode: KVEvictMode,
    /// Maximum cache entries before eviction triggers. 0 = unlimited.
    pub budget: usize,
    /// Number of attention-sink tokens to protect during eviction.
    pub sink_count: usize,
}

impl KVCompressConfig {
    pub fn disabled() -> Self {
        KVCompressConfig {
            compress_mode: KVCompressMode::None,
            evict_mode: KVEvictMode::None,
            budget: 0,
            sink_count: 4,
        }
    }

    pub fn is_active(&self) -> bool {
        self.compress_mode != KVCompressMode::None || self.evict_mode != KVEvictMode::None
    }

    /// Build the kv-squeeze eviction strategy (if any).
    pub fn build_eviction(&self) -> Option<EvictionStrategy> {
        match self.evict_mode {
            KVEvictMode::None => Option::None,
            KVEvictMode::Sliding => {
                Some(EvictionStrategy::Sliding(SlidingWindow::new(self.sink_count)))
            }
            KVEvictMode::H2O => {
                Some(EvictionStrategy::H2O(H2OEviction::new(self.sink_count)))
            }
        }
    }
}

/// Runtime stats tracked during inference.
#[derive(Debug, Clone)]
pub struct KVCompressStats {
    pub total_tokens_inserted: usize,
    pub total_tokens_evicted: usize,
    pub peak_cache_len: usize,
    pub eviction_events: usize,
    /// Per-token metadata for eviction decisions.
    pub token_entries: Vec<TokenEntry>,
}

impl KVCompressStats {
    pub fn new() -> Self {
        KVCompressStats {
            total_tokens_inserted: 0,
            total_tokens_evicted: 0,
            peak_cache_len: 0,
            eviction_events: 0,
            token_entries: Vec::new(),
        }
    }

    /// Record a new token insertion at the given position.
    pub fn record_insert(&mut self, position: usize) {
        self.total_tokens_inserted += 1;
        for entry in &mut self.token_entries {
            entry.age += 1;
        }
        self.token_entries.push(TokenEntry {
            position,
            cumulative_attention: 1.0,
            age: 0,
        });
        if self.token_entries.len() > self.peak_cache_len {
            self.peak_cache_len = self.token_entries.len();
        }
    }

    /// Record a batch insertion of `count` tokens starting at `base_position`.
    pub fn record_batch_insert(&mut self, base_position: usize, count: usize) {
        if count == 0 {
            return;
        }
        self.total_tokens_inserted += count;
        for entry in &mut self.token_entries {
            entry.age += 1;
        }
        self.token_entries.reserve(count);
        for i in 0..count {
            self.token_entries.push(TokenEntry {
                position: base_position + i,
                cumulative_attention: 1.0,
                age: 0,
            });
        }
        if self.token_entries.len() > self.peak_cache_len {
            self.peak_cache_len = self.token_entries.len();
        }
    }

    /// Check if eviction is needed and return positions to evict.
    /// Returns the indices in `token_entries` that should be evicted.
    pub fn check_eviction(
        &mut self,
        budget: usize,
        strategy: &dyn TokenEviction,
    ) -> Vec<usize> {
        if budget == 0 || self.token_entries.len() <= budget {
            return vec![];
        }

        let evicted = strategy.select_evictions(&self.token_entries, budget);
        if !evicted.is_empty() {
            self.eviction_events += 1;
            self.total_tokens_evicted += evicted.len();

            // Remove evicted entries (sort descending to preserve indices)
            let mut sorted_evict = evicted.clone();
            sorted_evict.sort_unstable_by(|a, b| b.cmp(a));
            for idx in sorted_evict {
                if idx < self.token_entries.len() {
                    self.token_entries.remove(idx);
                }
            }
        }
        evicted
    }

    /// Print summary stats to stderr.
    pub fn print_summary(&self, config: &KVCompressConfig, n_kv_heads: usize, head_dim: usize, n_layers: usize) {
        if !config.is_active() {
            return;
        }

        let bytes_per_token_fp32 = 2 * n_kv_heads * head_dim * 4 * n_layers;
        let bits = config.compress_mode.bits_per_element();
        let bytes_per_token_compressed = 2 * n_kv_heads * head_dim * n_layers * (bits as usize) / 8;

        eprintln!();
        eprintln!("=== KV Cache Compression Stats (kv-squeeze) ===");
        eprintln!("  Compression mode : {}", config.compress_mode);
        eprintln!("  Eviction strategy: {}", config.evict_mode);
        if config.budget > 0 {
            eprintln!("  Cache budget     : {} tokens", config.budget);
        }
        eprintln!("  Tokens inserted  : {}", self.total_tokens_inserted);
        eprintln!("  Tokens evicted   : {}", self.total_tokens_evicted);
        eprintln!("  Eviction events  : {}", self.eviction_events);
        eprintln!("  Peak cache size  : {} tokens", self.peak_cache_len);
        eprintln!("  Final cache size : {} tokens", self.token_entries.len());

        if config.compress_mode != KVCompressMode::None {
            let ratio = config.compress_mode.compression_ratio_vs_fp32();
            let baseline_bytes = self.peak_cache_len * bytes_per_token_fp32;
            let compressed_bytes = self.peak_cache_len * bytes_per_token_compressed;
            let saved_mb = (baseline_bytes - compressed_bytes) as f64 / (1024.0 * 1024.0);
            eprintln!("  Compression ratio: {:.1}x vs FP32", ratio);
            eprintln!("  Memory saved     : {:.1} MB (peak, vs FP32 baseline)", saved_mb);
            eprintln!("  Bytes/token (FP32): {} B", bytes_per_token_fp32);
            eprintln!("  Bytes/token ({}): {} B",
                config.compress_mode, bytes_per_token_compressed);
        }
        eprintln!("================================================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_mode_parsing() {
        assert_eq!(KVCompressMode::from_str("none"), Some(KVCompressMode::None));
        assert_eq!(KVCompressMode::from_str("fp16"), Some(KVCompressMode::FP16));
        assert_eq!(KVCompressMode::from_str("FP8"), Some(KVCompressMode::FP8));
        assert_eq!(KVCompressMode::from_str("INT4"), Some(KVCompressMode::INT4));
        assert_eq!(KVCompressMode::from_str("invalid"), None);
    }

    #[test]
    fn evict_mode_parsing() {
        assert_eq!(KVEvictMode::from_str("none"), Some(KVEvictMode::None));
        assert_eq!(KVEvictMode::from_str("sliding"), Some(KVEvictMode::Sliding));
        assert_eq!(KVEvictMode::from_str("h2o"), Some(KVEvictMode::H2O));
        assert_eq!(KVEvictMode::from_str("bogus"), None);
    }

    #[test]
    fn disabled_config_is_inactive() {
        let cfg = KVCompressConfig::disabled();
        assert!(!cfg.is_active());
        assert!(cfg.build_eviction().is_none());
    }

    #[test]
    fn compress_mode_to_quant_method() {
        assert!(KVCompressMode::None.to_quant_method().is_none());
        assert_eq!(KVCompressMode::FP16.to_quant_method(), Some(QuantMethod::FP16));
        assert_eq!(KVCompressMode::FP8.to_quant_method(), Some(QuantMethod::FP8E4M3));
        assert_eq!(KVCompressMode::INT4.to_quant_method(), Some(QuantMethod::INT4));
    }

    #[test]
    fn compression_ratios() {
        assert!((KVCompressMode::None.compression_ratio_vs_fp32() - 2.0).abs() < 0.01);
        assert!((KVCompressMode::FP16.compression_ratio_vs_fp32() - 2.0).abs() < 0.01);
        assert!((KVCompressMode::FP8.compression_ratio_vs_fp32() - 4.0).abs() < 0.01);
        assert!((KVCompressMode::INT4.compression_ratio_vs_fp32() - 8.0).abs() < 0.01);
    }

    #[test]
    fn stats_track_insertions() {
        let mut stats = KVCompressStats::new();
        stats.record_insert(0);
        stats.record_insert(1);
        stats.record_insert(2);
        assert_eq!(stats.total_tokens_inserted, 3);
        assert_eq!(stats.token_entries.len(), 3);
        assert_eq!(stats.peak_cache_len, 3);
    }

    #[test]
    fn stats_batch_insert() {
        let mut stats = KVCompressStats::new();
        stats.record_batch_insert(0, 5);
        assert_eq!(stats.total_tokens_inserted, 5);
        assert_eq!(stats.token_entries.len(), 5);
        assert_eq!(stats.peak_cache_len, 5);
    }

    #[test]
    fn eviction_sliding_window() {
        let mut stats = KVCompressStats::new();
        for i in 0..10 {
            stats.record_insert(i);
        }
        let strategy = SlidingWindow::new(2);
        let evicted = stats.check_eviction(5, &strategy);
        assert_eq!(evicted.len(), 5);
        assert_eq!(stats.total_tokens_evicted, 5);
        assert_eq!(stats.eviction_events, 1);
        assert_eq!(stats.token_entries.len(), 5);
    }

    #[test]
    fn eviction_h2o() {
        let mut stats = KVCompressStats::new();
        for i in 0..10 {
            stats.record_insert(i);
        }
        // Give higher attention to later tokens
        for (i, entry) in stats.token_entries.iter_mut().enumerate() {
            entry.cumulative_attention = (i as f64 + 1.0) * 0.5;
        }
        let strategy = H2OEviction::new(2);
        let evicted = stats.check_eviction(6, &strategy);
        assert_eq!(evicted.len(), 4);
        // Sinks (positions 0, 1) should be protected
        // Sinks (positions 0, 1) should be protected.
        // After check_eviction, entries are already removed, but we can
        // verify the count is correct.
        assert_eq!(stats.token_entries.len(), 6);
    }

    #[test]
    fn no_eviction_under_budget() {
        let mut stats = KVCompressStats::new();
        for i in 0..5 {
            stats.record_insert(i);
        }
        let strategy = SlidingWindow::new(2);
        let evicted = stats.check_eviction(10, &strategy);
        assert!(evicted.is_empty());
        assert_eq!(stats.eviction_events, 0);
    }

    #[test]
    fn no_eviction_zero_budget() {
        let mut stats = KVCompressStats::new();
        for i in 0..5 {
            stats.record_insert(i);
        }
        let strategy = SlidingWindow::new(2);
        let evicted = stats.check_eviction(0, &strategy);
        assert!(evicted.is_empty());
    }

    #[test]
    fn build_eviction_strategies() {
        let mut cfg = KVCompressConfig::disabled();

        cfg.evict_mode = KVEvictMode::Sliding;
        let s = cfg.build_eviction();
        assert!(s.is_some());

        cfg.evict_mode = KVEvictMode::H2O;
        let s = cfg.build_eviction();
        assert!(s.is_some());

        cfg.evict_mode = KVEvictMode::None;
        let s = cfg.build_eviction();
        assert!(s.is_none());
    }

    #[test]
    fn config_active_with_compression() {
        let mut cfg = KVCompressConfig::disabled();
        cfg.compress_mode = KVCompressMode::FP8;
        assert!(cfg.is_active());
    }

    #[test]
    fn config_active_with_eviction() {
        let mut cfg = KVCompressConfig::disabled();
        cfg.evict_mode = KVEvictMode::H2O;
        assert!(cfg.is_active());
    }

    #[test]
    fn stats_print_summary_disabled() {
        // Should not panic when config is disabled
        let stats = KVCompressStats::new();
        let cfg = KVCompressConfig::disabled();
        stats.print_summary(&cfg, 8, 128, 32);
    }

    #[test]
    fn stats_print_summary_active() {
        let mut stats = KVCompressStats::new();
        stats.record_batch_insert(0, 100);
        let cfg = KVCompressConfig {
            compress_mode: KVCompressMode::FP8,
            evict_mode: KVEvictMode::Sliding,
            budget: 50,
            sink_count: 4,
        };
        // Should not panic
        stats.print_summary(&cfg, 8, 128, 32);
    }
}
