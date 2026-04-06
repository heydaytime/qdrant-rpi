//! RPI Configuration types

use std::hash::{Hash, Hasher};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

/// Configuration for Radial Priority Indexing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(rename_all = "snake_case")]
pub struct RpiConfig {
    /// Number of quality shells (k=1 to k=max_shells).
    /// Points demoted beyond max_shells are evicted.
    /// Default: 5
    #[serde(default = "default_max_shells")]
    #[validate(range(min = 2, max = 20))]
    pub max_shells: u8,

    /// Base epsilon (search radius) for k=1 shell.
    /// Scales as epsilon * k for higher shells.
    /// This is the Euclidean distance threshold for considering a match.
    /// Default: 0.1
    #[serde(default = "default_base_epsilon")]
    #[validate(range(min = 0.001, max = 10.0))]
    pub base_epsilon: f32,

    /// Name of the source vector to use for RPI shells.
    /// If None, uses the default vector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_vector: Option<String>,

    /// Number of consecutive negative signals before demoting a point.
    /// Default: 2
    #[serde(default = "default_demotion_threshold")]
    #[validate(range(min = 1, max = 100))]
    pub demotion_threshold: u8,

    /// Whether to use HNSW for k=1 shell (true) or brute force for all shells (false).
    /// Default: true (HNSW for k=1, brute force for k>1)
    #[serde(default = "default_hnsw_for_shell_one")]
    pub hnsw_for_shell_one: bool,

    /// Whether to track LRU (Least Recently Used) access patterns.
    /// When enabled, frequently accessed points may be promoted to lower shells.
    /// Default: true
    #[serde(default = "default_track_lru")]
    pub track_lru: bool,

    /// Minimum hit count before a point can be promoted to a lower shell.
    /// Only relevant if track_lru is true.
    /// Default: 10
    #[serde(default = "default_promotion_threshold")]
    #[validate(range(min = 1, max = 1000))]
    pub promotion_threshold: u32,

    /// When average search depth exceeds this threshold, trigger rebalancing.
    /// A value of 4.0 means if searches are consistently hitting k=4+, rebalance.
    /// Default: 3.0
    #[serde(default = "default_rebalance_threshold")]
    #[validate(range(min = 1.5, max = 10.0))]
    pub rebalance_threshold: f32,
}

fn default_max_shells() -> u8 {
    5
}

fn default_base_epsilon() -> f32 {
    0.1
}

fn default_demotion_threshold() -> u8 {
    2
}

fn default_hnsw_for_shell_one() -> bool {
    true
}

fn default_track_lru() -> bool {
    true
}

fn default_promotion_threshold() -> u32 {
    10
}

fn default_rebalance_threshold() -> f32 {
    3.0
}

impl Default for RpiConfig {
    fn default() -> Self {
        Self {
            max_shells: default_max_shells(),
            base_epsilon: default_base_epsilon(),
            source_vector: None,
            demotion_threshold: default_demotion_threshold(),
            hnsw_for_shell_one: default_hnsw_for_shell_one(),
            track_lru: default_track_lru(),
            promotion_threshold: default_promotion_threshold(),
            rebalance_threshold: default_rebalance_threshold(),
        }
    }
}

// Manually implement Eq because f32 fields don't implement Eq.
// PartialEq is derived and handles float comparison correctly.
impl Eq for RpiConfig {}

impl Hash for RpiConfig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash all non-float fields. We skip hashing float fields
        // (base_epsilon, rebalance_threshold) because floats cannot
        // be reliably hashed. This follows the same pattern as
        // StrictModeConfig in lib/segment/src/types.rs
        self.max_shells.hash(state);
        self.source_vector.hash(state);
        self.demotion_threshold.hash(state);
        self.hnsw_for_shell_one.hash(state);
        self.track_lru.hash(state);
        self.promotion_threshold.hash(state);
        // Skip: base_epsilon (f32)
        // Skip: rebalance_threshold (f32)
    }
}

impl RpiConfig {
    /// Calculate the epsilon (search radius) for a given shell
    pub fn epsilon_for_shell(&self, shell: u8) -> f32 {
        self.base_epsilon * f32::from(shell)
    }

    /// Convert epsilon to a score threshold for Euclidean distance.
    /// Local shard post-processes Euclidean scores back to distance values.
    /// For Euclidean distance, lower is better, so threshold is epsilon itself.
    pub fn score_threshold_for_shell(&self, shell: u8) -> f32 {
        self.epsilon_for_shell(shell)
    }
}

/// Statistics for RPI operations
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct RpiStats {
    /// Total number of shell searches performed
    pub total_searches: u64,
    /// Distribution of which shells produced hits (shell_number -> count)
    pub shell_hit_distribution: Vec<u64>,
    /// Number of demotions performed
    pub total_demotions: u64,
    /// Number of promotions performed (if LRU tracking enabled)
    pub total_promotions: u64,
    /// Number of evictions (points that exceeded max_shells)
    pub total_evictions: u64,
    /// Average search depth (weighted average of shell hits)
    pub average_search_depth: f32,
    /// Number of rebalancing operations triggered
    pub rebalance_count: u64,
}

impl RpiStats {
    pub fn new(max_shells: u8) -> Self {
        Self {
            shell_hit_distribution: vec![0; max_shells as usize + 1], // +1 for "miss" at index 0
            ..Default::default()
        }
    }

    /// Record a shell hit
    pub fn record_hit(&mut self, shell: u8) {
        self.total_searches += 1;
        if (shell as usize) < self.shell_hit_distribution.len() {
            self.shell_hit_distribution[shell as usize] += 1;
        }
        self.update_average_depth();
    }

    /// Record a miss (searched all shells, found nothing)
    pub fn record_miss(&mut self) {
        self.total_searches += 1;
        self.shell_hit_distribution[0] += 1; // Index 0 = miss
    }

    fn update_average_depth(&mut self) {
        if self.total_searches == 0 {
            self.average_search_depth = 0.0;
            return;
        }

        let weighted_sum: u64 = self
            .shell_hit_distribution
            .iter()
            .enumerate()
            .skip(1) // Skip misses at index 0
            .map(|(shell, &count)| shell as u64 * count)
            .sum();

        let total_hits: u64 = self.shell_hit_distribution.iter().skip(1).sum();

        if total_hits > 0 {
            self.average_search_depth = weighted_sum as f32 / total_hits as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epsilon_scaling() {
        let config = RpiConfig {
            base_epsilon: 0.1,
            ..Default::default()
        };

        assert!((config.epsilon_for_shell(1) - 0.1).abs() < 1e-6);
        assert!((config.epsilon_for_shell(2) - 0.2).abs() < 1e-6);
        assert!((config.epsilon_for_shell(5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_score_threshold() {
        let config = RpiConfig {
            base_epsilon: 0.1,
            ..Default::default()
        };

        // For shell 1: epsilon = 0.1, threshold = 0.1
        assert!((config.score_threshold_for_shell(1) - 0.1).abs() < 1e-6);
        // For shell 2: epsilon = 0.2, threshold = 0.2
        assert!((config.score_threshold_for_shell(2) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_stats_average_depth() {
        let mut stats = RpiStats::new(5);

        // 10 hits at shell 1
        for _ in 0..10 {
            stats.record_hit(1);
        }
        assert!((stats.average_search_depth - 1.0).abs() < 1e-6);

        // 10 more hits at shell 2
        for _ in 0..10 {
            stats.record_hit(2);
        }
        // Average should be (10*1 + 10*2) / 20 = 1.5
        assert!((stats.average_search_depth - 1.5).abs() < 1e-6);
    }
}
