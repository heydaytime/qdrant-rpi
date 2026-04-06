//! LRU Tracking for RPI
//!
//! Tracks access patterns to support:
//! - Promotion of frequently accessed points to lower shells
//! - Identification of when rebalancing is needed
//! - Statistics collection for monitoring cache convergence

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::config::RpiStats;

/// Thread-safe tracker for RPI access patterns
pub struct RpiTracker {
    stats: RwLock<RpiStats>,
    /// Cumulative search depth for calculating running average
    cumulative_depth: AtomicU64,
    /// Number of non-miss searches
    hit_count: AtomicU64,
}

impl RpiTracker {
    pub fn new(max_shells: u8) -> Self {
        Self {
            stats: RwLock::new(RpiStats::new(max_shells)),
            cumulative_depth: AtomicU64::new(0),
            hit_count: AtomicU64::new(0),
        }
    }

    /// Record a search that hit at a particular shell
    pub fn record_hit(&self, shell: u8) {
        let mut stats = self.stats.write();
        stats.record_hit(shell);

        // Update atomic counters for fast average calculation
        self.cumulative_depth
            .fetch_add(u64::from(shell), Ordering::Relaxed);
        self.hit_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a search miss (searched all shells, found nothing)
    pub fn record_miss(&self) {
        let mut stats = self.stats.write();
        stats.record_miss();
    }

    /// Record a demotion operation
    pub fn record_demotion(&self) {
        let mut stats = self.stats.write();
        stats.total_demotions += 1;
    }

    /// Record a promotion operation
    pub fn record_promotion(&self) {
        let mut stats = self.stats.write();
        stats.total_promotions += 1;
    }

    /// Record an eviction (point exceeded max_shells)
    pub fn record_eviction(&self) {
        let mut stats = self.stats.write();
        stats.total_evictions += 1;
    }

    /// Record a rebalancing operation
    pub fn record_rebalance(&self) {
        let mut stats = self.stats.write();
        stats.rebalance_count += 1;
    }

    /// Get current average search depth (fast path using atomics)
    pub fn average_depth(&self) -> f32 {
        let hits = self.hit_count.load(Ordering::Relaxed);
        if hits == 0 {
            return 0.0;
        }
        let cumulative = self.cumulative_depth.load(Ordering::Relaxed);
        cumulative as f32 / hits as f32
    }

    /// Check if rebalancing is needed based on average search depth
    pub fn needs_rebalance(&self, threshold: f32) -> bool {
        let hits = self.hit_count.load(Ordering::Relaxed);
        // Need sufficient data before triggering rebalance
        if hits < 100 {
            return false;
        }
        self.average_depth() > threshold
    }

    /// Get a snapshot of current stats
    pub fn get_stats(&self) -> RpiStats {
        self.stats.read().clone()
    }

    /// Reset statistics (typically after rebalancing)
    pub fn reset_stats(&self, max_shells: u8) {
        let mut stats = self.stats.write();
        *stats = RpiStats::new(max_shells);
        self.cumulative_depth.store(0, Ordering::Relaxed);
        self.hit_count.store(0, Ordering::Relaxed);
    }
}

impl Default for RpiTracker {
    fn default() -> Self {
        Self::new(5)
    }
}

/// Point-level access tracking data stored in payload
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PointAccessData {
    /// Number of times this point was accessed (returned in search results)
    pub hit_count: u32,
    /// Last access timestamp (unix seconds)
    pub last_access: u64,
    /// Number of times this point was demoted
    pub demotion_count: u8,
    /// Current shell number
    pub current_shell: u8,
    /// Original shell (always 1 for new points)
    pub original_shell: u8,
}

impl PointAccessData {
    pub fn new() -> Self {
        let now = current_timestamp();
        Self {
            hit_count: 0,
            last_access: now,
            demotion_count: 0,
            current_shell: 1,
            original_shell: 1,
        }
    }

    /// Record an access to this point
    pub fn record_access(&mut self) {
        self.hit_count = self.hit_count.saturating_add(1);
        self.last_access = current_timestamp();
    }

    /// Record a demotion
    pub fn record_demotion(&mut self, new_shell: u8) {
        self.demotion_count = self.demotion_count.saturating_add(1);
        self.current_shell = new_shell;
    }

    /// Record a promotion
    pub fn record_promotion(&mut self, new_shell: u8) {
        self.current_shell = new_shell;
        // Reset hit count after promotion to avoid immediate re-promotion
        self.hit_count = 0;
    }

    /// Check if this point is eligible for promotion
    pub fn eligible_for_promotion(&self, min_hits: u32) -> bool {
        self.current_shell > 1 && self.hit_count >= min_hits
    }
}

/// Get current unix timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Shared RPI tracker that can be cloned across threads
pub type SharedRpiTracker = Arc<RpiTracker>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_average_depth() {
        let tracker = RpiTracker::new(5);

        // Record some hits
        for _ in 0..50 {
            tracker.record_hit(1);
        }
        for _ in 0..50 {
            tracker.record_hit(2);
        }

        // Average should be (50*1 + 50*2) / 100 = 1.5
        assert!((tracker.average_depth() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_needs_rebalance() {
        let tracker = RpiTracker::new(5);

        // Not enough data yet
        for _ in 0..50 {
            tracker.record_hit(5);
        }
        assert!(!tracker.needs_rebalance(3.0));

        // Now we have enough
        for _ in 0..60 {
            tracker.record_hit(5);
        }
        // Average depth is 5.0, threshold is 3.0
        assert!(tracker.needs_rebalance(3.0));
    }

    #[test]
    fn test_point_access_data() {
        let mut data = PointAccessData::new();

        assert_eq!(data.current_shell, 1);
        assert_eq!(data.hit_count, 0);

        data.record_access();
        data.record_access();
        assert_eq!(data.hit_count, 2);

        data.record_demotion(2);
        assert_eq!(data.current_shell, 2);
        assert_eq!(data.demotion_count, 1);

        assert!(!data.eligible_for_promotion(10));
        for _ in 0..10 {
            data.record_access();
        }
        assert!(data.eligible_for_promotion(10));
    }
}
