//! Shell Search for RPI
//!
//! Implements sequential shell search with fallthrough:
//! 1. Search k=1 (highest quality) first
//! 2. If no results within epsilon, search k=2
//! 3. Continue until hit or k=max_shells
//! 4. Return shell number with results (definitive miss if none found)

use schemars::JsonSchema;
use segment::data_types::vectors::DenseVector;
use segment::types::PointIdType;
use serde::{Deserialize, Serialize};

use super::config::RpiConfig;
use super::scaling::scale_vector;
use super::shell_vector_name;

/// Request for shell-aware search
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ShellSearchRequest {
    /// The raw query vector (unscaled)
    pub vector: DenseVector,
    /// Maximum number of results to return
    pub limit: usize,
    /// Optional override for base epsilon
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epsilon: Option<f32>,
    /// Maximum shell to search (defaults to collection's max_shells)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_shell: Option<u8>,
    /// Whether to include payload in results
    #[serde(default)]
    pub with_payload: bool,
    /// Whether to include vectors in results
    #[serde(default)]
    pub with_vector: bool,
}

/// Metadata about a shell search result
/// The actual ScoredPoint results are returned separately by the API layer
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ShellSearchMetadata {
    /// Which shell produced the hit (0 = miss, 1-N = shell number)
    pub hit_shell: u8,
    /// How many shells were searched before finding results
    pub searched_shells: u8,
    /// Whether this is a definitive miss (searched all shells, nothing exists)
    pub definitive_miss: bool,
    /// Number of results returned
    pub result_count: usize,
}

impl ShellSearchMetadata {
    /// Create metadata for a successful hit
    pub fn hit(shell: u8, searched: u8, count: usize) -> Self {
        Self {
            hit_shell: shell,
            searched_shells: searched,
            definitive_miss: false,
            result_count: count,
        }
    }

    /// Create metadata for a definitive miss
    pub fn miss(max_shells: u8) -> Self {
        Self {
            hit_shell: 0,
            searched_shells: max_shells,
            definitive_miss: true,
            result_count: 0,
        }
    }
}

/// Parameters for searching a single shell
#[derive(Debug, Clone)]
pub struct ShellSearchParams {
    /// Which shell to search
    pub shell: u8,
    /// The scaled query vector for this shell
    pub scaled_query: DenseVector,
    /// The vector name for this shell
    pub vector_name: String,
    /// Score threshold (Euclidean distance epsilon)
    pub score_threshold: f32,
    /// Result limit
    pub limit: usize,
}

impl ShellSearchParams {
    /// Create search parameters for a specific shell
    pub fn for_shell(
        shell: u8,
        query: &[f32],
        config: &RpiConfig,
        limit: usize,
        epsilon_override: Option<f32>,
    ) -> Self {
        let base_epsilon = epsilon_override.unwrap_or(config.base_epsilon);
        let epsilon_k = base_epsilon * shell as f32;

        Self {
            shell,
            scaled_query: scale_vector(query, shell),
            vector_name: shell_vector_name(shell),
            score_threshold: epsilon_k,
            limit,
        }
    }
}

/// Iterator that generates search parameters for each shell in sequence
pub struct ShellSearchIterator<'a> {
    query: &'a [f32],
    config: &'a RpiConfig,
    limit: usize,
    epsilon_override: Option<f32>,
    max_shell: u8,
    current_shell: u8,
}

impl<'a> ShellSearchIterator<'a> {
    pub fn new(
        query: &'a [f32],
        config: &'a RpiConfig,
        limit: usize,
        epsilon_override: Option<f32>,
        max_shell_override: Option<u8>,
    ) -> Self {
        Self {
            query,
            config,
            limit,
            epsilon_override,
            max_shell: max_shell_override.unwrap_or(config.max_shells),
            current_shell: 1,
        }
    }
}

impl<'a> Iterator for ShellSearchIterator<'a> {
    type Item = ShellSearchParams;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_shell > self.max_shell {
            return None;
        }

        let params = ShellSearchParams::for_shell(
            self.current_shell,
            self.query,
            self.config,
            self.limit,
            self.epsilon_override,
        );

        self.current_shell += 1;
        Some(params)
    }
}

/// Request to explore/select a specific point from search results.
/// This signals that the user found this result useful, and can trigger
/// implicit demotion of other points that were passed over.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExplorePointRequest {
    /// The point that was explored/selected
    pub point_id: PointIdType,
    /// The original query vector that led to this exploration
    pub query_vector: DenseVector,
    /// Which shell this point was found in
    pub explored_shell: u8,
}

/// Response from exploring a point
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExplorePointResponse {
    /// Number of points that were marked for potential demotion
    pub points_marked: usize,
    /// Number of points that were actually demoted
    pub points_demoted: usize,
    /// Number of points that were evicted
    pub points_evicted: usize,
}

/// Request to insert a new point with RPI awareness
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RpiInsertRequest {
    /// Point ID to insert
    pub point_id: PointIdType,
    /// The raw embedding vector (will be stored at shell 1)
    pub vector: DenseVector,
    /// Optional payload to attach
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<segment::types::Payload>,
}

/// Batch insert request for RPI
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RpiBatchInsertRequest {
    /// Points to insert
    pub points: Vec<RpiInsertRequest>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_search_params() {
        let config = RpiConfig {
            base_epsilon: 0.1,
            max_shells: 5,
            ..Default::default()
        };

        let query = vec![1.0, 0.0, 0.0];

        let params = ShellSearchParams::for_shell(1, &query, &config, 10, None);
        assert_eq!(params.shell, 1);
        assert_eq!(params.vector_name, "rpi_shell_1");
        assert_eq!(params.scaled_query, vec![1.0, 0.0, 0.0]);
        assert!((params.score_threshold - 0.1).abs() < 1e-6);

        let params = ShellSearchParams::for_shell(3, &query, &config, 10, None);
        assert_eq!(params.shell, 3);
        assert_eq!(params.vector_name, "rpi_shell_3");
        assert_eq!(params.scaled_query, vec![3.0, 0.0, 0.0]);
        // epsilon_3 = 0.3, threshold = 0.3
        assert!((params.score_threshold - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_shell_search_iterator() {
        let config = RpiConfig {
            base_epsilon: 0.1,
            max_shells: 3,
            ..Default::default()
        };

        let query = vec![1.0, 0.0, 0.0];
        let iter = ShellSearchIterator::new(&query, &config, 10, None, None);

        let shells: Vec<u8> = iter.map(|p| p.shell).collect();
        assert_eq!(shells, vec![1, 2, 3]);
    }

    #[test]
    fn test_shell_search_response() {
        let hit = ShellSearchMetadata::hit(2, 2, 0);
        assert_eq!(hit.hit_shell, 2);
        assert_eq!(hit.searched_shells, 2);
        assert!(!hit.definitive_miss);

        let miss = ShellSearchMetadata::miss(5);
        assert_eq!(miss.hit_shell, 0);
        assert_eq!(miss.searched_shells, 5);
        assert!(miss.definitive_miss);
    }
}
