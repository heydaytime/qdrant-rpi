//! Demotion Engine for RPI
//!
//! Handles the movement of points between shells based on feedback signals.
//! - Demotion: Move from shell k to shell k+1 when quality is poor
//! - Promotion: Move from shell k to shell k-1 when frequently accessed (LRU)
//! - Eviction: Delete when exceeding max_shells

use segment::data_types::vectors::DenseVector;
use segment::types::PointIdType;

use super::config::RpiConfig;
use super::scaling::{scale_vector, unscale_vector};
use super::shell_vector_name;
use super::tracking::PointAccessData;

/// Result of a demotion operation
#[derive(Debug, Clone)]
pub enum DemotionResult {
    /// Point was demoted to a new shell
    Demoted { from_shell: u8, to_shell: u8 },
    /// Point was evicted (exceeded max_shells)
    Evicted { from_shell: u8 },
    /// Point was not demoted (below threshold)
    NotDemoted {
        current_demotions: u8,
        threshold: u8,
    },
}

/// Result of a promotion operation
#[derive(Debug, Clone)]
pub enum PromotionResult {
    /// Point was promoted to a lower shell
    Promoted { from_shell: u8, to_shell: u8 },
    /// Point is already at shell 1, cannot promote further
    AlreadyAtBest,
    /// Point doesn't have enough hits for promotion
    InsufficientHits { current: u32, required: u32 },
}

/// Demotion request - contains all information needed to demote a point
#[derive(Debug, Clone)]
pub struct DemotionRequest {
    pub point_id: PointIdType,
    pub current_shell: u8,
    pub original_vector: DenseVector,
    pub access_data: PointAccessData,
}

/// Promotion request - contains all information needed to promote a point
#[derive(Debug, Clone)]
pub struct PromotionRequest {
    pub point_id: PointIdType,
    pub current_shell: u8,
    pub original_vector: DenseVector,
    pub access_data: PointAccessData,
}

/// Operations to perform on the collection for demotion/promotion
#[derive(Debug, Clone)]
pub struct ShellMoveOperation {
    pub point_id: PointIdType,
    /// Vector name to delete from (None if evicting)
    pub delete_from: Option<String>,
    /// (Vector name, scaled vector) to insert to (None if evicting)
    pub insert_to: Option<(String, DenseVector)>,
    /// Updated access data to store in payload
    pub updated_access_data: PointAccessData,
}

impl ShellMoveOperation {
    /// Create a demotion operation
    pub fn demote(
        point_id: PointIdType,
        from_shell: u8,
        to_shell: u8,
        original_vector: &[f32],
        mut access_data: PointAccessData,
    ) -> Self {
        access_data.record_demotion(to_shell);

        Self {
            point_id,
            delete_from: Some(shell_vector_name(from_shell)),
            insert_to: Some((
                shell_vector_name(to_shell),
                scale_vector(original_vector, to_shell),
            )),
            updated_access_data: access_data,
        }
    }

    /// Create an eviction operation (delete from current shell, no insert)
    pub fn evict(point_id: PointIdType, from_shell: u8, access_data: PointAccessData) -> Self {
        Self {
            point_id,
            delete_from: Some(shell_vector_name(from_shell)),
            insert_to: None,
            updated_access_data: access_data,
        }
    }

    /// Create a promotion operation
    pub fn promote(
        point_id: PointIdType,
        from_shell: u8,
        to_shell: u8,
        original_vector: &[f32],
        mut access_data: PointAccessData,
    ) -> Self {
        access_data.record_promotion(to_shell);

        Self {
            point_id,
            delete_from: Some(shell_vector_name(from_shell)),
            insert_to: Some((
                shell_vector_name(to_shell),
                scale_vector(original_vector, to_shell),
            )),
            updated_access_data: access_data,
        }
    }
}

/// Evaluate whether a point should be demoted based on accumulated negative signals
pub fn evaluate_demotion(
    config: &RpiConfig,
    access_data: &PointAccessData,
    additional_negative_signals: u8,
) -> DemotionResult {
    let total_demotions = access_data
        .demotion_count
        .saturating_add(additional_negative_signals);

    if total_demotions < config.demotion_threshold {
        return DemotionResult::NotDemoted {
            current_demotions: total_demotions,
            threshold: config.demotion_threshold,
        };
    }

    let new_shell = access_data.current_shell.saturating_add(1);

    if new_shell > config.max_shells {
        DemotionResult::Evicted {
            from_shell: access_data.current_shell,
        }
    } else {
        DemotionResult::Demoted {
            from_shell: access_data.current_shell,
            to_shell: new_shell,
        }
    }
}

/// Evaluate whether a point should be promoted based on LRU access patterns
pub fn evaluate_promotion(config: &RpiConfig, access_data: &PointAccessData) -> PromotionResult {
    if access_data.current_shell == 1 {
        return PromotionResult::AlreadyAtBest;
    }

    if access_data.hit_count < config.promotion_threshold {
        return PromotionResult::InsufficientHits {
            current: access_data.hit_count,
            required: config.promotion_threshold,
        };
    }

    PromotionResult::Promoted {
        from_shell: access_data.current_shell,
        to_shell: access_data.current_shell - 1,
    }
}

/// Calculate the original (unscaled) vector from a stored shell vector
pub fn extract_original_vector(stored_vector: &[f32], shell: u8) -> DenseVector {
    unscale_vector(stored_vector, shell)
}

/// Batch demotion evaluator - identifies all points that need demotion
/// based on a query that was answered from a higher shell
pub struct ImplicitDemotionEvaluator {
    /// Points that should be demoted (were in lower shells but passed over)
    pub points_to_demote: Vec<PointIdType>,
    /// The query that triggered this evaluation
    pub query_vector: DenseVector,
    /// The shell that actually answered the query
    pub answering_shell: u8,
}

impl ImplicitDemotionEvaluator {
    pub fn new(query_vector: DenseVector, answering_shell: u8) -> Self {
        Self {
            points_to_demote: Vec::new(),
            query_vector,
            answering_shell,
        }
    }

    /// Add a point that should be considered for demotion
    /// (was found in a shell lower than the answering shell but wasn't chosen)
    pub fn add_passed_over_point(&mut self, point_id: PointIdType) {
        self.points_to_demote.push(point_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_demotion_below_threshold() {
        let config = RpiConfig {
            demotion_threshold: 3,
            max_shells: 5,
            ..Default::default()
        };

        let access_data = PointAccessData {
            demotion_count: 1,
            current_shell: 1,
            ..Default::default()
        };

        match evaluate_demotion(&config, &access_data, 1) {
            DemotionResult::NotDemoted {
                current_demotions,
                threshold,
            } => {
                assert_eq!(current_demotions, 2);
                assert_eq!(threshold, 3);
            }
            _ => panic!("Expected NotDemoted"),
        }
    }

    #[test]
    fn test_evaluate_demotion_at_threshold() {
        let config = RpiConfig {
            demotion_threshold: 2,
            max_shells: 5,
            ..Default::default()
        };

        let access_data = PointAccessData {
            demotion_count: 1,
            current_shell: 1,
            ..Default::default()
        };

        match evaluate_demotion(&config, &access_data, 1) {
            DemotionResult::Demoted {
                from_shell,
                to_shell,
            } => {
                assert_eq!(from_shell, 1);
                assert_eq!(to_shell, 2);
            }
            _ => panic!("Expected Demoted"),
        }
    }

    #[test]
    fn test_evaluate_demotion_eviction() {
        let config = RpiConfig {
            demotion_threshold: 1,
            max_shells: 3,
            ..Default::default()
        };

        let access_data = PointAccessData {
            demotion_count: 0,
            current_shell: 3, // Already at max
            ..Default::default()
        };

        match evaluate_demotion(&config, &access_data, 1) {
            DemotionResult::Evicted { from_shell } => {
                assert_eq!(from_shell, 3);
            }
            _ => panic!("Expected Evicted"),
        }
    }

    #[test]
    fn test_evaluate_promotion() {
        let config = RpiConfig {
            promotion_threshold: 10,
            ..Default::default()
        };

        // Not enough hits
        let access_data = PointAccessData {
            hit_count: 5,
            current_shell: 2,
            ..Default::default()
        };

        match evaluate_promotion(&config, &access_data) {
            PromotionResult::InsufficientHits { current, required } => {
                assert_eq!(current, 5);
                assert_eq!(required, 10);
            }
            _ => panic!("Expected InsufficientHits"),
        }

        // Enough hits
        let access_data = PointAccessData {
            hit_count: 15,
            current_shell: 2,
            ..Default::default()
        };

        match evaluate_promotion(&config, &access_data) {
            PromotionResult::Promoted {
                from_shell,
                to_shell,
            } => {
                assert_eq!(from_shell, 2);
                assert_eq!(to_shell, 1);
            }
            _ => panic!("Expected Promoted"),
        }
    }

    #[test]
    fn test_shell_move_operation() {
        use segment::types::PointIdType;

        let original = vec![1.0, 0.0, 0.0];
        let access_data = PointAccessData::new();
        let point_id = PointIdType::NumId(123);

        let op = ShellMoveOperation::demote(point_id, 1, 2, &original, access_data);

        assert_eq!(op.point_id, point_id);
        assert_eq!(op.delete_from, Some("rpi_shell_1".to_string()));

        let (insert_name, insert_vec) = op.insert_to.unwrap();
        assert_eq!(insert_name, "rpi_shell_2");
        // Vector should be scaled by 2
        assert_eq!(insert_vec, vec![2.0, 0.0, 0.0]);
    }
}
