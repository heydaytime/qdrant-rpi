//! RPI operation transformation
//!
//! This module provides functions to transform operations for RPI.
//! When RPI is enabled, insert operations need to add scaled shell vectors.

use std::collections::HashMap;

use segment::data_types::vectors::DEFAULT_VECTOR_NAME;
use shard::operations::point_ops::{
    BatchPersisted, BatchVectorStructPersisted, PointInsertOperationsInternal, PointOperations,
    PointStructPersisted, VectorPersisted, VectorStructPersisted,
};

use super::{RpiConfig, scale_vector, shell_vector_name};

/// Transform a PointOperations for RPI.
///
/// For UpsertPoints operations, this adds shell vectors for each point:
/// - New points get a scaled vector at shell 1 (highest quality)
/// - The source vector is scaled by k=1 and stored as rpi_shell_1
///
/// Returns the modified operation.
pub fn transform_point_operation_for_rpi(
    operation: PointOperations,
    rpi_config: &RpiConfig,
) -> PointOperations {
    match operation {
        PointOperations::UpsertPoints(insert_op) => {
            let transformed = transform_insert_operation(insert_op, rpi_config);
            PointOperations::UpsertPoints(transformed)
        }
        PointOperations::UpsertPointsConditional(conditional) => {
            // Transform the inner points operation
            let transformed_points = transform_insert_operation(conditional.points_op, rpi_config);
            PointOperations::UpsertPointsConditional(
                shard::operations::point_ops::ConditionalInsertOperationInternal {
                    points_op: transformed_points,
                    condition: conditional.condition,
                    update_mode: conditional.update_mode,
                },
            )
        }
        // Other operations don't need transformation
        other => other,
    }
}

/// Transform an insert operation to add RPI shell vectors.
fn transform_insert_operation(
    operation: PointInsertOperationsInternal,
    rpi_config: &RpiConfig,
) -> PointInsertOperationsInternal {
    match operation {
        PointInsertOperationsInternal::PointsBatch(batch) => {
            let transformed = transform_batch(batch, rpi_config);
            PointInsertOperationsInternal::PointsBatch(transformed)
        }
        PointInsertOperationsInternal::PointsList(points) => {
            let transformed: Vec<_> = points
                .into_iter()
                .map(|p| transform_point(p, rpi_config))
                .collect();
            PointInsertOperationsInternal::PointsList(transformed)
        }
    }
}

/// Transform a batch insert to add RPI shell vectors.
fn transform_batch(batch: BatchPersisted, rpi_config: &RpiConfig) -> BatchPersisted {
    let source_name = rpi_config
        .source_vector
        .as_deref()
        .unwrap_or(DEFAULT_VECTOR_NAME);

    let shell_1_name = shell_vector_name(1);

    match batch.vectors {
        BatchVectorStructPersisted::Single(vectors) => {
            // Single vector mode: scale each vector for shell 1
            let shell_vectors: Vec<VectorPersisted> = vectors
                .iter()
                .map(|v| VectorPersisted::Dense(scale_vector(v, 1)))
                .collect();

            // Convert to named vectors with both original and shell
            let mut named: HashMap<String, Vec<VectorPersisted>> = HashMap::new();
            named.insert(
                DEFAULT_VECTOR_NAME.to_string(),
                vectors.into_iter().map(VectorPersisted::Dense).collect(),
            );
            named.insert(shell_1_name, shell_vectors);

            BatchPersisted {
                ids: batch.ids,
                vectors: BatchVectorStructPersisted::Named(named),
                payloads: batch.payloads,
            }
        }
        BatchVectorStructPersisted::MultiDense(vectors) => {
            // Multi-dense vectors: we only support scaling the first vector in each set
            // This is a limitation - multi-dense RPI would need more complex handling
            log::warn!(
                "RPI with multi-dense vectors: only the first vector in each set will be used for shell scaling"
            );

            let shell_vectors: Vec<VectorPersisted> = vectors
                .iter()
                .map(|multi| {
                    if let Some(first) = multi.first() {
                        VectorPersisted::Dense(scale_vector(first, 1))
                    } else {
                        VectorPersisted::Dense(vec![])
                    }
                })
                .collect();

            // Convert to named vectors
            let mut named: HashMap<String, Vec<VectorPersisted>> = HashMap::new();
            named.insert(
                DEFAULT_VECTOR_NAME.to_string(),
                vectors
                    .into_iter()
                    .map(VectorPersisted::MultiDense)
                    .collect(),
            );
            named.insert(shell_1_name, shell_vectors);

            BatchPersisted {
                ids: batch.ids,
                vectors: BatchVectorStructPersisted::Named(named),
                payloads: batch.payloads,
            }
        }
        BatchVectorStructPersisted::Named(mut named_vectors) => {
            // Named vectors: find source vector and add shell vector
            if let Some(source_vectors) = named_vectors.get(source_name) {
                let shell_vectors: Vec<VectorPersisted> = source_vectors
                    .iter()
                    .map(|v| match v {
                        VectorPersisted::Dense(dense) => {
                            VectorPersisted::Dense(scale_vector(dense, 1))
                        }
                        VectorPersisted::MultiDense(multi) => {
                            // Use first vector from multi-dense
                            if let Some(first) = multi.first() {
                                VectorPersisted::Dense(scale_vector(first, 1))
                            } else {
                                VectorPersisted::Dense(vec![])
                            }
                        }
                        VectorPersisted::Sparse(_) => {
                            // Sparse vectors can't be used for RPI
                            log::warn!("RPI source vector is sparse - shell vector will be empty");
                            VectorPersisted::Dense(vec![])
                        }
                    })
                    .collect();

                named_vectors.insert(shell_1_name, shell_vectors);
            } else {
                log::warn!(
                    "RPI source vector '{source_name}' not found in batch - skipping shell vector creation"
                );
            }

            BatchPersisted {
                ids: batch.ids,
                vectors: BatchVectorStructPersisted::Named(named_vectors),
                payloads: batch.payloads,
            }
        }
    }
}

/// Transform a single point to add RPI shell vector.
fn transform_point(
    mut point: PointStructPersisted,
    rpi_config: &RpiConfig,
) -> PointStructPersisted {
    let source_name = rpi_config
        .source_vector
        .as_deref()
        .unwrap_or(DEFAULT_VECTOR_NAME);

    let shell_1_name = shell_vector_name(1);

    match &point.vector {
        VectorStructPersisted::Single(vector) => {
            // Single vector: scale it and convert to named
            let shell_vector = VectorPersisted::Dense(scale_vector(vector, 1));

            let mut named: HashMap<String, VectorPersisted> = HashMap::new();
            named.insert(
                DEFAULT_VECTOR_NAME.to_string(),
                VectorPersisted::Dense(vector.clone()),
            );
            named.insert(shell_1_name, shell_vector);

            point.vector = VectorStructPersisted::Named(named);
        }
        VectorStructPersisted::MultiDense(vectors) => {
            // Multi-dense: use first vector for shell
            let shell_vector = if let Some(first) = vectors.first() {
                VectorPersisted::Dense(scale_vector(first, 1))
            } else {
                VectorPersisted::Dense(vec![])
            };

            let mut named: HashMap<String, VectorPersisted> = HashMap::new();
            named.insert(
                DEFAULT_VECTOR_NAME.to_string(),
                VectorPersisted::MultiDense(vectors.clone()),
            );
            named.insert(shell_1_name, shell_vector);

            point.vector = VectorStructPersisted::Named(named);
        }
        VectorStructPersisted::Named(vectors) => {
            // Named vectors: find source and add shell
            if let Some(source_vector) = vectors.get(source_name) {
                let shell_vector = match source_vector {
                    VectorPersisted::Dense(dense) => VectorPersisted::Dense(scale_vector(dense, 1)),
                    VectorPersisted::MultiDense(multi) => {
                        if let Some(first) = multi.first() {
                            VectorPersisted::Dense(scale_vector(first, 1))
                        } else {
                            VectorPersisted::Dense(vec![])
                        }
                    }
                    VectorPersisted::Sparse(_) => {
                        log::warn!("RPI source vector is sparse - shell vector will be empty");
                        VectorPersisted::Dense(vec![])
                    }
                };

                let mut new_vectors = vectors.clone();
                new_vectors.insert(shell_1_name, shell_vector);
                point.vector = VectorStructPersisted::Named(new_vectors);
            } else {
                log::warn!(
                    "RPI source vector '{}' not found in point {} - skipping shell vector creation",
                    source_name,
                    point.id
                );
            }
        }
    }

    point
}

#[cfg(test)]
mod tests {
    use segment::types::PointIdType;

    use super::*;

    fn make_test_config() -> RpiConfig {
        RpiConfig {
            max_shells: 5,
            base_epsilon: 0.1,
            source_vector: None,
            demotion_threshold: 2,
            hnsw_for_shell_one: true,
            track_lru: true,
            promotion_threshold: 10,
            rebalance_threshold: 3.0,
        }
    }

    #[test]
    fn test_transform_single_point() {
        let config = make_test_config();

        let point = PointStructPersisted {
            id: PointIdType::NumId(1),
            vector: VectorStructPersisted::Single(vec![1.0, 0.0, 0.0]),
            payload: None,
        };

        let transformed = transform_point(point, &config);

        // Should now be named vectors
        match &transformed.vector {
            VectorStructPersisted::Named(named) => {
                // Should have original vector
                assert!(named.contains_key(DEFAULT_VECTOR_NAME));

                // Should have shell 1 vector
                let shell_1_name = shell_vector_name(1);
                assert!(named.contains_key(&shell_1_name));

                // Shell vector should be scaled by k=1 (unchanged)
                if let Some(VectorPersisted::Dense(shell_vec)) = named.get(&shell_1_name) {
                    assert_eq!(shell_vec, &vec![1.0, 0.0, 0.0]);
                } else {
                    panic!("Expected dense shell vector");
                }
            }
            _ => panic!("Expected named vectors after transform"),
        }
    }

    #[test]
    fn test_transform_batch() {
        let config = make_test_config();

        let batch = BatchPersisted {
            ids: vec![PointIdType::NumId(1), PointIdType::NumId(2)],
            vectors: BatchVectorStructPersisted::Single(vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
            ]),
            payloads: None,
        };

        let transformed = transform_batch(batch, &config);

        // Should now be named vectors
        match &transformed.vectors {
            BatchVectorStructPersisted::Named(named) => {
                // Should have original vectors
                assert!(named.contains_key(DEFAULT_VECTOR_NAME));
                assert_eq!(named.get(DEFAULT_VECTOR_NAME).unwrap().len(), 2);

                // Should have shell 1 vectors
                let shell_1_name = shell_vector_name(1);
                assert!(named.contains_key(&shell_1_name));
                assert_eq!(named.get(&shell_1_name).unwrap().len(), 2);
            }
            _ => panic!("Expected named vectors after transform"),
        }
    }
}
