use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroU32;
use std::path::Path;

use api::rest::SearchRequestInternal;
use collection::collection::Collection;
use collection::config::{CollectionConfigInternal, CollectionParams, WalConfig};
use collection::operations::CollectionUpdateOperations;
use collection::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorStructPersisted,
    WriteOrdering,
};
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::{PointRequestInternal, UpdateStatus, VectorsConfig};
use collection::operations::vector_ops::{
    PointVectorsPersisted, UpdateVectorsOp, VectorOperations,
};
use collection::operations::vector_params_builder::VectorParamsBuilder;
use collection::rpi::{self, RpiConfig};
use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, VectorStructInternal};
use segment::types::{Distance, PointIdType, WithPayloadInterface};
use shard::operations::point_ops::VectorPersisted;
use shard::search::CoreSearchRequestBatch;
use tempfile::Builder;
use tokio::time::{Duration, sleep};

use crate::common::{TEST_OPTIMIZERS_CONFIG, new_local_collection};

#[tokio::test(flavor = "multi_thread")]
async fn test_rpi_shell_progression_3_to_2_to_1() {
    let collection_dir = Builder::new().prefix("rpi-cache").tempdir().unwrap();
    let collection = rpi_collection_fixture(
        collection_dir.path(),
        RpiConfig {
            max_shells: 3,
            base_epsilon: 0.25,
            source_vector: None,
            demotion_threshold: 2,
            hnsw_for_shell_one: true,
            track_lru: true,
            promotion_threshold: 1,
            rebalance_threshold: 2.0,
        },
    )
    .await;
    let point_id = PointIdType::NumId(1);

    upsert_single_point(&collection, point_id, vec![1.0, 0.0, 0.0, 0.0]).await;

    // Force the point to start at shell 3 (no shell 1/shell 2 vectors).
    update_shell_vector(&collection, point_id, 3, vec![3.0, 0.0, 0.0, 0.0]).await;
    delete_shell_vectors(&collection, point_id, &[1, 2]).await;

    // 1st search: point should be found in shell 3 and then promoted to shell 2.
    let first = search_by_vector(&collection, vec![1.0, 0.0, 0.0, 0.0]).await;
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].id, point_id);
    wait_for_update_queue_to_drain(&collection).await;
    wait_for_shell_state(&collection, point_id, &[2], &[3]).await;

    // 2nd search: point should be found in shell 2 and then promoted to shell 1.
    let second = search_by_vector(&collection, vec![1.0, 0.0, 0.0, 0.0]).await;
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].id, point_id);
    wait_for_update_queue_to_drain(&collection).await;
    wait_for_shell_state(&collection, point_id, &[1], &[2]).await;

    // 3rd search: should hit shell 1 and stay there.
    let third = search_by_vector(&collection, vec![1.0, 0.0, 0.0, 0.0]).await;
    assert_eq!(third.len(), 1);
    assert_eq!(third[0].id, point_id);

    let stats = collection
        .info(&ShardSelectorInternal::All)
        .await
        .unwrap()
        .rpi_stats
        .expect("RPI stats must be present");
    assert!(
        stats.shell_hit_distribution[3] >= 1,
        "Expected a shell-3 hit"
    );
    assert!(
        stats.shell_hit_distribution[2] >= 1,
        "Expected a shell-2 hit"
    );
    assert!(
        stats.shell_hit_distribution[1] >= 1,
        "Expected a shell-1 hit"
    );

    let record = retrieve_point(&collection, point_id).await;
    let payload = record
        .payload
        .expect("RPI payload metadata must be present");
    let current_shell = payload
        .0
        .get(rpi::payload_fields::CURRENT_SHELL)
        .and_then(|v| v.as_u64())
        .expect("_rpi_current_shell should be a number");
    assert_eq!(current_shell, 1);

    collection.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rpi_high_shell_demotion_and_eviction() {
    let collection_dir = Builder::new().prefix("rpi-demote").tempdir().unwrap();
    let collection = rpi_collection_fixture(
        collection_dir.path(),
        RpiConfig {
            max_shells: 5,
            base_epsilon: 0.25,
            source_vector: None,
            demotion_threshold: 2,
            hnsw_for_shell_one: true,
            track_lru: true,
            promotion_threshold: 1000,
            rebalance_threshold: 2.0,
        },
    )
    .await;
    let point_id = PointIdType::NumId(2);

    upsert_single_point(&collection, point_id, vec![1.0, 0.0, 0.0, 0.0]).await;
    update_shell_vector(&collection, point_id, 4, vec![4.0, 0.0, 0.0, 0.0]).await;
    delete_shell_vectors(&collection, point_id, &[1, 2, 3, 5]).await;

    for _ in 0..100 {
        let results = search_by_vector(&collection, vec![1.0, 0.0, 0.0, 0.0]).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, point_id);
    }
    wait_for_update_queue_to_drain(&collection).await;
    wait_for_shell_state(&collection, point_id, &[5], &[4]).await;

    let shell_5_hit = search_by_vector(&collection, vec![1.0, 0.0, 0.0, 0.0]).await;
    assert_eq!(shell_5_hit.len(), 1);
    assert_eq!(shell_5_hit[0].id, point_id);
    wait_for_update_queue_to_drain(&collection).await;
    wait_for_current_shell(&collection, point_id, 6).await;

    let record = retrieve_point(&collection, point_id).await;
    if let Some(VectorStructInternal::Named(named)) = record.vector {
        let has_shell_vectors = (1..=5)
            .map(rpi::shell_vector_name)
            .any(|shell_name| named.contains_key(shell_name.as_str()));
        assert!(
            !has_shell_vectors,
            "Evicted point must not keep shell vectors"
        );
    }

    let miss = search_by_vector(&collection, vec![1.0, 0.0, 0.0, 0.0]).await;
    assert!(miss.is_empty());

    let stats = collection
        .info(&ShardSelectorInternal::All)
        .await
        .unwrap()
        .rpi_stats
        .expect("RPI stats must be present");
    assert!(stats.total_demotions >= 1, "Expected at least one demotion");
    assert!(stats.total_evictions >= 1, "Expected at least one eviction");

    collection.stop_gracefully().await;
}

async fn rpi_collection_fixture(path: &Path, rpi_config: RpiConfig) -> Collection {
    let wal_config = WalConfig {
        wal_capacity_mb: 1,
        wal_segments_ahead: 0,
        wal_retain_closed: 1,
    };

    let base_params = VectorParamsBuilder::new(4, Distance::Euclid).build();
    let mut named_vectors = BTreeMap::new();
    named_vectors.insert(DEFAULT_VECTOR_NAME.to_string().into(), base_params.clone());
    for shell in 1..=rpi_config.max_shells {
        let mut shell_params = base_params.clone();
        if shell > 1 {
            shell_params.hnsw_config = None;
        }
        named_vectors.insert(rpi::shell_vector_name(shell).into(), shell_params);
    }

    let collection_params = CollectionParams {
        vectors: VectorsConfig::Multi(named_vectors),
        shard_number: NonZeroU32::new(1).unwrap(),
        ..CollectionParams::empty()
    };

    let collection_config = CollectionConfigInternal {
        params: collection_params,
        optimizer_config: TEST_OPTIMIZERS_CONFIG.clone(),
        wal_config,
        hnsw_config: Default::default(),
        quantization_config: Default::default(),
        strict_mode_config: Default::default(),
        uuid: None,
        metadata: None,
        rpi_config: Some(rpi_config),
    };

    let snapshot_path = path.join("snapshots");
    new_local_collection(
        "rpi_test".to_string(),
        path,
        &snapshot_path,
        &collection_config,
    )
    .await
    .unwrap()
}

async fn upsert_single_point(collection: &Collection, point_id: PointIdType, vector: Vec<f32>) {
    let op = CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
        PointInsertOperationsInternal::PointsList(vec![PointStructPersisted {
            id: point_id,
            vector: VectorStructPersisted::Single(vector),
            payload: None,
        }]),
    ));

    let result = collection
        .update_from_client_simple(
            op,
            true,
            None,
            WriteOrdering::default(),
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(result.status, UpdateStatus::Completed);
}

async fn update_shell_vector(
    collection: &Collection,
    point_id: PointIdType,
    shell: u8,
    vector: Vec<f32>,
) {
    let mut vectors = HashMap::new();
    vectors.insert(
        rpi::shell_vector_name(shell).into(),
        VectorPersisted::Dense(vector),
    );

    let op = CollectionUpdateOperations::VectorOperation(VectorOperations::UpdateVectors(
        UpdateVectorsOp {
            points: vec![PointVectorsPersisted {
                id: point_id,
                vector: VectorStructPersisted::Named(vectors),
            }],
            update_filter: None,
        },
    ));

    let result = collection
        .update_from_client_simple(
            op,
            true,
            None,
            WriteOrdering::default(),
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(result.status, UpdateStatus::Completed);
}

async fn delete_shell_vectors(collection: &Collection, point_id: PointIdType, shells: &[u8]) {
    let vector_names = shells
        .iter()
        .map(|shell| rpi::shell_vector_name(*shell).into())
        .collect();

    let op = CollectionUpdateOperations::VectorOperation(VectorOperations::DeleteVectors(
        vec![point_id].into(),
        vector_names,
    ));

    let result = collection
        .update_from_client_simple(
            op,
            true,
            None,
            WriteOrdering::default(),
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(result.status, UpdateStatus::Completed);
}

async fn search_by_vector(
    collection: &Collection,
    vector: Vec<f32>,
) -> Vec<segment::types::ScoredPoint> {
    collection
        .core_search_batch(
            CoreSearchRequestBatch {
                searches: vec![
                    SearchRequestInternal {
                        vector: vector.into(),
                        with_payload: None,
                        with_vector: None,
                        filter: None,
                        params: None,
                        limit: 1,
                        offset: None,
                        score_threshold: None,
                    }
                    .into(),
                ],
            },
            None,
            ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap()
        .into_iter()
        .next()
        .unwrap_or_default()
}

async fn retrieve_point(
    collection: &Collection,
    point_id: PointIdType,
) -> shard::retrieve::record_internal::RecordInternal {
    collection
        .retrieve(
            PointRequestInternal {
                ids: vec![point_id],
                with_payload: Some(WithPayloadInterface::Bool(true)),
                with_vector: true.into(),
            },
            None,
            &ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap()
        .into_iter()
        .next()
        .expect("Point should exist")
}

async fn wait_for_shell_state(
    collection: &Collection,
    point_id: PointIdType,
    expected_present: &[u8],
    expected_absent: &[u8],
) {
    for _ in 0..400 {
        let record = retrieve_point(collection, point_id).await;
        let Some(VectorStructInternal::Named(named)) = record.vector else {
            sleep(Duration::from_millis(25)).await;
            continue;
        };

        let has_expected_present = expected_present
            .iter()
            .all(|shell| named.contains_key(rpi::shell_vector_name(*shell).as_str()));
        let has_expected_absent = expected_absent
            .iter()
            .all(|shell| !named.contains_key(rpi::shell_vector_name(*shell).as_str()));

        if has_expected_present && has_expected_absent {
            return;
        }

        sleep(Duration::from_millis(25)).await;
    }

    let record = retrieve_point(collection, point_id).await;
    let vectors = match record.vector {
        Some(VectorStructInternal::Named(named)) => named.keys().map(|k| k.to_string()).collect(),
        Some(VectorStructInternal::Single(_)) => vec!["<single>".to_string()],
        Some(VectorStructInternal::MultiDense(_)) => vec!["<multi_dense>".to_string()],
        None => vec!["<none>".to_string()],
    };
    let current_shell = record
        .payload
        .as_ref()
        .and_then(|p| p.0.get(rpi::payload_fields::CURRENT_SHELL))
        .and_then(|v| v.as_u64());
    panic!(
        "Timed out waiting for expected RPI shell state. vectors={vectors:?}, current_shell={current_shell:?}"
    );
}

async fn wait_for_update_queue_to_drain(collection: &Collection) {
    for _ in 0..200 {
        let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
        let Some(update_queue) = info.update_queue else {
            return;
        };

        let deferred = update_queue.deferred_points.unwrap_or(0);
        if update_queue.length == 0 && deferred == 0 {
            return;
        }

        sleep(Duration::from_millis(25)).await;
    }
}

async fn wait_for_current_shell(collection: &Collection, point_id: PointIdType, expected: u64) {
    for _ in 0..200 {
        let record = retrieve_point(collection, point_id).await;
        let current_shell = record
            .payload
            .as_ref()
            .and_then(|p| p.0.get(rpi::payload_fields::CURRENT_SHELL))
            .and_then(|v| v.as_u64());
        if current_shell == Some(expected) {
            return;
        }
        sleep(Duration::from_millis(25)).await;
    }

    panic!("Timed out waiting for _rpi_current_shell={expected}");
}
