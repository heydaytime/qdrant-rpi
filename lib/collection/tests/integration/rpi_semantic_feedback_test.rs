use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::path::Path;

use api::rest::SearchRequestInternal;
use collection::collection::Collection;
use collection::config::{CollectionConfigInternal, CollectionParams, WalConfig};
use collection::operations::CollectionUpdateOperations;
use collection::operations::point_ops::{
    BatchPersisted, BatchVectorStructPersisted, PointInsertOperationsInternal, PointOperations,
    WriteOrdering,
};
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::{PointRequestInternal, UpdateStatus, VectorsConfig};
use collection::operations::vector_params_builder::VectorParamsBuilder;
use collection::rpi::{self, RpiConfig};
use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, VectorStructInternal};
use segment::types::{Distance, PointIdType, WithPayloadInterface, WithVector};
use shard::search::CoreSearchRequestBatch;
use tempfile::Builder;
use tokio::time::{Duration, sleep};

use crate::common::{TEST_OPTIMIZERS_CONFIG, new_local_collection};

#[tokio::test(flavor = "multi_thread")]
async fn test_semantic_feedback_demotes_bad_candidate_vs_default() {
    let vectors = vec![
        vec![0.90, 0.20, 0.0, 0.0], // good fact: semantically correct but slightly farther
        vec![1.00, 0.00, 0.0, 0.0], // bad fact: embedding-closer but semantically wrong
        vec![0.00, 1.00, 0.0, 0.0], // distractor
    ];
    let query = vec![1.00, 0.00, 0.0, 0.0];
    let good_id = PointIdType::NumId(0);
    let bad_id = PointIdType::NumId(1);

    let baseline_dir = Builder::new().prefix("rpi-baseline-sem").tempdir().unwrap();
    let baseline = collection_fixture(baseline_dir.path(), None).await;
    upsert_batch(&baseline, &vectors).await;

    let mut baseline_good_top1 = 0usize;
    for _ in 0..12 {
        let top = search_top_ids(&baseline, query.clone(), 2).await;
        if top.first().copied() == Some(good_id) {
            baseline_good_top1 += 1;
        }
    }

    let rpi_dir = Builder::new().prefix("rpi-feedback-sem").tempdir().unwrap();
    let rpi_collection = collection_fixture(
        rpi_dir.path(),
        Some(RpiConfig {
            max_shells: 3,
            base_epsilon: 0.5,
            source_vector: None,
            demotion_threshold: 1,
            hnsw_for_shell_one: true,
            track_lru: true,
            promotion_threshold: 1,
            rebalance_threshold: 2.0,
        }),
    )
    .await;
    upsert_batch(&rpi_collection, &vectors).await;

    let mut rpi_good_top1_history = Vec::new();
    for _round in 0..12 {
        let shown = search_top_ids(&rpi_collection, query.clone(), 2).await;
        rpi_good_top1_history.push(shown.first().copied() == Some(good_id));
        if shown.contains(&good_id) {
            rpi_collection
                .rpi_apply_feedback(&shown, good_id, &ShardSelectorInternal::All)
                .await
                .unwrap();
        }
        wait_for_update_queue_to_drain(&rpi_collection).await;
    }

    let final_rpi = search_top_ids(&rpi_collection, query.clone(), 2).await;
    assert_eq!(
        final_rpi.first().copied(),
        Some(good_id),
        "RPI should converge so the semantically good point is top-1"
    );

    let final_baseline = search_top_ids(&baseline, query, 2).await;
    assert_eq!(
        final_baseline.first().copied(),
        Some(bad_id),
        "Default behavior keeps returning embedding-closest bad point"
    );

    let last_four_good = rpi_good_top1_history
        .iter()
        .rev()
        .take(4)
        .filter(|&&v| v)
        .count();

    let rpi_good_total = rpi_good_top1_history.iter().filter(|&&v| v).count();
    println!(
        "semantic-feedback comparison: baseline_good_top1={} rpi_good_top1={} last4_good={} final_baseline={:?} final_rpi={:?}",
        baseline_good_top1, rpi_good_total, last_four_good, final_baseline, final_rpi
    );

    assert!(
        last_four_good >= 3,
        "RPI should keep good point at top in late rounds"
    );
    assert!(
        baseline_good_top1 == 0,
        "Baseline should not self-correct without feedback"
    );

    let bad_record = retrieve_point(&rpi_collection, bad_id).await;
    if let Some(VectorStructInternal::Named(named)) = bad_record.vector {
        assert!(
            !named.contains_key(rpi::shell_vector_name(1).as_str()),
            "Bad point should be demoted out of shell 1"
        );
    }

    baseline.stop_gracefully().await;
    rpi_collection.stop_gracefully().await;
}

async fn collection_fixture(path: &Path, rpi_config: Option<RpiConfig>) -> Collection {
    let wal_config = WalConfig {
        wal_capacity_mb: 1,
        wal_segments_ahead: 0,
        wal_retain_closed: 1,
    };

    let vectors_config = if let Some(cfg) = &rpi_config {
        let base = VectorParamsBuilder::new(4, Distance::Euclid).build();
        let mut named = BTreeMap::new();
        named.insert(DEFAULT_VECTOR_NAME.to_string().into(), base.clone());
        for shell in 1..=cfg.max_shells {
            let mut params = base.clone();
            if shell > 1 {
                params.hnsw_config = None;
            }
            named.insert(rpi::shell_vector_name(shell).into(), params);
        }
        VectorsConfig::Multi(named)
    } else {
        VectorParamsBuilder::new(4, Distance::Euclid).build().into()
    };

    let collection_params = CollectionParams {
        vectors: vectors_config,
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
        rpi_config,
    };

    let snapshot_path = path.join("snapshots");
    new_local_collection(
        "rpi_semantics".to_string(),
        path,
        &snapshot_path,
        &collection_config,
    )
    .await
    .unwrap()
}

async fn upsert_batch(collection: &Collection, vectors: &[Vec<f32>]) {
    let ids = (0..vectors.len())
        .map(|i| PointIdType::NumId(i as u64))
        .collect();

    let batch = BatchPersisted {
        ids,
        vectors: BatchVectorStructPersisted::Single(vectors.to_vec()),
        payloads: None,
    };

    let op = CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
        PointInsertOperationsInternal::from(batch),
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

async fn search_top_ids(
    collection: &Collection,
    vector: Vec<f32>,
    limit: usize,
) -> Vec<PointIdType> {
    let result = collection
        .core_search_batch(
            CoreSearchRequestBatch {
                searches: vec![
                    SearchRequestInternal {
                        vector: vector.into(),
                        with_payload: None,
                        with_vector: None,
                        filter: None,
                        params: None,
                        limit,
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
        .unwrap();

    result[0].iter().map(|p| p.id).collect()
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
                with_vector: WithVector::Bool(true),
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
        .unwrap()
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
