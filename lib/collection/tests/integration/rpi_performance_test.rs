use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroU32;
use std::path::Path;
use std::time::Instant;

use api::rest::SearchRequestInternal;
use collection::collection::Collection;
use collection::config::{CollectionConfigInternal, CollectionParams, WalConfig};
use collection::operations::CollectionUpdateOperations;
use collection::operations::point_ops::{
    BatchPersisted, BatchVectorStructPersisted, PointInsertOperationsInternal, PointOperations,
    WriteOrdering,
};
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::{UpdateStatus, VectorsConfig};
use collection::operations::vector_ops::{PointVectorsPersisted, UpdateVectorsOp, VectorOperations};
use collection::operations::vector_params_builder::VectorParamsBuilder;
use collection::rpi::{self, RpiConfig};
use common::counter::hardware_accumulator::HwMeasurementAcc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use segment::data_types::vectors::DEFAULT_VECTOR_NAME;
use segment::types::{Distance, PointIdType};
use shard::operations::point_ops::{PointIdsList, VectorPersisted, VectorStructPersisted};
use shard::search::CoreSearchRequestBatch;
use tempfile::Builder;
use tokio::time::{Duration, sleep};

use crate::common::{TEST_OPTIMIZERS_CONFIG, new_local_collection};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "performance comparison test; run manually"]
async fn bench_rpi_vs_default_behavior() {
    const DIM: usize = 32;
    const POINTS: usize = 6_000;
    const QUERY_COUNT: usize = 300;
    const WARMUP: usize = 100;

    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<Vec<f32>> = (0..POINTS)
        .map(|_| (0..DIM).map(|_| rng.random::<f32>()).collect())
        .collect();

    let query_ids: Vec<usize> = (0..300).step_by(3).collect();
    let queries: Vec<Vec<f32>> = query_ids.iter().map(|&i| vectors[i].clone()).collect();

    let base_dir = Builder::new().prefix("rpi-perf-default").tempdir().unwrap();
    let base_collection = collection_fixture(base_dir.path(), None).await;
    upsert_batch(&base_collection, &vectors).await;
    run_queries(&base_collection, &queries, WARMUP).await;
    let baseline = run_queries(&base_collection, &queries, QUERY_COUNT).await;

    let rpi_cfg_shell1 = RpiConfig {
        max_shells: 3,
        base_epsilon: 0.25,
        source_vector: None,
        demotion_threshold: 2,
        hnsw_for_shell_one: true,
        track_lru: true,
        promotion_threshold: 1,
        rebalance_threshold: 2.0,
    };
    let shell1_dir = Builder::new().prefix("rpi-perf-shell1").tempdir().unwrap();
    let shell1_collection = collection_fixture(shell1_dir.path(), Some(rpi_cfg_shell1.clone())).await;
    upsert_batch(&shell1_collection, &vectors).await;
    run_queries(&shell1_collection, &queries, WARMUP).await;
    let rpi_shell1 = run_queries(&shell1_collection, &queries, QUERY_COUNT).await;

    let converge_dir = Builder::new().prefix("rpi-perf-converge").tempdir().unwrap();
    let converge_collection =
        collection_fixture(converge_dir.path(), Some(rpi_cfg_shell1.clone())).await;
    upsert_batch(&converge_collection, &vectors).await;

    force_hot_points_to_shell(
        &converge_collection,
        &query_ids,
        &vectors,
        3,
        &[1, 2],
    )
    .await;

    let rpi_early = run_queries(&converge_collection, &queries, QUERY_COUNT / 2).await;
    wait_for_update_queue_to_drain(&converge_collection).await;
    let rpi_late = run_queries(&converge_collection, &queries, QUERY_COUNT / 2).await;
    wait_for_update_queue_to_drain(&converge_collection).await;
    let rpi_steady = run_queries(&converge_collection, &queries, QUERY_COUNT / 2).await;

    let b = summarize("baseline_default", &baseline);
    let s1 = summarize("rpi_shell1", &rpi_shell1);
    let e = summarize("rpi_early_shell3", &rpi_early);
    let l = summarize("rpi_late_after_promotion", &rpi_late);
    let steady = summarize("rpi_steady_shell1", &rpi_steady);

    println!(
        "\nRPI Perf Summary\n  {}\n  {}\n  {}\n  {}\n  {}\n",
        b, s1, e, l, steady
    );

    let shell1_delta = percent_delta(avg_ms(&rpi_shell1), avg_ms(&baseline));
    let late_vs_base = percent_delta(avg_ms(&rpi_late), avg_ms(&baseline));
    let steady_vs_base = percent_delta(avg_ms(&rpi_steady), avg_ms(&baseline));
    let convergence_gain = percent_delta(avg_ms(&rpi_late), avg_ms(&rpi_early));

    println!(
        "Delta vs baseline: shell1={:+.2}%, late={:+.2}%, steady={:+.2}%, convergence_gain={:+.2}%",
        shell1_delta, late_vs_base, steady_vs_base, convergence_gain
    );

    base_collection.stop_gracefully().await;
    shell1_collection.stop_gracefully().await;
    converge_collection.stop_gracefully().await;
}

async fn collection_fixture(path: &Path, rpi_config: Option<RpiConfig>) -> Collection {
    let wal_config = WalConfig {
        wal_capacity_mb: 1,
        wal_segments_ahead: 0,
        wal_retain_closed: 1,
    };

    let vectors_config = if let Some(cfg) = &rpi_config {
        let base = VectorParamsBuilder::new(32, Distance::Euclid).build();
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
        VectorParamsBuilder::new(32, Distance::Euclid).build().into()
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
        "rpi_perf".to_string(),
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
        .update_from_client_simple(op, true, None, WriteOrdering::default(), HwMeasurementAcc::new())
        .await
        .unwrap();
    assert_eq!(result.status, UpdateStatus::Completed);
}

async fn run_queries(collection: &Collection, queries: &[Vec<f32>], count: usize) -> Vec<f64> {
    let mut times = Vec::with_capacity(count);
    for i in 0..count {
        let query = queries[i % queries.len()].clone();

        let start = Instant::now();
        let result = collection
            .core_search_batch(
                CoreSearchRequestBatch {
                    searches: vec![SearchRequestInternal {
                        vector: query.into(),
                        with_payload: None,
                        with_vector: None,
                        filter: None,
                        params: None,
                        limit: 5,
                        offset: None,
                        score_threshold: None,
                    }
                    .into()],
                },
                None,
                ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::new(),
            )
            .await
            .unwrap();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        assert!(!result[0].is_empty());
        times.push(elapsed);
    }
    times
}

async fn force_hot_points_to_shell(
    collection: &Collection,
    hot_ids: &[usize],
    vectors: &[Vec<f32>],
    shell: u8,
    remove_shells: &[u8],
) {
    let mut points = Vec::with_capacity(hot_ids.len());
    for &id in hot_ids {
        let mut named = HashMap::new();
        named.insert(
            rpi::shell_vector_name(shell).into(),
            VectorPersisted::Dense(rpi::scale_vector(&vectors[id], shell)),
        );
        points.push(PointVectorsPersisted {
            id: PointIdType::NumId(id as u64),
            vector: VectorStructPersisted::Named(named),
        });
    }

    let update = CollectionUpdateOperations::VectorOperation(VectorOperations::UpdateVectors(
        UpdateVectorsOp {
            points,
            update_filter: None,
        },
    ));
    let result = collection
        .update_from_client_simple(
            update,
            true,
            None,
            WriteOrdering::default(),
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(result.status, UpdateStatus::Completed);

    let remove_vector_names = remove_shells
        .iter()
        .map(|s| rpi::shell_vector_name(*s).into())
        .collect();
    let remove_ids = hot_ids
        .iter()
        .map(|id| PointIdType::NumId(*id as u64))
        .collect();

    let delete = CollectionUpdateOperations::VectorOperation(VectorOperations::DeleteVectors(
        PointIdsList {
            points: remove_ids,
            shard_key: None,
        },
        remove_vector_names,
    ));
    let result = collection
        .update_from_client_simple(
            delete,
            true,
            None,
            WriteOrdering::default(),
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(result.status, UpdateStatus::Completed);
}

async fn wait_for_update_queue_to_drain(collection: &Collection) {
    for _ in 0..300 {
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

fn avg_ms(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
}

fn percentile_ms(samples: &[f64], percentile: f64) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[idx]
}

fn summarize(name: &str, samples: &[f64]) -> String {
    format!(
        "{}: avg={:.3}ms p50={:.3}ms p95={:.3}ms",
        name,
        avg_ms(samples),
        percentile_ms(samples, 0.50),
        percentile_ms(samples, 0.95)
    )
}

fn percent_delta(new: f64, baseline: f64) -> f64 {
    if baseline == 0.0 {
        return 0.0;
    }
    ((new - baseline) / baseline) * 100.0
}
