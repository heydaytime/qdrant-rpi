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
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngExt;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, VectorStructInternal};
use segment::types::{Distance, PointIdType, WithPayloadInterface, WithVector};
use shard::search::CoreSearchRequestBatch;
use tempfile::Builder;
use tokio::time::{Duration, sleep};

use crate::common::{TEST_OPTIMIZERS_CONFIG, new_local_collection};

#[derive(Clone)]
struct Intent {
    query: Vec<f32>,
    good_id: PointIdType,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "large scale quality benchmark; run manually"]
async fn bench_rpi_quality_at_scale_vs_default() {
    const DIM: usize = 32;
    const INTENTS: usize = 1_500;
    const BAD_PER_INTENT: usize = 5;
    const EPOCHS: usize = 5;
    const TOP_K: usize = BAD_PER_INTENT + 1;

    let (vectors, intents) = generate_semantic_confusion_dataset(DIM, INTENTS, BAD_PER_INTENT);

    let baseline_dir = Builder::new().prefix("rpi-scale-base").tempdir().unwrap();
    let baseline = collection_fixture(baseline_dir.path(), None, DIM).await;
    upsert_batch(&baseline, &vectors).await;

    let rpi_dir = Builder::new().prefix("rpi-scale-rpi").tempdir().unwrap();
    let rpi_collection = collection_fixture(
        rpi_dir.path(),
        Some(RpiConfig {
            max_shells: 6,
            base_epsilon: 0.6,
            source_vector: None,
            demotion_threshold: 1,
            hnsw_for_shell_one: true,
            track_lru: true,
            promotion_threshold: 1,
            rebalance_threshold: 1.8,
        }),
        DIM,
    )
    .await;
    upsert_batch(&rpi_collection, &vectors).await;

    let noisy_dir = Builder::new().prefix("rpi-scale-noisy").tempdir().unwrap();
    let rpi_noisy = collection_fixture(
        noisy_dir.path(),
        Some(RpiConfig {
            max_shells: 6,
            base_epsilon: 0.6,
            source_vector: None,
            demotion_threshold: 1,
            hnsw_for_shell_one: true,
            track_lru: true,
            promotion_threshold: 1,
            rebalance_threshold: 1.8,
        }),
        DIM,
    )
    .await;
    upsert_batch(&rpi_noisy, &vectors).await;

    let mut baseline_epoch_acc = Vec::with_capacity(EPOCHS);
    let mut rpi_epoch_acc = Vec::with_capacity(EPOCHS);
    let mut noisy_epoch_acc = Vec::with_capacity(EPOCHS);
    let mut noisy_rng = StdRng::seed_from_u64(99);

    for epoch in 0..EPOCHS {
        let baseline_acc = evaluate_top1_accuracy(&baseline, &intents, TOP_K).await;
        baseline_epoch_acc.push(baseline_acc);

        let rpi_acc = train_with_feedback_epoch(&rpi_collection, &intents, TOP_K).await;
        rpi_epoch_acc.push(rpi_acc);
        wait_for_update_queue_to_drain(&rpi_collection).await;

        let noisy_acc = train_with_feedback_epoch_noisy(
            &rpi_noisy,
            &intents,
            TOP_K,
            0.85,
            &mut noisy_rng,
        )
        .await;
        noisy_epoch_acc.push(noisy_acc);
        wait_for_update_queue_to_drain(&rpi_noisy).await;

        println!(
            "epoch={} baseline_top1={:.4} rpi_oracle_top1={:.4} rpi_noisy_top1={:.4}",
            epoch + 1,
            baseline_acc,
            rpi_acc,
            noisy_acc
        );
    }

    let baseline_final = evaluate_top1_accuracy(&baseline, &intents, TOP_K).await;
    let rpi_final = evaluate_top1_accuracy(&rpi_collection, &intents, TOP_K).await;
    let noisy_final = evaluate_top1_accuracy(&rpi_noisy, &intents, TOP_K).await;

    let shell1_stats = sample_shell1_purity(&rpi_collection, &intents).await;
    let shell1_stats_noisy = sample_shell1_purity(&rpi_noisy, &intents).await;

    println!(
        "scale-quality summary: baseline_final={:.4} rpi_oracle_final={:.4} rpi_noisy_final={:.4} abs_gain_oracle={:+.4} abs_gain_noisy={:+.4} rel_gain_oracle={:+.2}% rel_gain_noisy={:+.2}% shell1_good_oracle={:.4} shell1_bad_oracle={:.4} shell1_good_noisy={:.4} shell1_bad_noisy={:.4}",
        baseline_final,
        rpi_final,
        noisy_final,
        rpi_final - baseline_final,
        noisy_final - baseline_final,
        pct_gain(rpi_final, baseline_final),
        pct_gain(noisy_final, baseline_final),
        shell1_stats.0,
        shell1_stats.1,
        shell1_stats_noisy.0,
        shell1_stats_noisy.1
    );

    assert!(baseline_final < 0.25, "Baseline should stay confused at scale");
    assert!(
        rpi_final > 0.80,
        "RPI should converge to high top-1 quality after feedback"
    );
    assert!(
        noisy_final > 0.60,
        "RPI with noisy feedback should still beat baseline clearly"
    );
    assert!(
        rpi_final > baseline_final,
        "RPI should outperform baseline on final top-1"
    );
    assert!(
        noisy_final > baseline_final,
        "RPI with noisy feedback should outperform baseline"
    );

    baseline.stop_gracefully().await;
    rpi_collection.stop_gracefully().await;
    rpi_noisy.stop_gracefully().await;
}

fn generate_semantic_confusion_dataset(
    dim: usize,
    intents: usize,
    bad_per_intent: usize,
) -> (Vec<Vec<f32>>, Vec<Intent>) {
    let mut rng = StdRng::seed_from_u64(2026);
    let mut vectors = Vec::with_capacity(intents * (1 + bad_per_intent));
    let mut intent_data = Vec::with_capacity(intents);

    for intent_idx in 0..intents {
        let base: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        // Good point is semantically right and moderately close.
        let good_noise = rng.random_range(0.020..0.045);
        let good = perturb(&base, good_noise, &mut rng);
        let good_id = PointIdType::NumId((intent_idx * (1 + bad_per_intent)) as u64);
        vectors.push(good);

        // Bad points are semantically wrong and often, but not always, embedding-closer.
        for _ in 0..bad_per_intent {
            let bad_noise = rng.random_range(0.015..0.055);
            vectors.push(perturb(&base, bad_noise, &mut rng));
        }

        intent_data.push(Intent {
            query: base,
            good_id,
        });
    }

    (vectors, intent_data)
}

fn perturb(base: &[f32], noise: f32, rng: &mut StdRng) -> Vec<f32> {
    base.iter()
        .map(|v| v + rng.random_range(-noise..noise))
        .collect()
}

async fn collection_fixture(
    path: &Path,
    rpi_config: Option<RpiConfig>,
    dim: usize,
) -> Collection {
    let wal_config = WalConfig {
        wal_capacity_mb: 1,
        wal_segments_ahead: 0,
        wal_retain_closed: 1,
    };

    let vectors_config = if let Some(cfg) = &rpi_config {
        let base = VectorParamsBuilder::new(dim as u64, Distance::Euclid).build();
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
        VectorParamsBuilder::new(dim as u64, Distance::Euclid).build().into()
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
        "rpi_scale".to_string(),
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

async fn evaluate_top1_accuracy(collection: &Collection, intents: &[Intent], top_k: usize) -> f32 {
    let mut hits = 0usize;
    for intent in intents {
        let top_ids = search_top_ids(collection, intent.query.clone(), top_k).await;
        if top_ids.first().copied() == Some(intent.good_id) {
            hits += 1;
        }
    }
    hits as f32 / intents.len() as f32
}

async fn train_with_feedback_epoch(collection: &Collection, intents: &[Intent], top_k: usize) -> f32 {
    let mut hits = 0usize;
    for intent in intents {
        let shown = search_top_ids(collection, intent.query.clone(), top_k).await;
        if shown.first().copied() == Some(intent.good_id) {
            hits += 1;
        }

        if shown.contains(&intent.good_id) {
            collection
                .rpi_apply_feedback(&shown, intent.good_id, &ShardSelectorInternal::All)
                .await
                .unwrap();
        }
    }
    hits as f32 / intents.len() as f32
}

async fn train_with_feedback_epoch_noisy(
    collection: &Collection,
    intents: &[Intent],
    top_k: usize,
    feedback_accuracy: f32,
    rng: &mut StdRng,
) -> f32 {
    let mut hits = 0usize;
    for intent in intents {
        let shown = search_top_ids(collection, intent.query.clone(), top_k).await;
        if shown.first().copied() == Some(intent.good_id) {
            hits += 1;
        }

        if shown.is_empty() {
            continue;
        }

        let selected = if shown.contains(&intent.good_id) && rng.random::<f32>() < feedback_accuracy {
            intent.good_id
        } else {
            shown[0]
        };

        collection
            .rpi_apply_feedback(&shown, selected, &ShardSelectorInternal::All)
            .await
            .unwrap();
    }
    hits as f32 / intents.len() as f32
}

async fn search_top_ids(collection: &Collection, vector: Vec<f32>, limit: usize) -> Vec<PointIdType> {
    let result = collection
        .core_search_batch(
            CoreSearchRequestBatch {
                searches: vec![SearchRequestInternal {
                    vector: vector.into(),
                    with_payload: None,
                    with_vector: None,
                    filter: None,
                    params: None,
                    limit,
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

    result[0].iter().map(|p| p.id).collect()
}

async fn sample_shell1_purity(collection: &Collection, intents: &[Intent]) -> (f32, f32) {
    let mut good_in_shell1 = 0usize;
    let mut bad_in_shell1 = 0usize;
    let mut bad_checked = 0usize;

    for intent_idx in 0..intents.len() {
        let good_id = intents[intent_idx].good_id;
        if point_in_shell1(collection, good_id).await {
            good_in_shell1 += 1;
        }

        let first_bad = PointIdType::NumId((intent_idx * 6 + 1) as u64);
        if point_exists(collection, first_bad).await {
            bad_checked += 1;
            if point_in_shell1(collection, first_bad).await {
                bad_in_shell1 += 1;
            }
        }
    }

    (
        good_in_shell1 as f32 / intents.len() as f32,
        if bad_checked == 0 {
            0.0
        } else {
            bad_in_shell1 as f32 / bad_checked as f32
        },
    )
}

async fn point_exists(collection: &Collection, point_id: PointIdType) -> bool {
    collection
        .retrieve(
            PointRequestInternal {
                ids: vec![point_id],
                with_payload: Some(WithPayloadInterface::Bool(false)),
                with_vector: WithVector::Bool(false),
            },
            None,
            &ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .map(|records| !records.is_empty())
        .unwrap_or(false)
}

async fn point_in_shell1(collection: &Collection, point_id: PointIdType) -> bool {
    let record = collection
        .retrieve(
            PointRequestInternal {
                ids: vec![point_id],
                with_payload: Some(WithPayloadInterface::Bool(false)),
                with_vector: WithVector::Bool(true),
            },
            None,
            &ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .ok()
        .and_then(|mut records| records.pop());

    let Some(record) = record else {
        return false;
    };
    let Some(VectorStructInternal::Named(named)) = record.vector else {
        return false;
    };

    named.contains_key(rpi::shell_vector_name(1).as_str())
}

async fn wait_for_update_queue_to_drain(collection: &Collection) {
    for _ in 0..500 {
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

fn pct_gain(new: f32, old: f32) -> f32 {
    if old == 0.0 {
        return 0.0;
    }
    ((new - old) / old) * 100.0
}
