use std::collections::HashMap;
use std::mem;
use std::sync::Arc;
use std::time::Duration;

use ahash::{AHashMap, AHashSet};
use common::counter::hardware_accumulator::HwMeasurementAcc;
use futures::{TryFutureExt, future};
use itertools::{Either, Itertools};
use segment::data_types::vectors::{NamedQuery, VectorInternal};
use segment::types::{
    ExtendedPointId, Filter, Order, Payload, PointIdType, ScoredPoint, WithPayloadInterface,
    WithVector,
};
use serde_json::{Map as JsonMap, Value as JsonValue};
use shard::operations::point_ops::{VectorPersisted, VectorStructPersisted};
use shard::retrieve::record_internal::RecordInternal;
use shard::search::CoreSearchRequestBatch;
use tokio::time::Instant;

use super::Collection;
use crate::events::SlowQueryEvent;
use crate::operations::CollectionUpdateOperations;
use crate::operations::consistency_params::ReadConsistency;
use crate::operations::payload_ops::{PayloadOps, SetPayloadOp};
use crate::operations::point_ops::WriteOrdering;
use crate::operations::query_enum::QueryEnum;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::types::*;
use crate::operations::vector_ops::{PointVectorsPersisted, UpdateVectorsOp, VectorOperations};
use crate::rpi::{self, PointAccessData, RpiConfig};

impl Collection {
    #[cfg(feature = "testing")]
    pub async fn search(
        &self,
        request: CoreSearchRequest,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<ScoredPoint>> {
        if request.limit == 0 {
            return Ok(vec![]);
        }
        // search is a special case of search_batch with a single batch
        let request_batch = CoreSearchRequestBatch {
            searches: vec![request],
        };
        let results = self
            .do_core_search_batch(
                request_batch,
                read_consistency,
                shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await?;
        Ok(results.into_iter().next().unwrap())
    }

    pub async fn core_search_batch(
        &self,
        request: CoreSearchRequestBatch,
        read_consistency: Option<ReadConsistency>,
        shard_selection: ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let rpi_config = self.collection_config.read().await.rpi_config.clone();
        if let Some(rpi_config) = rpi_config {
            return self
                .rpi_core_search_batch(
                    request,
                    read_consistency,
                    shard_selection,
                    timeout,
                    hw_measurement_acc,
                    &rpi_config,
                )
                .await;
        }
        self.core_search_batch_no_rpi(
            request,
            read_consistency,
            shard_selection,
            timeout,
            hw_measurement_acc,
        )
        .await
    }

    async fn core_search_batch_no_rpi(
        &self,
        request: CoreSearchRequestBatch,
        read_consistency: Option<ReadConsistency>,
        shard_selection: ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let start = Instant::now();
        // shortcuts batch if all requests with limit=0
        if request.searches.iter().all(|s| s.limit == 0) {
            return Ok(vec![]);
        }

        let is_payload_required = request
            .searches
            .iter()
            .all(|s| s.with_payload.as_ref().is_some_and(|p| p.is_required()));
        let with_vectors = request
            .searches
            .iter()
            .all(|s| s.with_vector.as_ref().is_some_and(|wv| wv.is_enabled()));

        let metadata_required = is_payload_required || with_vectors;

        let sum_limits: usize = request.searches.iter().map(|s| s.limit).sum();
        let sum_offsets: usize = request.searches.iter().map(|s| s.offset).sum();

        // Number of records we need to retrieve to fill the search result.
        let require_transfers = self.shards_holder.read().await.len() * (sum_limits + sum_offsets);
        // Actually used number of records.
        let used_transfers = sum_limits;

        let is_required_transfer_large_enough = require_transfers
            > used_transfers.saturating_mul(super::query::PAYLOAD_TRANSFERS_FACTOR_THRESHOLD);

        if metadata_required && is_required_transfer_large_enough {
            // If there is a significant offset, we need to retrieve the whole result
            // set without payload first and then retrieve the payload.
            // It is required to do this because the payload might be too large to send over the
            // network.
            let mut without_payload_requests = Vec::with_capacity(request.searches.len());
            for search in &request.searches {
                let mut without_payload_request = search.clone();
                without_payload_request
                    .with_payload
                    .replace(WithPayloadInterface::Bool(false));
                without_payload_request
                    .with_vector
                    .replace(WithVector::Bool(false));
                without_payload_requests.push(without_payload_request);
            }
            let without_payload_batch = CoreSearchRequestBatch {
                searches: without_payload_requests,
            };
            let without_payload_results = self
                .do_core_search_batch(
                    without_payload_batch,
                    read_consistency,
                    &shard_selection,
                    timeout,
                    hw_measurement_acc.clone(),
                )
                .await?;
            // update timeout
            let timeout = timeout.map(|t| t.saturating_sub(start.elapsed()));
            let filled_results = without_payload_results
                .into_iter()
                .zip(request.searches.into_iter())
                .map(|(without_payload_result, req)| {
                    self.fill_search_result_with_payload(
                        without_payload_result,
                        req.with_payload.clone(),
                        req.with_vector.unwrap_or_default(),
                        read_consistency,
                        &shard_selection,
                        timeout,
                        hw_measurement_acc.clone(),
                    )
                });
            future::try_join_all(filled_results).await
        } else {
            let result = self
                .do_core_search_batch(
                    request,
                    read_consistency,
                    &shard_selection,
                    timeout,
                    hw_measurement_acc,
                )
                .await?;
            Ok(result)
        }
    }

    async fn rpi_core_search_batch(
        &self,
        request: CoreSearchRequestBatch,
        read_consistency: Option<ReadConsistency>,
        shard_selection: ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
        rpi_config: &RpiConfig,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let rpi_tracker = self.rpi_tracker.as_ref();
        let has_rpi_candidates = request
            .searches
            .iter()
            .any(|req| rpi_request_is_eligible(req, rpi_config));

        if !has_rpi_candidates {
            return self
                .core_search_batch_no_rpi(
                    request,
                    read_consistency,
                    shard_selection,
                    timeout,
                    hw_measurement_acc,
                )
                .await;
        }

        let mut results = Vec::with_capacity(request.searches.len());

        for search in request.searches {
            if !rpi_request_is_eligible(&search, rpi_config) {
                let batch = CoreSearchRequestBatch {
                    searches: vec![search],
                };
                let mut single_result = self
                    .core_search_batch_no_rpi(
                        batch,
                        read_consistency,
                        shard_selection.clone(),
                        timeout,
                        hw_measurement_acc.clone(),
                    )
                    .await?;
                results.push(single_result.pop().unwrap_or_default());
                continue;
            }

            let mut shell_result = Vec::new();
            let mut answering_shell = None;
            for shell in 1..=rpi_config.max_shells {
                let Some(shell_request) = rpi_request_for_shell(&search, rpi_config, shell) else {
                    break;
                };

                let batch = CoreSearchRequestBatch {
                    searches: vec![shell_request],
                };

                let mut single_result = self
                    .core_search_batch_no_rpi(
                        batch,
                        read_consistency,
                        shard_selection.clone(),
                        timeout,
                        hw_measurement_acc.clone(),
                    )
                    .await?;

                if let Some(found) = single_result.pop() {
                    if !found.is_empty() {
                        answering_shell = Some(shell);
                        shell_result = found;
                        break;
                    }
                }
            }

            if let Some(shell) = answering_shell {
                if let Some(tracker) = rpi_tracker {
                    tracker.record_hit(shell);
                }
                self.record_rpi_accesses(
                    &search,
                    &shell_result,
                    shell,
                    rpi_config,
                    read_consistency,
                    &shard_selection,
                    timeout,
                    hw_measurement_acc.clone(),
                )
                .await;
            } else if let Some(tracker) = rpi_tracker {
                tracker.record_miss();
            }

            results.push(shell_result);
        }

        Ok(results)
    }

    #[allow(clippy::too_many_arguments)]
    async fn record_rpi_accesses(
        &self,
        search_request: &CoreSearchRequest,
        results: &[ScoredPoint],
        answering_shell: u8,
        rpi_config: &RpiConfig,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) {
        let Some(rpi_access) = &self.rpi_access else {
            return;
        };

        // Keep shell-1 reads cheap: stable high-quality hits should stay on the hot path
        // without synchronous metadata write amplification.
        if answering_shell == 1 {
            return;
        }

        let should_rebalance = answering_shell >= 4
            && self
                .rpi_tracker
                .as_ref()
                .is_some_and(|tracker| tracker.needs_rebalance(rpi_config.rebalance_threshold));

        let passed_over_point_ids = if should_rebalance {
            self.collect_passed_over_points(
                search_request,
                answering_shell,
                rpi_config,
                read_consistency,
                shard_selection,
                timeout,
                hw_measurement_acc.clone(),
            )
            .await
            .unwrap_or_default()
        } else {
            Vec::new()
        };

        let mut payload_only_updates: Vec<(PointIdType, PointAccessData)> =
            Vec::with_capacity(results.len());
        let mut promotion_candidates: Vec<(PointIdType, PointAccessData)> = Vec::new();
        let mut demotion_candidates: Vec<(PointIdType, PointAccessData)> = Vec::new();

        {
            let mut access = rpi_access.write();

            for point in results {
                let entry = access.entry(point.id).or_insert_with(PointAccessData::new);
                if entry.current_shell != answering_shell {
                    entry.current_shell = answering_shell;
                }

                entry.record_access();
                let snapshot = entry.clone();

                match rpi::evaluate_promotion(rpi_config, &snapshot) {
                    rpi::PromotionResult::Promoted { .. } => {
                        promotion_candidates.push((point.id, snapshot));
                    }
                    _ if should_rebalance && answering_shell >= 4 => {
                        demotion_candidates.push((point.id, snapshot));
                    }
                    _ => {
                        if should_persist_access_payload(answering_shell, &snapshot) {
                            payload_only_updates.push((point.id, snapshot));
                        }
                    }
                }
            }
        }

        if should_rebalance {
            if let Some(tracker) = &self.rpi_tracker {
                tracker.record_rebalance();
            }
        }

        if !passed_over_point_ids.is_empty() {
            let mut access = rpi_access.write();
            for point_id in passed_over_point_ids {
                let entry = access.entry(point_id).or_insert_with(PointAccessData::new);
                demotion_candidates.push((point_id, entry.clone()));
            }
        }

        let mut moved_points = AHashSet::new();

        for (point_id, access_data) in promotion_candidates {
            match self
                .try_promote_point(
                    point_id,
                    access_data.clone(),
                    rpi_config,
                    read_consistency,
                    shard_selection,
                    timeout,
                    hw_measurement_acc.clone(),
                )
                .await
            {
                Some(updated_access_data) => {
                    moved_points.insert(point_id);
                    self.update_rpi_access_entry(point_id, updated_access_data);
                }
                None => {
                    if should_persist_access_payload(answering_shell, &access_data) {
                        payload_only_updates.push((point_id, access_data));
                    }
                }
            }
        }

        for (point_id, access_data) in demotion_candidates {
            if moved_points.contains(&point_id) {
                continue;
            }

            match self
                .try_demote_point(
                    point_id,
                    access_data.clone(),
                    rpi_config,
                    read_consistency,
                    shard_selection,
                    timeout,
                    hw_measurement_acc.clone(),
                )
                .await
            {
                Some(updated_access_data) => {
                    moved_points.insert(point_id);
                    self.update_rpi_access_entry(point_id, updated_access_data);
                }
                None => {
                    if should_persist_access_payload(answering_shell, &access_data) {
                        payload_only_updates.push((point_id, access_data));
                    }
                }
            }
        }

        for (point_id, access_data) in payload_only_updates {
            if let Err(err) = self
                .persist_rpi_access_payload(point_id, &access_data)
                .await
            {
                log::debug!("Failed to persist RPI payload for point {point_id}: {err}");
            }
        }

    }

    #[allow(clippy::too_many_arguments)]
    async fn collect_passed_over_points(
        &self,
        search_request: &CoreSearchRequest,
        answering_shell: u8,
        rpi_config: &RpiConfig,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<PointIdType>> {
        if answering_shell <= 1 {
            return Ok(Vec::new());
        }

        let passover_limit = search_request.limit.max(8).min(64);
        let mut passed_over = AHashSet::new();

        for shell in 1..answering_shell {
            let Some(mut shell_request) = rpi_request_for_shell(search_request, rpi_config, shell)
            else {
                continue;
            };

            shell_request.limit = passover_limit;
            shell_request.offset = 0;
            shell_request.score_threshold = None;

            let mut single_result = self
                .core_search_batch_no_rpi(
                    CoreSearchRequestBatch {
                        searches: vec![shell_request],
                    },
                    read_consistency,
                    shard_selection.clone(),
                    timeout,
                    hw_measurement_acc.clone(),
                )
                .await?;

            if let Some(found) = single_result.pop() {
                for point in found {
                    passed_over.insert(point.id);
                }
            }
        }

        Ok(passed_over.into_iter().collect())
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_promote_point(
        &self,
        point_id: PointIdType,
        access_data: PointAccessData,
        rpi_config: &RpiConfig,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> Option<PointAccessData> {
        let rpi::PromotionResult::Promoted {
            from_shell,
            to_shell,
        } = rpi::evaluate_promotion(rpi_config, &access_data)
        else {
            return None;
        };

        let shell_vector = match self
            .fetch_shell_vector(
                point_id,
                from_shell,
                read_consistency,
                shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await
        {
            Ok(Some(vector)) => vector,
            Ok(None) => return None,
            Err(err) => {
                log::debug!("Failed to fetch vector for RPI promotion point {point_id}: {err}");
                return None;
            }
        };

        let original_vector = rpi::extract_original_vector(&shell_vector, from_shell);
        let shell_move = rpi::ShellMoveOperation::promote(
            point_id,
            from_shell,
            to_shell,
            &original_vector,
            access_data,
        );
        let updated_access_data = shell_move.updated_access_data.clone();

        if let Err(err) = self.apply_shell_move_operation(shell_move).await {
            log::debug!("Failed to apply RPI promotion for point {point_id}: {err}");
            return None;
        }

        if let Some(tracker) = &self.rpi_tracker {
            tracker.record_promotion();
        }

        Some(updated_access_data)
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_demote_point(
        &self,
        point_id: PointIdType,
        access_data: PointAccessData,
        rpi_config: &RpiConfig,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> Option<PointAccessData> {
        match rpi::evaluate_demotion(rpi_config, &access_data, rpi_config.demotion_threshold) {
            rpi::DemotionResult::Demoted {
                from_shell,
                to_shell,
            } => {
                let shell_vector = match self
                    .fetch_shell_vector(
                        point_id,
                        from_shell,
                        read_consistency,
                        shard_selection,
                        timeout,
                        hw_measurement_acc,
                    )
                    .await
                {
                    Ok(Some(vector)) => vector,
                    Ok(None) => return None,
                    Err(err) => {
                        log::debug!(
                            "Failed to fetch vector for RPI demotion point {point_id}: {err}"
                        );
                        return None;
                    }
                };

                let original_vector = rpi::extract_original_vector(&shell_vector, from_shell);
                let shell_move = rpi::ShellMoveOperation::demote(
                    point_id,
                    from_shell,
                    to_shell,
                    &original_vector,
                    access_data,
                );
                let updated_access_data = shell_move.updated_access_data.clone();

                if let Err(err) = self.apply_shell_move_operation(shell_move).await {
                    log::debug!("Failed to apply RPI demotion for point {point_id}: {err}");
                    return None;
                }

                if let Some(tracker) = &self.rpi_tracker {
                    tracker.record_demotion();
                }

                Some(updated_access_data)
            }
            rpi::DemotionResult::Evicted { from_shell } => {
                let mut evicted_access_data = access_data;
                evicted_access_data.current_shell = rpi_config.max_shells.saturating_add(1);
                let shell_move =
                    rpi::ShellMoveOperation::evict(point_id, from_shell, evicted_access_data);
                let updated_access_data = shell_move.updated_access_data.clone();

                if let Err(err) = self.apply_shell_move_operation(shell_move).await {
                    log::debug!("Failed to evict RPI point {point_id}: {err}");
                    return None;
                }

                if let Some(tracker) = &self.rpi_tracker {
                    tracker.record_eviction();
                }

                Some(updated_access_data)
            }
            rpi::DemotionResult::NotDemoted { .. } => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn fetch_shell_vector(
        &self,
        point_id: PointIdType,
        shell: u8,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Option<Vec<f32>>> {
        let shell_name = rpi::shell_vector_name(shell);
        let request = PointRequestInternal {
            ids: vec![point_id],
            with_payload: Some(WithPayloadInterface::Bool(false)),
            with_vector: WithVector::Selector(vec![shell_name.clone().into()]),
        };

        let mut records = self
            .retrieve(
                request,
                read_consistency,
                shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await?;

        let Some(record) = records.pop() else {
            return Ok(None);
        };

        let Some(vector_struct) = record.vector else {
            return Ok(None);
        };

        let segment::data_types::vectors::VectorStructInternal::Named(vectors) = vector_struct
        else {
            return Ok(None);
        };

        match vectors.get(shell_name.as_str()) {
            Some(VectorInternal::Dense(dense)) => Ok(Some(dense.clone())),
            _ => Ok(None),
        }
    }

    async fn apply_shell_move_operation(
        &self,
        operation: rpi::ShellMoveOperation,
    ) -> CollectionResult<()> {
        let point_id = operation.point_id;

        if let Some((insert_name, insert_vector)) = operation.insert_to {
            let mut vectors = HashMap::new();
            vectors.insert(insert_name.into(), VectorPersisted::Dense(insert_vector));

            let update_operation = CollectionUpdateOperations::VectorOperation(
                VectorOperations::UpdateVectors(UpdateVectorsOp {
                    points: vec![PointVectorsPersisted {
                        id: point_id,
                        vector: VectorStructPersisted::Named(vectors),
                    }],
                    update_filter: None,
                }),
            );

            self.update_from_client_simple(
                update_operation,
                false,
                None,
                WriteOrdering::Weak,
                HwMeasurementAcc::disposable(),
            )
            .await?;
        }

        if let Some(delete_name) = operation.delete_from {
            let delete_operation = CollectionUpdateOperations::VectorOperation(
                VectorOperations::DeleteVectors(vec![point_id].into(), vec![delete_name.into()]),
            );

            self.update_from_client_simple(
                delete_operation,
                false,
                None,
                WriteOrdering::Weak,
                HwMeasurementAcc::disposable(),
            )
            .await?;
        }

        self.persist_rpi_access_payload(point_id, &operation.updated_access_data)
            .await
    }

    async fn persist_rpi_access_payload(
        &self,
        point_id: PointIdType,
        access_data: &PointAccessData,
    ) -> CollectionResult<()> {
        let payload = rpi_payload_from_access_data(access_data);
        let payload_operation =
            CollectionUpdateOperations::PayloadOperation(PayloadOps::SetPayload(SetPayloadOp {
                payload,
                points: Some(vec![point_id]),
                filter: None,
                key: None,
            }));

        self.update_from_client_simple(
            payload_operation,
            false,
            None,
            WriteOrdering::Weak,
            HwMeasurementAcc::disposable(),
        )
        .await
        .map(|_| ())
    }

    fn update_rpi_access_entry(&self, point_id: PointIdType, access_data: PointAccessData) {
        if let Some(access) = &self.rpi_access {
            access.write().insert(point_id, access_data);
        }
    }

    #[cfg(feature = "testing")]
    pub async fn rpi_apply_feedback(
        &self,
        shown_points: &[PointIdType],
        selected_point: PointIdType,
        shard_selection: &ShardSelectorInternal,
    ) -> CollectionResult<()> {
        let Some(rpi_config) = self.collection_config.read().await.rpi_config.clone() else {
            return Ok(());
        };

        for point_id in shown_points {
            let Some(shell) = self
                .detect_rpi_shell_for_point(
                    *point_id,
                    None,
                    shard_selection,
                    None,
                    HwMeasurementAcc::disposable(),
                    rpi_config.max_shells,
                )
                .await?
            else {
                continue;
            };

            let access_data = self.snapshot_rpi_access_data(*point_id, shell);

            if *point_id == selected_point {
                if let Some(updated) = self
                    .try_promote_point(
                        *point_id,
                        access_data,
                        &rpi_config,
                        None,
                        shard_selection,
                        None,
                        HwMeasurementAcc::disposable(),
                    )
                    .await
                {
                    self.update_rpi_access_entry(*point_id, updated);
                }
            } else if let Some(updated) = self
                .try_demote_point(
                    *point_id,
                    access_data,
                    &rpi_config,
                    None,
                    shard_selection,
                    None,
                    HwMeasurementAcc::disposable(),
                )
                .await
            {
                self.update_rpi_access_entry(*point_id, updated);
            }
        }

        Ok(())
    }

    #[cfg(feature = "testing")]
    fn snapshot_rpi_access_data(&self, point_id: PointIdType, current_shell: u8) -> PointAccessData {
        if let Some(access) = &self.rpi_access {
            if let Some(existing) = access.read().get(&point_id).cloned() {
                let mut data = existing;
                data.current_shell = current_shell;
                return data;
            }
        }

        PointAccessData {
            current_shell,
            original_shell: 1,
            ..PointAccessData::new()
        }
    }

    #[cfg(feature = "testing")]
    #[allow(clippy::too_many_arguments)]
    async fn detect_rpi_shell_for_point(
        &self,
        point_id: PointIdType,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
        max_shells: u8,
    ) -> CollectionResult<Option<u8>> {
        let request = PointRequestInternal {
            ids: vec![point_id],
            with_payload: Some(WithPayloadInterface::Bool(false)),
            with_vector: WithVector::Bool(true),
        };

        let mut records = self
            .retrieve(
                request,
                read_consistency,
                shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await?;

        let Some(record) = records.pop() else {
            return Ok(None);
        };

        let Some(vector_struct) = record.vector else {
            return Ok(None);
        };

        let segment::data_types::vectors::VectorStructInternal::Named(named) = vector_struct else {
            return Ok(None);
        };

        for shell in 1..=max_shells {
            if named.contains_key(rpi::shell_vector_name(shell).as_str()) {
                return Ok(Some(shell));
            }
        }

        Ok(None)
    }

    async fn do_core_search_batch(
        &self,
        request: CoreSearchRequestBatch,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let request = Arc::new(request);

        let instant = Instant::now();

        // query all shards concurrently
        let all_searches_res = {
            let shard_holder = self.shards_holder.read().await;
            let target_shards = shard_holder.select_shards(shard_selection)?;
            let all_searches = target_shards.into_iter().map(|(shard, shard_key)| {
                let shard_key = shard_key.cloned();
                shard
                    .core_search(
                        request.clone(),
                        read_consistency,
                        shard_selection.is_shard_id(),
                        timeout,
                        hw_measurement_acc.clone(),
                    )
                    .and_then(move |mut records| async move {
                        if shard_key.is_none() {
                            return Ok(records);
                        }
                        for batch in &mut records {
                            for point in batch {
                                point.shard_key.clone_from(&shard_key);
                            }
                        }
                        Ok(records)
                    })
            });
            future::try_join_all(all_searches).await?
        };

        let result = self
            .merge_from_shards(
                all_searches_res,
                request.clone(),
                !shard_selection.is_shard_id(),
            )
            .await;

        let filters_refs = request.searches.iter().map(|req| req.filter.as_ref());

        self.post_process_if_slow_request(instant.elapsed(), filters_refs);

        result
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn fill_search_result_with_payload(
        &self,
        search_result: Vec<ScoredPoint>,
        with_payload: Option<WithPayloadInterface>,
        with_vector: WithVector,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<ScoredPoint>> {
        // short-circuit if not needed
        if let (&Some(WithPayloadInterface::Bool(false)), &WithVector::Bool(false)) =
            (&with_payload, &with_vector)
        {
            return Ok(search_result
                .into_iter()
                .map(|point| ScoredPoint {
                    payload: None,
                    vector: None,
                    ..point
                })
                .collect());
        };

        let retrieve_request = PointRequestInternal {
            ids: search_result.iter().map(|x| x.id).collect(),
            with_payload,
            with_vector,
        };
        let retrieved_records = self
            .retrieve(
                retrieve_request,
                read_consistency,
                shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await?;

        let mut records_map: AHashMap<ExtendedPointId, RecordInternal> = retrieved_records
            .into_iter()
            .map(|rec| (rec.id, rec))
            .collect();
        let enriched_result = search_result
            .into_iter()
            .filter_map(|mut scored_point| {
                // Points might get deleted between search and retrieve.
                // But it's not a problem, because we don't want to return deleted points.
                // So we just filter out them.
                records_map.remove(&scored_point.id).map(|record| {
                    scored_point.payload = record.payload;
                    scored_point.vector = record.vector;
                    scored_point
                })
            })
            .collect();
        Ok(enriched_result)
    }

    async fn merge_from_shards(
        &self,
        mut all_searches_res: Vec<Vec<Vec<ScoredPoint>>>,
        request: Arc<CoreSearchRequestBatch>,
        is_client_request: bool,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let batch_size = request.searches.len();

        let collection_params = self.collection_config.read().await.params.clone();

        // Merge results from shards in order and deduplicate based on point ID
        let mut top_results: Vec<Vec<ScoredPoint>> = Vec::with_capacity(batch_size);
        let mut seen_ids = AHashSet::new();

        for (batch_index, request) in request.searches.iter().enumerate() {
            let order = if request.query.is_distance_scored() {
                collection_params
                    .get_distance(request.query.get_vector_name())?
                    .distance_order()
            } else {
                // Score comes from special handling of the distances in a way that it doesn't
                // directly represent distance anymore, so the order is always `LargeBetter`
                Order::LargeBetter
            };

            let results_from_shards = all_searches_res
                .iter_mut()
                .map(|res| res.get_mut(batch_index).map_or(Vec::new(), mem::take));

            let merged_iter = match order {
                Order::LargeBetter => Either::Left(results_from_shards.kmerge_by(|a, b| a > b)),
                Order::SmallBetter => Either::Right(results_from_shards.kmerge_by(|a, b| a < b)),
            }
            .filter(|point| seen_ids.insert(point.id));

            // Skip `offset` only for client requests
            // to avoid applying `offset` twice in distributed mode.
            let top_res = if is_client_request && request.offset > 0 {
                merged_iter
                    .skip(request.offset)
                    .take(request.limit)
                    .collect()
            } else {
                merged_iter.take(request.offset + request.limit).collect()
            };

            top_results.push(top_res);

            seen_ids.clear();
        }

        Ok(top_results)
    }

    pub fn post_process_if_slow_request<'a>(
        &self,
        duration: Duration,
        filters: impl IntoIterator<Item = Option<&'a Filter>>,
    ) {
        if duration > crate::problems::UnindexedField::slow_query_threshold() {
            let filters = filters.into_iter().flatten().cloned().collect_vec();

            let schema = self.payload_index_schema.read().schema.clone();

            issues::publish(SlowQueryEvent {
                collection_id: self.id.clone(),
                filters,
                schema,
            });
        }
    }
}

fn rpi_payload_from_access_data(access_data: &PointAccessData) -> Payload {
    let mut payload_map = JsonMap::new();
    payload_map.insert(
        rpi::payload_fields::HIT_COUNT.to_string(),
        JsonValue::from(access_data.hit_count),
    );
    payload_map.insert(
        rpi::payload_fields::LAST_ACCESS.to_string(),
        JsonValue::from(access_data.last_access),
    );
    payload_map.insert(
        rpi::payload_fields::CURRENT_SHELL.to_string(),
        JsonValue::from(access_data.current_shell),
    );
    payload_map.insert(
        rpi::payload_fields::DEMOTION_COUNT.to_string(),
        JsonValue::from(access_data.demotion_count),
    );
    payload_map.insert(
        rpi::payload_fields::ORIGINAL_SHELL.to_string(),
        JsonValue::from(access_data.original_shell),
    );
    Payload::from(payload_map)
}

fn rpi_request_is_eligible(request: &CoreSearchRequest, rpi_config: &RpiConfig) -> bool {
    if rpi_config.max_shells < 1 {
        return false;
    }
    if !request.query.is_distance_scored() {
        return false;
    }
    matches!(
        request.query,
        QueryEnum::Nearest(NamedQuery {
            query: VectorInternal::Dense(_),
            ..
        })
    )
}

fn rpi_request_for_shell(
    request: &CoreSearchRequest,
    rpi_config: &RpiConfig,
    shell: u8,
) -> Option<CoreSearchRequest> {
    let QueryEnum::Nearest(named_query) = &request.query else {
        return None;
    };

    let VectorInternal::Dense(vector) = &named_query.query else {
        return None;
    };

    let params = rpi::ShellSearchParams::for_shell(
        shell,
        vector.as_slice(),
        rpi_config,
        request.limit,
        None,
    );

    Some(CoreSearchRequest {
        query: QueryEnum::Nearest(NamedQuery {
            query: VectorInternal::Dense(params.scaled_query),
            using: Some(params.vector_name.into()),
        }),
        filter: request.filter.clone(),
        params: request.params.clone(),
        limit: request.limit,
        offset: request.offset,
        with_payload: request.with_payload.clone(),
        with_vector: request.with_vector.clone(),
        score_threshold: Some(params.score_threshold),
    })
}

fn should_persist_access_payload(answering_shell: u8, access_data: &PointAccessData) -> bool {
    // Shell-1 is intentionally in-memory only for read-path speed.
    if answering_shell <= 1 {
        return false;
    }

    // For shell 2+ keep payload writes sparse: persist at first hit,
    // near promotion boundaries, and periodically for observability.
    let hits = access_data.hit_count;
    hits <= 2 || hits % 16 == 0
}
