use std::sync::Arc;
use std::time::{Duration, Instant};

use api::grpc::Usage;
use api::grpc::qdrant::points_server::Points;
use api::grpc::qdrant::{
    ClearPayloadPoints, CountPoints, CountResponse, CreateFieldIndexCollection,
    DeleteFieldIndexCollection, DeletePayloadPoints, DeletePointVectors, DeletePoints,
    DiscoverBatchPoints, DiscoverBatchResponse, DiscoverPoints, DiscoverResponse, FacetCounts,
    FacetResponse, GetPoints, GetResponse, PointsOperationResponse, QueryBatchPoints,
    QueryBatchResponse, QueryGroupsResponse, QueryPointGroups, QueryPoints, QueryResponse,
    RecommendBatchPoints, RecommendBatchResponse, RecommendGroupsResponse, RecommendPointGroups,
    RecommendPoints, RecommendResponse, RpiFeedbackPoints, ScrollPoints, ScrollResponse,
    SearchBatchPoints, SearchBatchResponse, SearchGroupsResponse, SearchMatrixOffsets,
    SearchMatrixOffsetsResponse, SearchMatrixPairs, SearchMatrixPairsResponse, SearchMatrixPoints,
    SearchPointGroups, SearchPoints, SearchResponse, SetPayloadPoints, ShellSearchPoints,
    ShellSearchResponse, UpdateBatchPoints, UpdateBatchResponse, UpdatePointVectors, UpsertPoints,
};
use collection::operations::consistency_params::ReadConsistency;
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::CoreSearchRequest;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use storage::content_manager::toc::request_hw_counter::RequestHwCounter;
use storage::dispatcher::Dispatcher;
use tonic::{Request, Response, Status};

use super::query_common::*;
use super::update_common::*;
use super::validate;
use crate::common::inference::api_keys::extract_inference_auth;
use crate::common::inference::params::InferenceParams;
use crate::common::strict_mode::*;
use crate::common::update::{InternalUpdateParams, do_rpi_feedback};
use crate::settings::ServiceConfig;
use crate::tonic::auth::extract_auth;

pub struct PointsService {
    dispatcher: Arc<Dispatcher>,
    service_config: ServiceConfig,
}

impl PointsService {
    pub fn new(dispatcher: Arc<Dispatcher>, service_config: ServiceConfig) -> Self {
        Self {
            dispatcher,
            service_config,
        }
    }

    fn get_request_collection_hw_usage_counter(
        &self,
        collection_name: String,
        wait: Option<bool>,
    ) -> RequestHwCounter {
        let counter = HwMeasurementAcc::new_with_metrics_drain(
            self.dispatcher.get_collection_hw_metrics(collection_name),
        );

        let waiting = wait != Some(false);
        RequestHwCounter::new(counter, self.service_config.hardware_reporting() && waiting)
    }
}

#[tonic::async_trait]
impl Points for PointsService {
    async fn upsert(
        &self,
        mut request: Request<UpsertPoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);
        let timeout = request.get_ref().timeout.map(Duration::from_secs);
        let api_keys = extract_inference_auth(&request);
        let inference_params = InferenceParams::new(api_keys, timeout);

        let collection_name = request.get_ref().collection_name.clone();
        let wait = Some(request.get_ref().wait.unwrap_or(false));
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, wait);

        upsert(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            InternalUpdateParams::default(),
            auth,
            inference_params,
            hw_metrics,
        )
        .await
        .map(|resp| resp.map(PointsOperationResponse::from))
    }

    async fn delete(
        &self,
        mut request: Request<DeletePoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);
        let collection_name = request.get_ref().collection_name.clone();
        let wait = Some(request.get_ref().wait.unwrap_or(false));
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, wait);

        delete(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            InternalUpdateParams::default(),
            auth,
            hw_metrics,
        )
        .await
        .map(|resp| resp.map(PointsOperationResponse::from))
    }

    async fn rpi_feedback(
        &self,
        mut request: Request<RpiFeedbackPoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        let auth = extract_auth(&mut request);
        let inner_request = request.into_inner();
        let collection_name = inner_request.collection_name.clone();
        let wait = Some(inner_request.wait.unwrap_or(false));
        let _hw_metrics =
            self.get_request_collection_hw_usage_counter(collection_name.clone(), wait);

        let shown_points: Vec<segment::types::PointIdType> = inner_request
            .shown_points
            .into_iter()
            .map(|p| p.try_into())
            .collect::<Result<_, _>>()
            .map_err(|e| Status::invalid_argument(format!("Invalid point ID: {e}")))?;

        let selected_point: segment::types::PointIdType = inner_request
            .selected_point
            .ok_or_else(|| Status::invalid_argument("selected_point is required"))?
            .try_into()
            .map_err(|e| Status::invalid_argument(format!("Invalid selected point ID: {e}")))?;

        // Convert gRPC ShardKeySelector directly to ShardSelectorInternal
        let shard_selector: ShardSelectorInternal = match inner_request.shard_key_selector {
            Some(sk) => sk.try_into().map_err(|e| {
                Status::invalid_argument(format!("Invalid shard key selector: {e}"))
            })?,
            None => ShardSelectorInternal::Empty,
        };

        let result = do_rpi_feedback(
            &self.dispatcher,
            collection_name.clone(),
            shown_points,
            selected_point,
            shard_selector,
            auth,
        )
        .await;

        match result {
            Ok(update_result) => {
                let internal_response = points_operation_response_internal(
                    std::time::Instant::now(),
                    update_result,
                    None, // No hardware usage tracking for now
                );
                let response = PointsOperationResponse::from(internal_response);
                Ok(Response::new(response))
            }
            Err(e) => Err(Status::internal(format!("RPI feedback failed: {e}"))),
        }
    }

    async fn shell_search(
        &self,
        mut request: Request<ShellSearchPoints>,
    ) -> Result<Response<ShellSearchResponse>, Status> {
        let auth = extract_auth(&mut request);

        let inner_request = request.into_inner();
        let collection_name = inner_request.collection_name.clone();
        // Shell search doesn't have a wait field, pass None
        let _hw_metrics =
            self.get_request_collection_hw_usage_counter(collection_name.clone(), None);

        // Convert gRPC request to internal ShellSearchRequest
        let search_request = api::rest::schema::ShellSearchRequest {
            vector: inner_request.vector,
            limit: inner_request.limit as usize,
            epsilon: inner_request.epsilon,
            max_shell: inner_request.max_shell.map(|m| m as u8),
            with_payload: inner_request.with_payload.is_some(),
            with_vector: inner_request.with_vectors.is_some(),
            shard_key: None, // Will use shard_key_selector below
        };

        let shard_selection = match inner_request.shard_key_selector {
            Some(sk) => sk.try_into().map_err(|e| {
                Status::invalid_argument(format!("Invalid shard key selector: {e}"))
            })?,
            None => ShardSelectorInternal::All,
        };

        let read_consistency = inner_request
            .read_consistency
            .map(ReadConsistency::try_from)
            .transpose()
            .map_err(|e| Status::invalid_argument(format!("Invalid read consistency: {e}")))?;

        let timeout = inner_request.timeout.map(std::time::Duration::from_secs);

        let verification_pass =
            collection::operations::verification::new_unchecked_verification_pass();
        let res = crate::common::query::do_shell_search_points(
            self.dispatcher.toc(&auth, &verification_pass),
            &collection_name,
            search_request,
            read_consistency,
            shard_selection,
            auth,
            timeout,
            _hw_metrics.get_counter(),
        )
        .await;

        match res {
            Ok(response) => {
                // Convert rest ScoredPoints to grpc ScoredPoints
                let grpc_points: Result<Vec<_>, _> =
                    response.result.into_iter().map(|p| p.try_into()).collect();

                let grpc_points =
                    grpc_points.map_err(|e| Status::internal(format!("Conversion error: {e}")))?;

                // Convert to gRPC response
                let grpc_response = ShellSearchResponse {
                    result: grpc_points,
                    metadata: Some(api::grpc::qdrant::ShellSearchMetadata {
                        hit_shell: response.metadata.hit_shell as u32,
                        searched_shells: response.metadata.searched_shells as u32,
                        definitive_miss: response.metadata.definitive_miss,
                        result_count: response.metadata.result_count as u64,
                    }),
                    time: response.time,
                    usage: None,
                };
                Ok(Response::new(grpc_response))
            }
            Err(e) => Err(Status::internal(format!("Shell search failed: {e}"))),
        }
    }

    async fn get(&self, mut request: Request<GetPoints>) -> Result<Response<GetResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);
        let inner_request = request.into_inner();
        let collection_name = inner_request.collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);

        get(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            inner_request,
            None,
            auth,
            hw_metrics,
        )
        .await
    }

    async fn update_vectors(
        &self,
        mut request: Request<UpdatePointVectors>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        validate(request.get_ref())?;

        // Nothing to verify here.

        let auth = extract_auth(&mut request);
        let timeout = request.get_ref().timeout.map(Duration::from_secs);
        let api_keys = extract_inference_auth(&request);
        let inference_params = InferenceParams::new(api_keys, timeout);

        let collection_name = request.get_ref().collection_name.clone();
        let wait = Some(request.get_ref().wait.unwrap_or(false));
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, wait);

        update_vectors(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            InternalUpdateParams::default(),
            auth,
            inference_params,
            hw_metrics,
        )
        .await
        .map(|resp| resp.map(PointsOperationResponse::from))
    }

    async fn delete_vectors(
        &self,
        mut request: Request<DeletePointVectors>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);

        let collection_name = request.get_ref().collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);

        delete_vectors(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            InternalUpdateParams::default(),
            auth,
            hw_metrics,
        )
        .await
        .map(|resp| resp.map(Into::into))
    }

    async fn set_payload(
        &self,
        mut request: Request<SetPayloadPoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);

        let collection_name = request.get_ref().collection_name.clone();
        let wait = Some(request.get_ref().wait.unwrap_or(false));
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, wait);

        set_payload(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            InternalUpdateParams::default(),
            auth,
            hw_metrics,
        )
        .await
        .map(|resp| resp.map(Into::into))
    }

    async fn overwrite_payload(
        &self,
        mut request: Request<SetPayloadPoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);

        let collection_name = request.get_ref().collection_name.clone();
        let wait = Some(request.get_ref().wait.unwrap_or(false));
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, wait);

        overwrite_payload(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            InternalUpdateParams::default(),
            auth,
            hw_metrics,
        )
        .await
        .map(|resp| resp.map(Into::into))
    }

    async fn delete_payload(
        &self,
        mut request: Request<DeletePayloadPoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);

        let collection_name = request.get_ref().collection_name.clone();
        let wait = Some(request.get_ref().wait.unwrap_or(false));
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, wait);

        delete_payload(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            InternalUpdateParams::default(),
            auth,
            hw_metrics,
        )
        .await
        .map(|resp| resp.map(Into::into))
    }

    async fn clear_payload(
        &self,
        mut request: Request<ClearPayloadPoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);

        let collection_name = request.get_ref().collection_name.clone();
        let wait = Some(request.get_ref().wait.unwrap_or(false));
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, wait);

        clear_payload(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            InternalUpdateParams::default(),
            auth,
            hw_metrics,
        )
        .await
        .map(|resp| resp.map(Into::into))
    }

    async fn update_batch(
        &self,
        mut request: Request<UpdateBatchPoints>,
    ) -> Result<Response<UpdateBatchResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);
        let timeout = request.get_ref().timeout.map(Duration::from_secs);
        let api_keys = extract_inference_auth(&request);
        let inference_params = InferenceParams::new(api_keys, timeout);

        let collection_name = request.get_ref().collection_name.clone();
        let wait = Some(request.get_ref().wait.unwrap_or(false));
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, wait);

        update_batch(
            &self.dispatcher,
            request.into_inner(),
            InternalUpdateParams::default(),
            auth,
            inference_params,
            hw_metrics,
        )
        .await
    }

    async fn create_field_index(
        &self,
        mut request: Request<CreateFieldIndexCollection>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);
        let collection_name = request.get_ref().collection_name.clone();
        let wait = Some(request.get_ref().wait.unwrap_or(false));
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, wait);

        create_field_index(
            self.dispatcher.clone(),
            request.into_inner(),
            InternalUpdateParams::default(),
            auth,
            hw_metrics,
        )
        .await
        .map(|resp| resp.map(Into::into))
    }

    async fn delete_field_index(
        &self,
        mut request: Request<DeleteFieldIndexCollection>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);

        delete_field_index(
            self.dispatcher.clone(),
            request.into_inner(),
            InternalUpdateParams::default(),
            auth,
        )
        .await
        .map(|resp| resp.map(Into::into))
    }

    async fn search(
        &self,
        mut request: Request<SearchPoints>,
    ) -> Result<Response<SearchResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);

        let collection_name = request.get_ref().collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);

        let res = search(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            None,
            auth,
            hw_metrics,
        )
        .await?;

        Ok(res)
    }

    async fn search_batch(
        &self,
        mut request: Request<SearchBatchPoints>,
    ) -> Result<Response<SearchBatchResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);

        let SearchBatchPoints {
            collection_name,
            search_points,
            read_consistency,
            timeout,
        } = request.into_inner();

        let timeout = timeout.map(Duration::from_secs);

        let mut requests = Vec::new();

        for mut search_point in search_points {
            let shard_key = search_point.shard_key_selector.take();

            let shard_selector = convert_shard_selector_for_read(None, shard_key)?;
            let core_search_request = CoreSearchRequest::try_from(search_point)?;

            requests.push((core_search_request, shard_selector));
        }

        let hw_metrics =
            self.get_request_collection_hw_usage_counter(collection_name.clone(), None);

        let res = core_search_batch(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            &collection_name,
            requests,
            read_consistency,
            auth,
            timeout,
            hw_metrics,
        )
        .await?;

        Ok(res)
    }

    async fn search_groups(
        &self,
        mut request: Request<SearchPointGroups>,
    ) -> Result<Response<SearchGroupsResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let collection_name = request.get_ref().collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);
        let res = search_groups(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            None,
            auth,
            hw_metrics,
        )
        .await?;

        Ok(res)
    }

    async fn scroll(
        &self,
        mut request: Request<ScrollPoints>,
    ) -> Result<Response<ScrollResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);

        let inner_request = request.into_inner();
        let collection_name = inner_request.collection_name.clone();

        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);

        scroll(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            inner_request,
            None,
            auth,
            hw_metrics,
        )
        .await
    }

    async fn recommend(
        &self,
        mut request: Request<RecommendPoints>,
    ) -> Result<Response<RecommendResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let collection_name = request.get_ref().collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);
        let res = recommend(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            auth,
            hw_metrics,
        )
        .await?;

        Ok(res)
    }

    async fn recommend_batch(
        &self,
        mut request: Request<RecommendBatchPoints>,
    ) -> Result<Response<RecommendBatchResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let RecommendBatchPoints {
            collection_name,
            recommend_points,
            read_consistency,
            timeout,
        } = request.into_inner();

        let hw_metrics =
            self.get_request_collection_hw_usage_counter(collection_name.clone(), None);

        let res = recommend_batch(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            &collection_name,
            recommend_points,
            read_consistency,
            auth,
            timeout.map(Duration::from_secs),
            hw_metrics,
        )
        .await?;

        Ok(res)
    }

    async fn recommend_groups(
        &self,
        mut request: Request<RecommendPointGroups>,
    ) -> Result<Response<RecommendGroupsResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let collection_name = request.get_ref().collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);

        let res = recommend_groups(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            auth,
            hw_metrics,
        )
        .await?;

        Ok(res)
    }

    async fn discover(
        &self,
        mut request: Request<DiscoverPoints>,
    ) -> Result<Response<DiscoverResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let collection_name = request.get_ref().collection_name.clone();

        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);
        let res = discover(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            auth,
            hw_metrics,
        )
        .await?;

        Ok(res)
    }

    async fn discover_batch(
        &self,
        mut request: Request<DiscoverBatchPoints>,
    ) -> Result<Response<DiscoverBatchResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let DiscoverBatchPoints {
            collection_name,
            discover_points,
            read_consistency,
            timeout,
        } = request.into_inner();

        let hw_metrics =
            self.get_request_collection_hw_usage_counter(collection_name.clone(), None);
        let res = discover_batch(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            &collection_name,
            discover_points,
            read_consistency,
            auth,
            timeout.map(Duration::from_secs),
            hw_metrics,
        )
        .await?;

        Ok(res)
    }

    async fn count(
        &self,
        mut request: Request<CountPoints>,
    ) -> Result<Response<CountResponse>, Status> {
        validate(request.get_ref())?;

        let auth = extract_auth(&mut request);
        let collection_name = request.get_ref().collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);
        let res = count(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            None,
            auth,
            hw_metrics,
        )
        .await?;

        Ok(res)
    }

    async fn query(
        &self,
        mut request: Request<QueryPoints>,
    ) -> Result<Response<QueryResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let timeout = request.get_ref().timeout.map(Duration::from_secs);
        let api_keys = extract_inference_auth(&request);
        let inference_params = InferenceParams::new(api_keys, timeout);
        let collection_name = request.get_ref().collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);

        let res = query(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            None,
            auth,
            hw_metrics,
            inference_params,
        )
        .await?;

        Ok(res)
    }

    async fn query_batch(
        &self,
        mut request: Request<QueryBatchPoints>,
    ) -> Result<Response<QueryBatchResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let timeout = request.get_ref().timeout.map(Duration::from_secs);
        let api_keys = extract_inference_auth(&request);
        let inference_params = InferenceParams::new(api_keys, timeout);

        let request = request.into_inner();
        let QueryBatchPoints {
            collection_name,
            query_points,
            read_consistency,
            timeout,
        } = request;
        let timeout = timeout.map(Duration::from_secs);
        let hw_metrics =
            self.get_request_collection_hw_usage_counter(collection_name.clone(), None);
        let res = query_batch(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            &collection_name,
            query_points,
            read_consistency,
            auth,
            timeout,
            hw_metrics,
            inference_params,
        )
        .await?;

        Ok(res)
    }

    async fn query_groups(
        &self,
        mut request: Request<QueryPointGroups>,
    ) -> Result<Response<QueryGroupsResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let timeout = request.get_ref().timeout.map(Duration::from_secs);
        let api_keys = extract_inference_auth(&request);
        let inference_params = InferenceParams::new(api_keys, timeout);
        let collection_name = request.get_ref().collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);

        let res = query_groups(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            None,
            auth,
            hw_metrics,
            inference_params,
        )
        .await?;

        Ok(res)
    }
    async fn facet(
        &self,
        mut request: Request<FacetCounts>,
    ) -> Result<Response<FacetResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let collection_name = request.get_ref().collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);
        facet(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            auth,
            hw_metrics,
        )
        .await
    }

    async fn search_matrix_pairs(
        &self,
        mut request: Request<SearchMatrixPoints>,
    ) -> Result<Response<SearchMatrixPairsResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let timing = Instant::now();
        let collection_name = request.get_ref().collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);
        let search_matrix_response = search_points_matrix(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            auth,
            hw_metrics.get_counter(),
        )
        .await?;

        let pairs_response = SearchMatrixPairsResponse {
            result: Some(SearchMatrixPairs::from(search_matrix_response)),
            time: timing.elapsed().as_secs_f64(),
            usage: Usage::from_hardware_usage(hw_metrics.to_grpc_api()).into_non_empty(),
        };

        Ok(Response::new(pairs_response))
    }

    async fn search_matrix_offsets(
        &self,
        mut request: Request<SearchMatrixPoints>,
    ) -> Result<Response<SearchMatrixOffsetsResponse>, Status> {
        validate(request.get_ref())?;
        let auth = extract_auth(&mut request);
        let timing = Instant::now();
        let collection_name = request.get_ref().collection_name.clone();
        let hw_metrics = self.get_request_collection_hw_usage_counter(collection_name, None);
        let search_matrix_response = search_points_matrix(
            StrictModeCheckedTocProvider::new(&self.dispatcher),
            request.into_inner(),
            auth,
            hw_metrics.get_counter(),
        )
        .await?;

        let offsets_response = SearchMatrixOffsetsResponse {
            result: Some(SearchMatrixOffsets::from(search_matrix_response)),
            time: timing.elapsed().as_secs_f64(),
            usage: Usage::from_hardware_usage(hw_metrics.to_grpc_api()).into_non_empty(),
        };

        Ok(Response::new(offsets_response))
    }
}
