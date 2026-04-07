# RPI gRPC Guide

This guide describes the gRPC surface for RPI-enabled collections in this fork.

## Summary

RPI gRPC support is available for:

- create-collection with `rpi_config`,
- collection info readback of `config.rpi_config`,
- collection info `rpi_stats`.

RPI execution itself is automatic per collection; clients do not send an "RPI mode" query flag.

## Proto locations

- Collection service: `lib/api/src/grpc/proto/collections_service.proto`
- Collection messages: `lib/api/src/grpc/proto/collections.proto`
- Point/search services: `lib/api/src/grpc/proto/points_service.proto`, `lib/api/src/grpc/proto/points.proto`

## gRPC create with RPI

`CreateCollection` now includes:

- `optional RpiConfig rpi_config = 19;`

`RpiConfig` fields:

- `max_shells` (2..20)
- `base_epsilon` (0.001..10.0)
- `source_vector` (optional)
- `demotion_threshold` (1..100)
- `hnsw_for_shell_one` (bool)
- `track_lru` (bool)
- `promotion_threshold` (1..1000)
- `rebalance_threshold` (1.5..10.0)

Validation is enforced in generated API validation wiring.

## Readback via GetCollectionInfo

`CollectionConfig` now includes:

- `optional RpiConfig rpi_config = 8;`

`CollectionInfo` also includes runtime:

- `optional RpiStats rpi_stats = 10;`

So clients can retrieve both:

- static RPI configuration (`config.rpi_config`),
- live adaptation counters (`rpi_stats`).

## Query behavior over gRPC

Search requests are still standard `SearchPoints` / `QueryPoints` style requests.

Behavior is decided server-side by collection config:

- collection has `rpi_config`: eligible dense nearest searches use RPI shell path,
- collection has no `rpi_config`: standard Qdrant path.

No explicit per-request switch is required.

## Compatibility notes

- Existing clients that do not set `rpi_config` continue to work unchanged.
- Existing non-RPI collections continue to behave as standard Qdrant collections.
- RPI config remains creation-time config; update flow does not expose mutable RPI config.

## Practical test flow (gRPC)

1. Call `Collections.Create` with `rpi_config` populated.
2. Call `Collections.Get` and verify:
   - `result.config.rpi_config` is present,
   - `result.rpi_stats` is present.
3. Upsert points using `Points.Upsert`.
4. Search using `Points.Search` and validate expected retrieval behavior.

## Related implementation files

- `lib/storage/src/content_manager/conversions.rs` (gRPC CreateCollection -> internal)
- `lib/collection/src/operations/types.rs` (public collection config includes `rpi_config`)
- `lib/collection/src/operations/conversions.rs` (gRPC conversions for `RpiConfig` and `CollectionInfo`)
- `lib/api/src/grpc/proto/collections.proto` (proto definition)
- `lib/api/build.rs` (validation rules)
