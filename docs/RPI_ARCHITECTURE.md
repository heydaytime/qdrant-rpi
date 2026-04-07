# RPI Architecture Guide

This document explains how Radial Priority Indexing (RPI) works in this fork of Qdrant.

## Scope

RPI is a collection-level feature that adds shell-based quality adaptation on top of Qdrant vector search.

It is designed for repeated semantic workloads where quality should improve over time based on interaction patterns.

## Core concepts

- **Collection**: independent vector space and config boundary.
- **RPI-enabled collection**: collection with `rpi_config` set at creation.
- **Shells**: named vectors `rpi_shell_1 ... rpi_shell_N`.
- **Trust model**: lower shell index means higher trust.

Shell semantics:
- `k=1`: highest trust, hot path.
- `k>1`: lower trust fallback tiers.

## Math and geometry

RPI uses Euclidean distance for shell geometry.

- Vector scaling per shell: `v_k = k * v`.
- Query scaling per shell: `q_k = k * q`.
- Threshold scaling per shell: `epsilon_k = base_epsilon * k`.

This keeps angular behavior consistent across shells while encoding quality in radial magnitude.

## Data layout

For RPI collections, create-collection generates shell vector definitions derived from source vector config.

- Shell vectors are created as named vectors.
- Shell 1 may keep HNSW (controlled by `hnsw_for_shell_one`).
- Higher shells are configured without shell-specific HNSW override (plain path behavior in practice).

Per-point adaptation metadata is tracked via internal payload fields (`_rpi_*`) and in-memory tracking structures.

## Write path

When RPI is enabled:

- Point upsert operations are transformed.
- New points get shell vectors, starting at shell 1.
- Original vectors are preserved; RPI does not normalize everything to one shell.

When RPI is not enabled:

- Normal Qdrant write path is used.

## Read path

RPI path applies only to eligible search requests.

Eligibility (high level):
- dense nearest-neighbor style requests where shell-specific `using` + threshold rewriting is valid.

Execution order:
1. Search shell 1.
2. If empty, search shell 2.
3. Continue up to `max_shells`.
4. Return first non-empty shell result.

Important behavior:
- It is not cross-shell rank fusion.
- It is first-hit fallthrough.

If request is not RPI-eligible, normal search path is used.

## Adaptation model

RPI adaptation uses promotion/demotion/eviction with thresholds.

Promotion:
- Requires shell > 1.
- Requires `hit_count >= promotion_threshold`.
- Promotes one step toward shell 1.

Demotion:
- Triggered by rebalance/high-depth logic and pass-over evaluation.
- Demotes one shell at a time.

Eviction:
- When demotion would move beyond `max_shells`.

## Runtime stats

RPI tracks and reports:

- total searches,
- per-shell hit distribution,
- average search depth,
- promotion/demotion/eviction totals,
- rebalance count.

These appear in collection info as `rpi_stats`.

## Backward compatibility model

Compatibility is collection-level:

- Collections without `rpi_config`: standard Qdrant behavior.
- Collections with `rpi_config`: RPI behavior for eligible searches.

There is no per-request mode flag to force RPI on/off for a given collection.

## Operational expectations

Typical observed behavior:

- Warm shell-1 path approaches baseline latency.
- Early convergence / high-shell paths can be slower.
- Quality gains are strongest in repeated-query, feedback-rich workloads.

## Files of interest

- `lib/collection/src/rpi/mod.rs`
- `lib/collection/src/rpi/config.rs`
- `lib/collection/src/rpi/scaling.rs`
- `lib/collection/src/rpi/search.rs`
- `lib/collection/src/rpi/tracking.rs`
- `lib/collection/src/rpi/demotion.rs`
- `lib/collection/src/rpi/operations.rs`
- `lib/collection/src/collection/search.rs`
- `lib/collection/src/collection/point_ops.rs`
- `lib/storage/src/content_manager/toc/create_collection.rs`
