# Qdrant RPI Fork

This repository is a Qdrant fork that adds **Radial Priority Indexing (RPI)**.

RPI turns the vector store into a quality-aware semantic cache:
- shell `k=1` is the trusted hot tier,
- shells `k=2..N` are progressively lower trust,
- points move between shells based on observed behavior.

If you only need to run it, jump to [Build and Run](#build-and-run).

## What is different from upstream Qdrant

Core behavior in this fork:
- RPI is enabled per collection through `rpi_config` at collection creation.
- Vectors are stored in shell-specific named vectors: `rpi_shell_1`, `rpi_shell_2`, ...
- Search uses shell fallthrough: search shell 1 first, then 2, then 3, until first non-empty result.
- Query and vectors are scaled per shell (`v_k = k * v`, `q_k = k * q`) with Euclidean distance.
- Radius threshold scales with shell (`epsilon_k = epsilon_1 * k`).
- Shell 1 can keep HNSW enabled; shells 2+ default to plain search behavior.
- Promotion/demotion/eviction are built into runtime adaptation.

In short: this fork keeps the Qdrant storage and API model, but overlays a shell-based quality layer for repeated semantic workloads.

## Mental model

Think of each point as having:
- a semantic direction (embedding meaning), and
- a trust radius level (shell index).

RPI keeps direction but uses shell index as quality:
- lower shell index => higher trust and faster path,
- higher shell index => lower trust and candidate for demotion/eviction.

The system converges by repeatedly rewarding useful points (promotion) and pushing weak points down (demotion).

## Data model and shell layout

When RPI is enabled on collection creation, the collection gets extra named vectors:
- `rpi_shell_1`
- `rpi_shell_2`
- ... up to `rpi_shell_{max_shells}`

Behavioral rules:
- New points enter at shell 1.
- Each shell vector uses Euclidean distance.
- Shell 1 may use HNSW (configurable).
- Higher shells use non-HNSW path by design.

Internal payload metadata (`_rpi_*`) tracks per-point adaptation state:
- current shell,
- original shell,
- hit count,
- demotion count,
- last access timestamp.

## Write path

With RPI enabled:
- Upserts are transformed so points include shell vectors.
- Inserts begin in shell 1.
- Collection creation auto-generates shell vector configs from the source vector config.

Without RPI:
- Behavior is normal Qdrant behavior.

## Read path

For eligible dense nearest-neighbor queries in RPI collections:
1. Build shell-specific query for shell 1 (`using = rpi_shell_1`, scaled query, shell threshold).
2. If empty, repeat for shell 2, then shell 3, ... up to `max_shells`.
3. Return first non-empty shell result.

Important implications:
- It does **not** blend all shells into one merged top-k list.
- If shell 1 has hits, search stops there.
- Deeper shells are fallback and adaptation signals, not parallel rank fusion.

For ineligible queries (non-dense, non-nearest, etc.), normal Qdrant search path is used.

## Adaptation: promotion, demotion, eviction

RPI adaptation is threshold-based and stateful.

Promotion:
- Point must be in shell > 1.
- Point hit count must reach `promotion_threshold`.
- Then point is moved one shell up (`k -> k-1`).

Demotion:
- Triggered through the rebalance/high-depth path and pass-over logic.
- Point can move one shell down (`k -> k+1`).
- If point exceeds max shell, it is evicted from shell vectors.

Eviction:
- Happens when demotion would move beyond `max_shells`.

This gives a practical lifecycle:
- useful points gravitate toward shell 1,
- consistently weak points drift outward,
- outer-most points are removed.

## Configuration reference

RPI is configured with `rpi_config` on collection creation.

`rpi_config` fields:
- `max_shells` (default `5`, range `2..20`)
- `base_epsilon` (default `0.1`)
- `source_vector` (optional source vector name; defaults to default vector)
- `demotion_threshold` (default `2`)
- `hnsw_for_shell_one` (default `true`)
- `track_lru` (default `true`)
- `promotion_threshold` (default `10`)
- `rebalance_threshold` (default `3.0`)

Derived rule:
- `epsilon_k = base_epsilon * k`

## API compatibility notes

This fork keeps existing Qdrant behavior where possible and adds RPI as an optional extension.

Current integration notes:
- `rpi_config` can be set when creating a collection via REST and gRPC.
- `GetCollectionInfo` includes both `config.rpi_config` and `rpi_stats`.
- There is no per-request RPI toggle: RPI behavior is selected by collection configuration.
- If a collection does not have `rpi_config`, behavior is standard Qdrant behavior.

Detailed docs:
- `docs/RPI_ARCHITECTURE.md`
- `docs/RPI_GRPC_GUIDE.md`

## Runtime stats

RPI tracks operational stats, including:
- total searches,
- shell hit distribution,
- average search depth,
- promotions, demotions, evictions,
- rebalance count.

These are exposed as `rpi_stats` in collection info.

## Build and run

Build release binary:

```bash
cargo build --release -p qdrant
```

Run:

```bash
./target/release/qdrant
```

On Apple Silicon (M-series), the produced binary is `arm64` Mach-O.

## Test status in this fork

Validated in this repository state:
- strict clippy (`-D warnings`) for `collection`, `storage`, and `qdrant`
- workspace tests (`--lib --tests --bins`)
- RPI integration tests
- ignored RPI perf/quality benchmark tests run manually

RPI integration tests are under:
- `lib/collection/tests/integration/rpi_cache_logic_test.rs`
- `lib/collection/tests/integration/rpi_semantic_feedback_test.rs`
- `lib/collection/tests/integration/rpi_performance_test.rs` (ignored by default)
- `lib/collection/tests/integration/rpi_scale_quality_test.rs` (ignored by default)

## Operational guidance

Use RPI when:
- query patterns repeat,
- you have feedback or behavior signals,
- you want quality convergence over time.

Expectations:
- warm/steady shell-1 path is near baseline latency,
- convergence phases can be slower,
- quality gains are strongest in feedback-rich workloads.
