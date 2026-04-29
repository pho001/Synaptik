# Tuning Package

The `tuning` package owns measurement, search, validation, reporting, and persistence for executable profiles.

Its job is to answer practical questions such as:

- which `ExecutionProfile` is faster on this workload?
- which runtime defaults should be calibrated for this machine?
- how should candidate profiles be measured fairly?
- where should winners and runtime defaults be persisted?

It does not define execution semantics.

Execution still happens through the normal runtime stack:

- `Tensor`
- `CompiledGraph`
- `PreparedExecution`
- backend execution

## Three Distinct Workflows

The package currently supports three different workflows.

### 1. Benchmark

Benchmark compares an explicit list of candidates.

Typical use:

- compare baseline vs optimized profile
- compare explicit candidate variants
- inspect regressions with trace detail

Benchmark does not search the space automatically.

### 2. Per-graph autotune

Autotune evaluates candidate `ExecutionProfile` objects for one concrete workload.

Typical use:

- start from explicit graph policy plus calibrated runtime profile
- generate candidate profiles from a typed candidate space
- validate and measure them
- persist the winner and search history

Current production graph autotune intentionally starts from a small standard graph-policy candidate set:

- `graphPolicy=current`
- CPU region policy variants
- CPU fusion policy variants
- accelerator ownership policy variants

Research graph variants are explicit opt-in and are not production best-profile candidates by default.

### 3. Platform calibration

Calibration searches runtime defaults that should be reused across workloads on one hardware/JDK platform.

Typical use:

- tune thresholds, tiles, dispatch cutoffs, and numeric policy once per machine
- persist a `PlatformRuntimeProfile`
- later assemble executable profiles from that runtime profile plus graph policy

## Source Of Truth

The executable source of truth is:

- [../config/profile/ExecutionProfile.java](../config/profile/ExecutionProfile.java)

That principle is crucial.

- benchmark measures real `ExecutionProfile` objects
- autotune searches real `ExecutionProfile` objects
- calibration produces `PlatformRuntimeProfile`, which is later assembled into real `ExecutionProfile` objects

There should be no hidden benchmark-only execution model.

## Package Map

- architecture and ownership:
  - [ARCHITECTURE.md](./ARCHITECTURE.md)
- runtime and graph knob surface:
  - [KNOBS.md](./KNOBS.md)
- persistence layout:
  - [PERSISTENCE.md](./PERSISTENCE.md)
- reporting:
  - [REPORTING.md](./REPORTING.md)
- search:
  - [SEARCH.md](./SEARCH.md)
- workloads:
  - [WORKLOADS.md](./WORKLOADS.md)
- historical context:
  - [LEGACY-BENCHMARK-REVIEW.md](./LEGACY-BENCHMARK-REVIEW.md)

## Main Runtime Concepts

### `ExecutionProfile`

This is the runnable artifact benchmarked and autotuned.

It contains:

- dtype
- execution mode
- optimizer policy
- runtime config
- workload metadata

### `PlatformRuntimeProfile`

This is the machine-oriented runtime-default artifact produced by platform calibration.

It contains runtime families such as:

- matmul
- fused dispatch
- elementwise dispatch
- reduction
- scheduler
- materialization
- numerics
- conv2d dispatch

It does not contain graph stage order.

### `GraphExecutionPolicy`

This is the graph-side policy layer, effectively the optimizer configuration family.

It contains:

- stage order
- rewrite config
- CSE config
- fuse config
- memory config

## Typical Flow

For the main CLI flow:

1. calibrate platform runtime profile
2. assemble an execution profile seed from calibrated runtime + graph policy
3. run graph autotune for the workload
4. benchmark the winner against the baseline

That means:

- calibration and autotune are not the same thing
- calibration mostly answers runtime questions
- standard graph autotune validates production graph-policy candidates and persists the best graph-policy winner for the workload

## Example: Workload-Specific Autotune

Suppose the workload is the built-in `abc_sequence_matmul_f64`.

Autotune flow:

1. load calibrated runtime defaults for `FLOAT64` forward/backward
2. combine calibrated runtime defaults with the current graph policy
3. generate the standard production graph-policy candidates
4. validate correctness
5. measure compile / prepare / traced run / steady-state
6. persist:
   - best profile
   - history
   - reports

## Example: Platform Calibration

Suppose you calibrate only the `matmul` family for `f64`.

The calibration step may search across combinations such as:

- BLAS provider
- BLAS min-work threshold
- Java matmul tile sizes
- Java microkernel variant
- matmul parallel threshold

The result is not a graph winner.
It is a machine-specific runtime profile that future executable profiles can reuse.

## Important Ownership Rules

The tuning layer follows these rules:

- search chooses candidates, not execution semantics
- validation rejects numerically or semantically invalid candidates
- measurement records timings, not correctness
- persistence distinguishes:
  - runtime defaults
  - best runnable profile
  - history
  - explain artifacts

If those layers get mixed, the system drifts.

## Current Public Calibration Families

The CLI currently exposes family-specific calibration steps such as:

- `matmul`
- `attention-matmul`
- `conv2d`
- `fused-thresholds`
- `fused-cheap-contiguous`
- `fused-cheap-strided`
- `fused-noncheap-contiguous`
- `fused-noncheap-strided-width`
- `elementwise-dispatch`
- `reduction`
- `attention-thresholds`
- `scheduler`
- `materialization`
- `metal-selection` only as explicit accelerator opt-in
