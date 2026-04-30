# Tuning Architecture

The tuning layer is built around one hard rule:

- tuning must never invent a second execution model outside the normal runtime stack

Every measured candidate must be a genuinely runnable profile.

## Core Artifacts

You should distinguish four artifacts.

### 1. `PlatformRuntimeProfile`

Machine-specific runtime defaults.

Contains runtime families such as:

- matmul
- fused dispatch
- elementwise dispatch
- reduction
- scheduler
- materialization
- conv2d dispatch
- numerics

Does not contain:

- optimizer stage order
- workload-specific best graph policy

### 2. `GraphExecutionPolicy`

Graph policy layer.

Today this is effectively the optimizer configuration family:

- stage order
- rewrite config
- CSE config
- fuse config
- memory config

### 3. `ExecutionProfile`

Actual runnable artifact.

Assembled from:

- graph policy
- runtime profile
- dtype
- execution mode
- workload metadata

### 4. Persistence / explain artifacts

Examples:

- best profile records
- tuning history
- calibration reports
- benchmark reports

These are not direct runtime sources of truth.

For graph autotune, a best profile record is a workload graph-policy winner plus
measurement evidence. The embedded `ExecutionProfile` records the runtime used
when the candidate was measured, but future execution must reassemble the saved
graph policy with the current `PlatformRuntimeProfile`. This keeps calibration
per platform/dtype/mode and graph autotune per graph/workload.

## Assembly Boundary

The correct assembly point for a final executable profile is:

- [../config/profile/ExecutionProfileAssembler.java](../config/profile/ExecutionProfileAssembler.java)

This boundary matters because:

- calibration mutates `PlatformRuntimeProfile`
- graph autotune assembles and evaluates explicit `ExecutionProfile` candidates
- runtime executes `ExecutionProfile`
- winner loading for graph autotune extracts the persisted graph policy and
  rebases it on the latest platform runtime profile before benchmarking

Current standard graph autotune exposes production graph-policy variants for CPU
region policy, CPU fusion policy, and accelerator ownership policy while keeping
runtime frozen. Graph-resident fields such as arbitrary stage order, conv2d
lowering, fusion scoring, and partition scoring are not standard graph autotune
axes.

## Workflow Ownership

## Tuning knob ownership matrix

Phase 4 made tuning knob ownership explicit so graph autotune and platform
calibration do not search the same decision from different directions.

| Owner | Meaning | Examples |
|---|---|---|
| Graph/workload-owned | Workload-specific optimizer or graph policy selected by graph autotune. | graph offload policy, accelerator region strategy, CPU region policy, CPU fusion policy |
| Platform/dtype-owned | Hardware, dtype, execution-mode, and runtime thresholds selected by platform calibration. | BLAS thresholds, vector/parallel thresholds, fused dispatch widths, scheduler thresholds, `ACCELERATOR_BUFFER_MODE`, `METAL_SELECTION` |
| Obsolete | Historical knobs that must not re-enter production candidate spaces. | duplicate graph/runtime aliases and legacy report-derived policy |

`METAL_SELECTION` is explicit accelerator opt-in calibration, not default CPU
calibration. Accelerator transfer estimates such as `MetalTransferModel` and
runtime thresholds belong to platform/dtype-owned runtime policy. Graph autotune
may select graph policies that make accelerator regions possible, but it must not
rewrite platform calibration thresholds.

Profile-derived accelerator costs enter through RuntimeConfig, not profile file reads.

### Benchmark

Owns:

- explicit candidate comparison
- measurement
- reporting

Does not own:

- candidate-space search strategy
- platform runtime calibration

### Autotune

Owns:

- candidate-space exploration
- validation
- measurement
- winner persistence

Usually mutates:

- graph-side policy
- or explicit runtime/profile variants supplied by the candidate space

### Calibration

Owns:

- platform-family workload selection
- platform-family candidate spaces
- step-by-step runtime-default search

Produces:

- `PlatformRuntimeProfile`

## Search / Measure / Validate Split

The architecture is intentionally decomposed into sublayers:

- `candidate`
  - candidate generation / mutation
- `search`
  - ordering and bounded exploration
- `validate`
  - semantic/numeric correctness guardrail
- `measure`
  - timing policy and statistics
- `report`
  - explain artifacts
- `store`
  - persistence
- `session`
  - orchestration

This split replaces the older monolithic benchmark/autotune style where one huge class tried to do everything.

## Why Runtime Knobs Stay Outside The Optimizer

The optimizer changes graph structure.
Runtime knobs affect prepared execution policy.

Examples of graph policy:

- `AR -> CSE -> FUSE -> MEM`
- piecewise lowering enabled/disabled
- conv2d lowering mode

Examples of runtime policy:

- matmul BLAS threshold
- matmul microkernel
- fused ASM width
- reduction vector threshold
- scheduler chunk targets
- approximation mode

If runtime thresholds were pushed into optimizer stages, the architecture would blur compile-time semantics with hardware/runtime policy.

## Current Family Surface

The calibration registry currently exposes these production family ids:

- `MATMUL`
- `ATTENTION_MATMUL`
- `CONV2D_GEMM_DISPATCH`
- `FUSED_DISPATCH`
- `FUSED_CHEAP_CONTIGUOUS_WIDTH`
- `FUSED_CHEAP_STRIDED_WIDTH`
- `FUSED_NON_CHEAP_CONTIGUOUS_WIDTH`
- `FUSED_NON_CHEAP_STRIDED_WIDTH`
- `ELEMENTWISE_DISPATCH`
- `REDUCTION`
- `ATTENTION_THRESHOLDS`
- `SCHEDULER`
- `MATERIALIZATION`
- `METAL_SELECTION` as explicit accelerator opt-in only

Important current reality:

- conv2d dispatch is one dtype-aware family, not three public dtype-specific ids
- numerical policy is not a timing-selected calibration family
- Metal selection is never part of default CPU calibration

## BLAS Thread Policy

Current public BLAS thread policy is intentionally canonicalized to:

- `threads = 0`

Meaning:

- provider-managed auto behavior
- Synaptik does not try to own global process-wide BLAS thread counts as a runtime orchestration feature

This is an architectural decision, not an accidental omission.

## Preferred Persistence Model

The preferred persistent layout is:

```text
profiles/platform/<platform-id>/...
```

That makes calibration and autotune artifacts versionable together with the repository if desired, while still preserving compatibility fallbacks from older `build/...` locations.

## Architectural Anti-Patterns To Avoid

These should not return under a new name:

- benchmark-only execution models
- hidden candidate abstractions that runtime never executes
- reports acting as runtime source of truth
- executor-owned tuning policy that bypasses compile/prepare boundaries
- global mutable runtime side effects treated as profile-local state
