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

## Assembly Boundary

The correct assembly point for a final executable profile is:

- [../config/profile/ExecutionProfileAssembler.java](../config/profile/ExecutionProfileAssembler.java)

This boundary matters because:

- calibration mutates `PlatformRuntimeProfile`
- graph autotune compares `ExecutionProfile`
- runtime executes `ExecutionProfile`

## Workflow Ownership

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

The calibration enum currently contains:

- `MATMUL`
- `ATTENTION_MATMUL`
- `FUSED_THRESHOLDS`
- `FUSED_CHEAP_CONTIGUOUS`
- `FUSED_CHEAP_STRIDED`
- `FUSED_NON_CHEAP_CONTIGUOUS`
- `FUSED_NON_CHEAP_STRIDED`
- `FUSED_ARITHMETIC`
- `ELEMENTWISE_DISPATCH`
- `REDUCTION`
- `ATTENTION_THRESHOLDS`
- `SCHEDULER`
- `MATERIALIZATION`
- `CONV2D_GEMM_DISPATCH_F64`
- `CONV2D_GEMM_DISPATCH_F32`
- `CONV2D_GEMM_DISPATCH_BF16`
- `NUMERICS`

Important current reality:

- `FUSED_ARITHMETIC` exists in the enum
- standard `PlatformCalibrationDefaults` do not currently expose it as a normal preset step

So documentation should describe it as reserved/not-in-standard-presets, not as a commonly run family.

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
