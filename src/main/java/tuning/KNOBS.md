# Tuning Knobs

This document describes the real tuning surface, not a hypothetical future list.

You need to distinguish two large groups:

- platform runtime knobs
  - they are stored in `PlatformRuntimeProfile`
  - they are tuned in platform calibration
- graph policy knobs
  - they live in `ExecutionProfile.optimizer()`
  - they are typically searched in graph autotune

## Reading Guide

Use this document if you want to understand:

- what is actually calibrated per hardware today
- what is workload-specific candidate mutation
- which knobs are already public and which are only reserved
- which candidate ranges are currently used by standard calibration presets

## Runtime Vs Graph Policy

### Platform Runtime Knobs

These include:

- CPU thresholds
- tiles
- microkernels
- scheduler policy
- materialization thresholds
- numerics policy

These knobs are meant to be shared across workloads on a given machine.

### Graph Policy Knobs

These include:

- `optimizer.stageOrder`
- `optimizer.rewrite.*`
- `optimizer.fuse.*`
- `optimizer.memory.*`

These knobs are workload-sensitive. They are not part of `PlatformRuntimeProfile`.

## Public Runtime Families

### `MATMUL`

This family currently includes:

- `runtime.blas.provider`
  - in practice today mainly `NONE` or `OPENBLAS_FFM`
- `runtime.blas.matmulMinWork`
- `runtime.blas.f32RequireMgeK`
- `runtime.blas.f32MaxNOverK`
- `cpu.matMulParallelMinSize`
- `cpu.matMulTileM`
- `cpu.matMulTileN`
- `cpu.matMulTileK`
- `cpu.matMulMicroKernel`
- `cpu.attentionMatMulTileM`
- `cpu.attentionMatMulTileN`
- `cpu.attentionMatMulTileK`
- `cpu.attentionMatMulMicroKernel`

A knob that is stored in the runtime profile, but not swept by today's standard calibration presets:

- `cpu.loopUnrollFactor`

### `CONV2D_GEMM_DISPATCH_F64`

- `runtime.conv2d.blasProvider`
- `runtime.conv2d.f64MinWork`

### `CONV2D_GEMM_DISPATCH_F32`

- `runtime.conv2d.blasProvider`
- `runtime.conv2d.f32MinWork`
- `runtime.conv2d.f32RequireMgeK`
- `runtime.conv2d.f32MaxNOverK`

### `CONV2D_GEMM_DISPATCH_BF16`

- `runtime.conv2d.blasProvider`
- `runtime.conv2d.bf16MinWork`
- `runtime.conv2d.bf16RequireMgeK`
- `runtime.conv2d.bf16MaxNOverK`

### `FUSED_THRESHOLDS`

- `cpu.fusedCheapVectorMinSize`
- `cpu.fusedTranscendentalVectorMinSize`
- `cpu.fusedCheapParallelMinSize`
- `cpu.fusedTranscendentalParallelMinSize`

These control scheduler decisions for fused nodes, not backend selection.

### Fused ASM Width Knobs

These are now genuinely part of the tuning surface:

- `cpu.fusedCheapContiguousAsmVectorWidth`
- `cpu.fusedCheapStridedAsmVectorWidth`
- `cpu.fusedNonCheapContiguousAsmVectorWidth`
- `cpu.fusedNonCheapStridedAsmVectorWidth`

These width knobs are calibrated per dispatch family, not as one global number.

That is an important current reality:

- they are no longer just internal experimental settings
- standard platform calibration can search them

### `ELEMENTWISE_DISPATCH`

- `cpu.cheapVectorMinSize`
- `cpu.transcendentalVectorMinSize`
- `cpu.cheapParallelMinSize`
- `cpu.transcendentalParallelMinSize`

This applies to non-fused elementwise kernel families.

### `REDUCTION`

- `cpu.reductionVectorMinSize`
- `cpu.reductionParallelMinSize`
- `cpu.attentionVectorMinSize`
- `cpu.attentionParallelMinSize`
- `cpu.sumAccuracyMode`

The `attention*` thresholds are stored in the reduction profile family because they belong to structured reduction-like kernels, not generic fused/elementwise dispatch.

### `SCHEDULER`

- `cpu.lowCostTargetChunksPerWorker`
- `cpu.mediumCostTargetChunksPerWorker`
- `cpu.highCostTargetChunksPerWorker`
- `cpu.minScalarChunkSize`
- `cpu.minVectorChunkSize`
- `cpu.minReductionChunkSize`
- `cpu.commonPoolLowCostMaxWorkPerWorker`

These knobs only matter after the decision to run in parallel has already been made.

### `MATERIALIZATION`

- `cpu.contiguousMaterializeThreshold`

This decides from what size it becomes more beneficial to materialize a non-contiguous input into contiguous temporary storage.

### `NUMERICS`

- `runtime.approximation.approxMode`
- `runtime.approximation.forceExactTranscendentals`

These are public runtime policy knobs, not just local benchmark hacks.

## Current Calibration Ranges

The following ranges describe what standard `PlatformCalibrationDefaults` currently use, not every conceivable value.

### Matmul

`blasThreads`

- compatibility-only placeholder
- canonicalized to `0`
- runtime does not call BLAS thread setters; the provider keeps its own internal auto policy

`matMulParallelMinSize`

- `100_000`
- `500_000`
- `2_000_000`

`f32ShapeHeuristics`

- `f32RequireMgeK`
  - `true`
  - `false`
- `f32MaxNOverK`
  - `1.5`
  - `2.0`
  - `3.0`
  - `4.0`
  - `6.0`

`matMulMicroKernel`

- `FLOAT64`
  - `F64_2X1`
  - `F64_4X1`
  - `F64_2X2`
- `FLOAT32`
  - `F32_2X4`
  - `F32_2X8`
  - `F32_4X2`
  - `F32_4X4`

`matMulTiles`

- `FLOAT64`
  - `16x64x32`
  - `32x64x32`
  - `32x64x64`
  - `32x128x64`
- `FLOAT32`
  - `32x64x64`
  - `32x128x64`
  - `64x128x64`
  - `64x128x128`
  - `64x256x128`

`attentionMatMulTiles`

- `FLOAT64`
  - `16x64x32`
  - `32x64x32`
  - `32x128x64`
- `FLOAT32`
  - `32x64x64`
  - `32x128x64`
  - `64x128x64`
  - `64x128x128`
  - `64x256x128`

### Fused Thresholds

`cheapVector`

- `64`
- `128`
- `256`
- `512`
- `1024`

`transcendentalVector`

- `16`
- `32`
- `64`
- `128`
- `256`

`cheapParallel`

- `4096`
- `8192`
- `16384`
- `32768`

`transcendentalParallel`

- `1024`
- `2048`
- `4096`
- `8192`

### Fused ASM Widths

Candidate widths are derived from dtype and the available preferred vector species:

- always `1`
- if the hardware allows it, then also `2`
- if the hardware allows it, then also `4`
- in some `F32/BF16 cheap contiguous` cases also `8`

That means:

- width space is family-specific
- it is not one universal number for all fused workloads

### Elementwise Dispatch

`cheapVector`

- `128`
- `256`
- `512`
- `1024`
- `2048`

`transcendentalVector`

- `32`
- `64`
- `128`
- `256`
- `512`

`cheapParallel`

- `8192`
- `16384`
- `32768`
- `65536`

`transcendentalParallel`

- `2048`
- `4096`
- `8192`
- `16384`

### Reduction

`reductionVector`

- `512`
- `2048`
- `8192`
- `16384`

`reductionParallel`

- `8192`
- `16384`
- `32768`
- `65536`

`attentionThresholds`

- `attentionVector`
  - `512`
  - `2048`
  - `8192`
  - `16384`
- `attentionParallel`
  - `2048`
  - `8192`
  - `16384`
  - `32768`

### Scheduler

Scheduler calibration today usually performs only local refinement around the current seed winner values:

- target chunks per worker
- minimum chunk sizes
- common pool threshold

That is intentional. Scheduler knobs tend to depend strongly on whatever has already won in the other families.

### Materialization

The candidate set is built around the current threshold and extended with explicit anchor points:

- `262_144`
- `524_288`
- `1_048_576`

### Numerics

`approxMode`

- `OFF`
- `TRAINING_ONLY`
- `ALWAYS`

`forceExactTranscendentals`

- `true`
- `false`

## Graph Policy Knobs

These knobs are not part of the platform runtime profile, but they are part of `ExecutionProfile.optimizer()`.

### `optimizer.stageOrder`

Valid stage elements:

- `AR`
- `CSE`
- `FUSE`
- `MEM`

Candidate spaces commonly use:

- an explicit list of stage orders
- a constrained stage-order space
- an exhaustive permutation/subset space

### Rewrite Policy

Today the main meaningful knob to tune is:

- `optimizer.rewrite.conv2dLowering.mode`
  - `OFF`
  - `HEURISTIC`
  - `ALWAYS`

Piecewise lowering config also exists, but in normal autotune workflow it is not usually the primary knob surface.

## Knobs That Exist But Are Not Very Useful Today

### `runtime.fused.primaryBackend`

It technically exists in fused execution policy.

The practical CPU reality today:

- the meaningful backend is `ASM`

So this is not an especially interesting knob for standard tuning.

### `runtime.fused.allowBackendFallback`

It technically exists, but because CPU fused prepare currently relies on the ASM backend, it is not a major performance lever.

### `cpu.loopUnrollFactor`

It is stored in the runtime profile, but standard platform calibration does not currently sweep it.

## Example: Runtime Profile Candidate

A platform calibration candidate may change for example:

- `cpu.matMulTileM/N/K`
- `cpu.matMulMicroKernel`
- `cpu.matMulParallelMinSize`

but it is still one concrete `PlatformRuntimeProfile`, not a separate knob map.

## Example: Graph Candidate

A graph autotune candidate may change:

- `optimizer.stageOrder`

or for a `CONV2D` workload:

- `optimizer.rewrite.conv2dLowering.mode`

The result is again a normal `ExecutionProfile`.

## Common Mistakes

- treating runtime knobs as workload-specific graph policy
- assuming fused ASM widths are not part of the public tuning surface
- wanting to calibrate `runtime.fused.primaryBackend`, even though only `ASM` makes sense on CPU today
- ignoring that some stored runtime fields are not currently swept in standard presets

## Related Docs

- architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
- search: [SEARCH.md](./SEARCH.md)
- persistence: [PERSISTENCE.md](./PERSISTENCE.md)

## How To Read A Knob

Every knob should be read through three questions:

1. who owns it?
2. what decision does it control?
3. which workflow searches it?

Example:

- `cpu.fusedCheapVectorMinSize`
  - owner: `FUSED_THRESHOLDS`
  - decision: when cheap fused kernels become eligible for vector execution
  - searched by: platform calibration

Example:

- `optimizer.stageOrder`
  - owner: graph policy
  - decision: which optimizer stages run, and in which order
  - searched by: graph autotune, not platform calibration

## Practical Knob Categories

### Threshold knobs

These answer:

- "from what size does this tactic become worthwhile?"

Examples:

- vector thresholds
- parallel thresholds
- BLAS minimum work
- materialization thresholds

### Shape-gate knobs

These answer:

- "for this shape family, should the aggressive path even be considered?"

Examples:

- `runtime.blas.f32RequireMgeK`
- `runtime.blas.f32MaxNOverK`
- `runtime.conv2d.f32RequireMgeK`
- `runtime.conv2d.f32MaxNOverK`

### Structural selection knobs

These choose among several runtime implementations of the same primitive.

Examples:

- matmul microkernel
- matmul tiles
- attention matmul tiles
- approximation mode

## Example: BLAS Crossover

This runtime choice is driven mainly by:

- `runtime.blas.provider`
- `runtime.blas.matmulMinWork`
- `runtime.blas.f32RequireMgeK`
- `runtime.blas.f32MaxNOverK`

Interpretation example:

```text
provider = OPENBLAS_FFM
matmulMinWork = 2_000_000
f32RequireMgeK = true
f32MaxNOverK = 3.0
```

Meaning:

- BLAS is only eligible once work reaches `2_000_000`
- for `FLOAT32`, very wide-but-shallow shapes may still be rejected by the shape gates
