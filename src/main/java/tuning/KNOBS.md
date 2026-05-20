# Tuning Knobs

This document describes the current public tuning surface, not a hypothetical wishlist.

You should always distinguish two groups:

- platform runtime knobs
  - stored in `PlatformRuntimeProfile`
  - primarily calibrated per machine
- graph policy knobs
  - stored in `CompileConfig` through `GraphExecutionPolicy`
  - primarily searched by graph autotune

## Runtime Families

### `MATMUL`

Current public matmul-related runtime knobs include:

- `runtime.blas.provider`
- `runtime.blas.matmulMinWork`
- `runtime.blas.f32RequireMgeK`
- `runtime.blas.f32MaxNOverK`
- `runtime.blas.f32WideRequireMgeK`
- `runtime.blas.f32WideMaxNOverK`
- `cpu.matMulParallelMinSize`
- `cpu.matMulTileM`
- `cpu.matMulTileN`
- `cpu.matMulTileK`
- `cpu.matMulMicroKernel`
- `cpu.attentionMatMulTileM`
- `cpu.attentionMatMulTileN`
- `cpu.attentionMatMulTileK`
- `cpu.attentionMatMulMicroKernel`

Important current behavior:

- `runtime.blas.threads` is canonicalized to `0`
- `0` means provider-managed auto behavior
- Synaptik does not calibrate provider thread counts while this canonicalization exists

### `CONV2D_GEMM_DISPATCH`

Current conv2d dispatch knobs:

- `runtime.conv2d.blasProvider`
- `runtime.conv2d.f64MinWork`
- `runtime.conv2d.f32MinWork`
- `runtime.conv2d.f32RequireMgeK`
- `runtime.conv2d.f32MaxNOverK`
- `runtime.conv2d.bf16MinWork`
- `runtime.conv2d.bf16RequireMgeK`
- `runtime.conv2d.bf16MaxNOverK`

These affect lowered GEMM conv2d nodes, not the semantic decision to lower conv2d in the optimizer.

### `FUSED_DISPATCH`

- `cpu.fusedCheapVectorMinSize`
- `cpu.fusedTranscendentalVectorMinSize`
- `cpu.fusedCheapParallelMinSize`
- `cpu.fusedTranscendentalParallelMinSize`

These decide when fused nodes become worth vectorization or parallelization.

### Fused ASM Width Knobs

Current fused width knobs are family-specific:

- `cpu.fusedCheapContiguousAsmVectorWidth`
- `cpu.fusedCheapStridedAsmVectorWidth`
- `cpu.fusedNonCheapContiguousAsmVectorWidth`
- `cpu.fusedNonCheapStridedAsmVectorWidth`

These are real public runtime knobs now.
They are no longer just internal experiment flags.

### `ELEMENTWISE_DISPATCH`

- `cpu.cheapVectorMinSize`
- `cpu.transcendentalVectorMinSize`
- `cpu.cheapParallelMinSize`
- `cpu.transcendentalParallelMinSize`

These affect non-fused elementwise planning.

### `REDUCTION`

- `cpu.reductionVectorMinSize`
- `cpu.reductionParallelMinSize`
- `cpu.sumAccuracyMode`

### `ATTENTION_THRESHOLDS`

- `cpu.attentionVectorMinSize`
- `cpu.attentionParallelMinSize`

### `SCHEDULER`

- `cpu.lowCostTargetChunksPerWorker`
- `cpu.mediumCostTargetChunksPerWorker`
- `cpu.highCostTargetChunksPerWorker`
- `cpu.minScalarChunkSize`
- `cpu.minVectorChunkSize`
- `cpu.minReductionChunkSize`
- `cpu.commonPoolLowCostMaxWorkPerWorker`

### `MATERIALIZATION`

- `cpu.contiguousMaterializeThreshold`
- `cpu.cheapF64MaterializeThreshold`
- `cpu.cheapF32MaterializeThreshold`
- `cpu.cheapBF16MaterializeThreshold`
- `cpu.whereMaterializeThreshold`

## Graph Policy Knobs

These are graph-side configuration fields. They are not automatically production autotune knobs.

Current standard production graph autotune exposes only:

- `graphPolicy=current`

The following graph-resident fields are research-only or excluded from standard graph autotune:

- `compile.graphOptimization.*`
  - graph simplification/lowering contract, not a broad production tuning axis
- `compile.graphOptimization.rewrite.conv2dLowering.mode`
  - GEMM/BLAS/runtime-family proxy
- `compile.regionOptimization.fuse.*`
  - persisted fields, but current production region optimization does not consume their scoring values
- `compile.backendPlanning.search.*`
  - accelerator/backend partition scoring and target policy
- `compile.graphOptimization.cse.strictSafety`
  - research-only safety/correctness policy
- `compile.graphOptimization.rewrite.piecewiseLowering.*`
  - research-only import/manual-graph canonicalization policy
- `compile.memoryPlanning.*`
  - research-only at most; size/reuse fields are memory-system proxies

## Current Standard Calibration Ranges

The following are the current standard preset ranges used by `PlatformCalibrationDefaults`, not an exhaustive universe of possible values.

### Matmul

BLAS provider candidates:

- `NONE`
- `OPENBLAS_FFM`

BLAS min-work candidates:

- `1_000_000`
- `2_000_000`
- `4_000_000`

Matmul parallel threshold candidates:

- `100_000`
- `500_000`
- `2_000_000`

F32 shape heuristics:

- `f32RequireMgeK`
  - `true`
  - `false`
- `f32MaxNOverK`
  - `1.5`
  - `2.0`
  - `3.0`
  - `4.0`
  - `6.0`

Microkernel candidates:

- `FLOAT64`
  - `F64_2X1`
  - `F64_4X1`
  - `F64_2X2`
- `FLOAT32`
  - `F32_2X4`
  - `F32_2X8`
  - `F32_4X2`
  - `F32_4X4`

Tile candidates:

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

### Attention matmul

Microkernel candidates:

- same dtype-specific family as matmul

Tile candidates:

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

### Fused thresholds

Current standard threshold candidates:

- cheap vector min size:
  - `64`, `128`, `256`, `512`, `1024`
- transcendental vector min size:
  - `16`, `32`, `64`, `128`, `256`
- cheap parallel min size:
  - `4096`, `8192`, `16384`, `32768`
- transcendental parallel min size:
  - `1024`, `2048`, `4096`, `8192`

### Fused ASM widths

Current candidates depend on dtype:

- `FLOAT64`
  - `1`
  - `2`
  - `4`
- `FLOAT32`
  - `1`
  - `2`
  - `4`
  - `8`
- `BFLOAT16`
  - `1`
  - `2`
  - `4`
  - `8`

The family-specific width calibration uses workload families such as:

- cheap contiguous
- cheap strided
- non-cheap contiguous
- non-cheap strided

### Elementwise dispatch

Current standard candidates:

- vector thresholds:
  - `64`, `128`, `256`, `512`, `1024`, `2048`
- parallel thresholds:
  - `4096`, `8192`, `16384`, `32768`, `65536`

### Reduction

Current standard candidates include:

- vector min size:
  - `64`, `128`, `256`, `512`, `1024`
- parallel min size:
  - `4096`, `8192`, `16384`, `32768`, `65536`
- accuracy:
  - `FAST`
  - `KAHAN`

### Scheduler

Current standard candidates include combinations over:

- chunks-per-worker targets
- scalar/vector/reduction minimum chunk sizes
- common-pool low-cost max work per worker

These are designed to tune the planner's chunking decisions rather than semantic graph structure.

## Worked Examples

### Example 1: matmul BLAS threshold

Suppose:

- candidate A uses `runtime.blas.matmulMinWork = 1_000_000`
- candidate B uses `runtime.blas.matmulMinWork = 4_000_000`

Then:

- A will route more medium-sized matmuls to BLAS
- B will keep more of them on the Java microkernel path

This is a runtime policy difference, not an optimizer-stage difference.

### Example 2: fused ASM width

Suppose a fused cheap contiguous workload is calibrated for `FLOAT32`.

Candidates might include:

- width `1`
- width `2`
- width `4`
- width `8`

The winner is persisted into the platform runtime profile and later reused by prepared fused execution.

## Knobs That Exist But Need Careful Interpretation

### `runtime.blas.threads`

This knob exists in the public config/profile surface for compatibility and explicitness, but the effective current behavior is:

- store only `0`
- treat `0` as auto/provider-managed

So documentation should not claim that calibration actively orchestrates provider thread counts at runtime.
