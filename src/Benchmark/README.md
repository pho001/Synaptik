# Benchmark (src/Benchmark)

## Purpose

The Benchmark module measures optimizer/runtime performance, validates correctness against baseline, and autotunes stage order + tuning knobs.

Primary goals:

- compare optimizer stage combinations on the same workload
- report forward/training latency and speedups
- enforce numeric equivalence checks
- persist winning profiles for runtime reuse
- cache context-specific unsafe candidates to prune future autotune runs

## Main Components

- Entry framework:
  - [src/Benchmark/OptimizerBenchmarkFramework.java](../Benchmark/OptimizerBenchmarkFramework.java)
- Candidate model/factory:
  - [src/Benchmark/OptimizerCandidate.java](../Benchmark/OptimizerCandidate.java)
  - [src/Benchmark/OptimizerCandidateFactory.java](../Benchmark/OptimizerCandidateFactory.java)
  - [src/Benchmark/OptimizationStage.java](../Benchmark/OptimizationStage.java)
- Optimizer assembly:
  - [src/Benchmark/OptimizerBuilder.java](../Benchmark/OptimizerBuilder.java)
- Knobs and profile I/O:
  - [src/Benchmark/TuningKnobs.java](../Benchmark/TuningKnobs.java)
  - [src/Benchmark/OptimizerProfileIO.java](../Benchmark/OptimizerProfileIO.java)

## Run Flow

1. Build default candidates (`NO_OPT`, staged variants, `RECOMMENDED`, `INFERENCE_PERF`).
2. Apply profile chain to recommended/inference candidates:
  - persisted best-profile overrides
  - architecture presets (`os.arch`, including ARM/aarch64 and x86_64/amd64)
  - HW-bucket override for current machine (`config/optimizer-hw-profiles.tsv`)
3. Run scalar sanity check.
4. Run main benchmark:
  - forward-only measurement
  - training measurement (forward + backward)
  - correctness diffs versus baseline
5. Run per-stage breakdown table.
6. Optionally run two-phase autotune.

## Candidate Semantics

- `NO_OPT`: empty stage list, executed via explicit empty `GraphOptimizer` baseline.
- `RECOMMENDED`: training-oriented candidate, finalized via profile chain.
- `INFERENCE_PERF`: inference-oriented candidate, finalized via profile chain.

At runtime and benchmark startup, final candidate params are selected with this priority:

1. HW-bucket profile (`optimizer-hw-profiles.tsv`) for current bucket.
2. Architecture preset (`os.arch`).
3. Persisted best-profile override (`best-profile-*.json`).
4. Built-in defaults.

## Correctness Checks

Checks compare:

- output tensor (`Ta7`)
- gradients (`gradA`, `gradB`, `gradC`)

Comparison uses absolute and relative tolerances and rejects invalid (`NaN`/`Inf`) values.

## Two-Phase Autotune

Phase 1:

- broad candidate screening
- quick warmup + measurement
- baseline correctness validation

Phase 2:

- re-measure top finalists with heavier warmup/measurement
- compute final objective score

Persist behavior:

- save improved training winner:
  - `build/optimizer-autotune/best-profile-training.json`
  - `config/optimizer-profile.json` (runtime training profile)
- save improved inference winner:
  - `build/optimizer-autotune/best-profile-inference.json`
- update improved HW-bucket entries:
  - `config/optimizer-hw-profiles.tsv` (`TRAINING` + `INFERENCE`)
- keep previous winner if new score is not better

Unsafe-candidate cache:

- `build/optimizer-autotune/candidate-history.tsv`
- stores `UNSAFE` fingerprints (currently mismatch-based) with context signature
- context includes dtype, tolerances, workload shape/size, schema/engine versions, OS/arch/JVM/vendor/core count
- candidates marked unsafe in matching context are skipped on next autotune run

## Tuning Knobs

`TuningKnobs` bundles:

- optimizer policy knobs (`strictCseSafety`, `FuseConfig`)
- backend/kernel knobs (`KernelTuningConfig` for `cpu`, `cuda`, `opencl`)

## Full Knob Space (Current Implementation)

This section reflects the current candidate generation in:

- [src/Benchmark/OptimizerCandidateFactory.java](../Benchmark/OptimizerCandidateFactory.java)

### Stage Order

Autotune generates all subsets and permutations of:

- `stages` [AR, CSE, FUSE, MEM]

Total stage-order variants per knob profile:

- `sum(P(4, k), k=0..4) = 65`

### Optimizer Knobs

- `strictCseSafety` [true, false]
  - CSE safety strictness (`true` = conservative elimination, `false` = more aggressive)
- `fuse.maxClusterNodes` [32, 64, 80, 96]
  - max number of nodes per fusion cluster
- `fuse.scoreThreshold` [0.00, 0.55, 0.60, 0.85]
  - minimum fusion score for accepting a cluster
- `fuse.internalEdgeBonus` [0.25, 0.30, 0.50]
  - score bonus for edges internal to a candidate cluster
- `fuse.externalInputPenalty` [0.10, 0.20]
  - score penalty for each external input dependency
- `fuse.sharedExpensivePenalty` [0.50, 1.00]
  - penalty when expensive nodes are shared with non-fused users
- `fuse.nonCheapBonus` [0.30, 0.35, 0.40]
  - bonus for including non-trivial operators in the cluster
- `fuse.preserveSharedExpensiveNodes` [true, false]
  - keep shared expensive nodes outside fusion to avoid duplicated cost

### CPU Kernel Knobs

- `kernel.cpu.loopUnrollFactor` [1, 4]
  - loop unrolling hint for CPU kernels
- `kernel.cpu.matMulTileM` [16, 32, 64]
  - matmul tile size M
- `kernel.cpu.matMulTileN` [0, 16, 32]
  - matmul tile size N (`0` means disabled/default path)
- `kernel.cpu.matMulTileK` [0, 16, 32]
  - matmul tile size K (`0` means disabled/default path)
- `kernel.cpu.vectorMinSize` [256, 512, 1024, 2048, 1000000000]
  - minimum tensor length to allow vector path
- `kernel.cpu.parallelMinSize` [50000, 100000, 250000, 1000000, 2000000, 1000000000]
  - minimum tensor length to allow parallel path
- `kernel.cpu.matMulParallelMinSize` [100000, 500000, 2000000, 8000000]
  - minimum matmul work size (`M*N*K`) to allow parallel matmul path
- `kernel.cpu.parallelism` [0]
  - worker count (`0` = auto from available processors)
- `kernel.cpu.chunksPerWorker` [2, 4, 8]
  - target number of chunks scheduled per worker
- `kernel.cpu.minChunkSize` [2048, 4096, 8192]
  - lower bound for one parallel chunk
- `kernel.cpu.contiguousMaterializeThreshold` [0, 4096, 16384, 65536, 262144, 1000000000]
  - non-contiguous tensors below threshold use strided path; above threshold materialize-to-contiguous
- `kernel.cpu.lowCostNsPerElementThreshold` [0.5, 1.0, 2.0, 4.0]
  - scheduler threshold to treat op as low-cost and reduce parallel overhead
- `kernel.cpu.vectorPolicyCheap` [AUTO, FORCE_ON]
  - vector dispatch policy for cheap element-wise kernels
- `kernel.cpu.vectorPolicyTranscendental` [AUTO, FORCE_OFF]
  - vector dispatch policy for transcendental kernels (`exp`, `tanh`, ...)
- `kernel.cpu.vectorPolicyReduction` [AUTO]
  - vector dispatch policy for reduction kernels
- `kernel.cpu.vectorPolicy*` full enum values [AUTO, FORCE_ON, FORCE_OFF]
  - full runtime enum; autotune currently tests only the listed subsets per group
- `kernel.cpu.sumAccuracyMode` runtime values [FAST, KAHAN, NEUMAIER]
  - reduction numerical-stability mode
- `kernel.cpu.sumAccuracyMode` autotune grid [FAST]
  - currently fixed during autotune to keep search-space size bounded

### CUDA Kernel Knobs

- `kernel.cuda.loopUnrollFactor` [4, 8]
  - loop unrolling hint for CUDA kernels
- `kernel.cuda.matMulTileM` [16, 32]
  - matmul tile size M
- `kernel.cuda.matMulTileN` [16, 32]
  - matmul tile size N
- `kernel.cuda.matMulTileK` [16, 32]
  - matmul tile size K

### OpenCL Kernel Knobs

- `kernel.opencl.loopUnrollFactor` [1, 2, 4]
  - loop unrolling hint for OpenCL kernels
- `kernel.opencl.matMulTileM` [0, 16, 32]
  - matmul tile size M (`0` = disabled/default path)
- `kernel.opencl.matMulTileN` [0, 16, 32]
  - matmul tile size N (`0` = disabled/default path)
- `kernel.opencl.matMulTileK` [0, 16]
  - matmul tile size K (`0` = disabled/default path)

## Runtime-Only Knobs (Not in Autotune Grid)

These knobs influence benchmark behavior, but they are not part of `TuningKnobs` search candidates.

- `benchmark.dtype` [FLOAT32, FLOAT64]
  - benchmark input/output tensor dtype (system property: `-Dbenchmark.dtype=FLOAT32|FLOAT64`)
- `ABS_TOL` [1e-9]
  - absolute tolerance for benchmark diff checks
- `REL_TOL` [1e-7]
  - relative tolerance for benchmark diff checks
- `SIZE` [1000000]
  - vector length for main benchmark run
- `WARMUP_ITERS` [200]
  - warmup iterations for main forward/training benchmark
- `MEASURE_ITERS` [1000]
  - measured iterations for main forward/training benchmark
- `STAGE_WARMUP_ITERS` [50]
  - warmup iterations for stage breakdown table
- `STAGE_MEASURE_ITERS` [300]
  - measured iterations for stage breakdown table
- `AUTOTUNE_SIZE` [200000]
  - vector length used during autotuning
- `AUTOTUNE_WARMUP_ITERS` [12]
  - phase-1 warmup iterations per candidate
- `AUTOTUNE_MEASURE_ITERS` [40]
  - phase-1 measured iterations per candidate
- `AUTOTUNE_MAX_CANDIDATES` [500]
  - deterministic cap of evaluated candidates from full search space
- `AUTOTUNE_REFINE_TOP_K` [8]
  - number of best candidates promoted from each objective for phase 2
- `AUTOTUNE_REFINE_WARMUP_ITERS` [50]
  - phase-2 warmup iterations
- `AUTOTUNE_REFINE_MEASURE_ITERS` [300]
  - phase-2 measured iterations
- `AUTOTUNE_REFINE_REPEATS` [3]
  - repeats used to average phase-2 timing
- `ENABLE_AUTOTUNE` [true, false]
  - enables/disables autotune phase in benchmark run

### Candidate Count Notes

Current autotune candidate construction:

- `3466` knob profiles
- `65` stage-order variants per profile
- total generated candidates: `3466 * 65 = 225290`

Benchmark cap is controlled by `AUTOTUNE_MAX_CANDIDATES` in:

- [src/Benchmark/OptimizerBenchmarkFramework.java](../Benchmark/OptimizerBenchmarkFramework.java)

## Integration Points

- Optimizer construction:
  - [src/Graph/optimizer/GraphOptimizer.java](../Graph/optimizer/GraphOptimizer.java)
  - [src/Graph/optimizer/OptimizerFactory.java](../Graph/optimizer/OptimizerFactory.java)
- Runtime backend config application:
  - [src/Backend/ComputeEngine.java](../Backend/ComputeEngine.java)
