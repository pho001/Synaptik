# Benchmark (src/Benchmark)

## Purpose

The Benchmark module measures optimizer/runtime performance, validates correctness against baseline, and autotunes stage order + tuning knobs.

Primary goals:

- compare optimizer stage combinations on the same workload
- report forward/training latency and speedups
- enforce numeric equivalence checks
- persist winning profiles for runtime reuse

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
2. Apply persisted profile overrides to recommended/inference candidates.
3. Run scalar sanity check.
4. Run main benchmark:
  - forward-only measurement
  - training measurement (forward + backward)
  - correctness diffs versus baseline
5. Run per-stage breakdown table.
6. Optionally run two-phase autotune.

## Candidate Semantics

- `NO_OPT`: empty stage list, executed via explicit empty `GraphOptimizer` baseline.
- `RECOMMENDED`: training-oriented candidate, profile-overridden by persisted training winner.
- `INFERENCE_PERF`: inference-oriented candidate, profile-overridden by persisted inference winner.

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
- keep previous winner if new score is not better

## Tuning Knobs

`TuningKnobs` bundles:

- optimizer policy knobs:
  - `strictCseSafety`
  - `FuseConfig` fields (`maxClusterNodes`, score weights, shared-expensive policy)
- backend kernel knobs:
  - `KernelTuningConfig` (`cpu`, `cuda`, `opencl`)

CPU dispatch knobs include:

- `cpuVectorMinSize`
- `cpuParallelMinSize`
- `cpuParallelism`
- `cpuChunksPerWorker`
- `cpuMinChunkSize`
- `cpuContiguousMaterializeThreshold`
- `cpuSumAccuracyMode`

## Full Knob Space (Current Implementation)

This section reflects the current candidate generation in:

- [src/Benchmark/OptimizerCandidateFactory.java](../Benchmark/OptimizerCandidateFactory.java)

### Stage Order

Autotune generates all subsets and permutations of:

- `stages`: `[AR, CSE, FUSE, MEM]`

Total stage-order variants per knob profile:

- `sum(P(4, k), k=0..4) = 65`

### Optimizer Knobs

- `strictCseSafety`: `[true, false]`
- `fuse.maxClusterNodes`: `[32, 64, 80, 96]`
- `fuse.scoreThreshold`: `[0.00, 0.55, 0.60, 0.85]`
- `fuse.internalEdgeBonus`: `[0.25, 0.30, 0.50]`
- `fuse.externalInputPenalty`: `[0.10, 0.20]`
- `fuse.sharedExpensivePenalty`: `[0.50, 1.00]`
- `fuse.nonCheapBonus`: `[0.30, 0.35, 0.40]`
- `fuse.preserveSharedExpensiveNodes`: `[true, false]`

### CPU Kernel Knobs

- `kernel.cpu.loopUnrollFactor`: `[1, 4]`
- `kernel.cpu.matMulTileM`: `[16, 32, 64]`
- `kernel.cpu.matMulTileN`: `[0, 16, 32]`
- `kernel.cpu.matMulTileK`: `[0, 16, 32]`
- `kernel.cpu.vectorMinSize`: `[256, 512, 1024, 2048, 1000000000]`
- `kernel.cpu.parallelMinSize`: `[50000, 100000, 250000, 1000000, 2000000, 1000000000]`
- `kernel.cpu.parallelism`: `[0]` (`0` means auto-detect available processors)
- `kernel.cpu.chunksPerWorker`: `[2, 4, 8]`
- `kernel.cpu.minChunkSize`: `[2048, 4096, 8192]`
- `kernel.cpu.contiguousMaterializeThreshold`: `[0, 4096, 16384, 65536, 262144, 1000000000]`
- `kernel.cpu.lowCostNsPerElementThreshold`: `[0.5, 1.0, 2.0, 4.0]`
- `kernel.cpu.vectorPolicyCheap`: `[AUTO, FORCE_ON]`
- `kernel.cpu.vectorPolicyTranscendental`: `[AUTO, FORCE_OFF]`
- `kernel.cpu.vectorPolicyReduction`: `[AUTO]`
- `kernel.cpu.sumAccuracyMode` runtime values: `[FAST, KAHAN, NEUMAIER]`
- `kernel.cpu.sumAccuracyMode` autotune grid (current): `[FAST]`

### CUDA Kernel Knobs

- `kernel.cuda.loopUnrollFactor`: `[4, 8]`
- `kernel.cuda.matMulTileM`: `[16, 32]`
- `kernel.cuda.matMulTileN`: `[16, 32]`
- `kernel.cuda.matMulTileK`: `[16, 32]`

### OpenCL Kernel Knobs

- `kernel.opencl.loopUnrollFactor`: `[1, 2, 4]`
- `kernel.opencl.matMulTileM`: `[0, 16, 32]`
- `kernel.opencl.matMulTileN`: `[0, 16, 32]`
- `kernel.opencl.matMulTileK`: `[0, 16]`

### Candidate Count Notes

Current autotune candidate construction:

- `874` knob profiles
- `65` stage-order variants per profile
- total generated candidates: `874 * 65 = 56810`

Benchmark cap is controlled by `AUTOTUNE_MAX_CANDIDATES` in:

- [src/Benchmark/OptimizerBenchmarkFramework.java](../Benchmark/OptimizerBenchmarkFramework.java)

## Integration Points

- Optimizer construction:
  - [src/Graph/optimizer/GraphOptimizer.java](../Graph/optimizer/GraphOptimizer.java)
  - [src/Graph/optimizer/OptimizerFactory.java](../Graph/optimizer/OptimizerFactory.java)
- Runtime backend config application:
  - [src/Backend/ComputeEngine.java](../Backend/ComputeEngine.java)
