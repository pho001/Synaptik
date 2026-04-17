# Tuning Architecture

The tuning layer is built around one hard rule:

- tuning must not create a parallel execution model next to runtime

Every benchmarked or autotuned candidate must be genuinely runnable as:

- `ExecutionProfile`
- `CompiledGraph`
- `PreparedExecution`

So tuning does not benchmark an abstract "knob set". It benchmarks and evaluates a truly executable profile.

## Reading Guide

This document describes:

- how the final executable profile is assembled
- how benchmark, autotune, and platform calibration differ
- which layers own runtime knobs and which own graph policy
- which calibration families exist today
- how orchestration combines workload, measurement, validation, search, and persistence

## Core Artifacts

You need to distinguish four artifacts:

### 1. `PlatformRuntimeProfile`

Machine-specific runtime defaults.

It contains runtime families:

- matmul
- fused
- elementwise dispatch
- reduction
- scheduler
- materialization
- numerics

It does not contain:

- optimizer stage order
- rewrite policy
- workload-specific graph winners

### 2. `GraphExecutionPolicy`

Graph-level policy.

Today this is effectively a wrapper around:

- `OptimizerConfig`

So it contains:

- stage order
- rewrite config
- CSE config
- fuse config
- memory config

### 3. `ExecutionProfile`

Runnable artifact that is actually measured.

It is created by assembling:

- graph policy
- runtime profile
- dtype
- execution mode
- optional workload metadata

### 4. Persistence / Explain Artifacts

This group includes:

- best profile records
- tuning history
- platform calibration reports
- benchmark/autotune reports

These are not direct execute contracts.

## Assembly Boundary

The only correct place where the final runnable profile is assembled is:

- [ExecutionProfileAssembler.java](../config/profile/ExecutionProfileAssembler.java)

The assembler takes:

- `PlatformRuntimeProfile`
- `GraphExecutionPolicy`
- dtype
- execution mode

and returns:

- `ExecutionProfile`

That is a critical boundary:

- platform calibration mutates `PlatformRuntimeProfile`
- graph autotune mutates or selects `ExecutionProfile`
- runtime always ultimately receives `ExecutionProfile`

## Workflow Split

The tuning package currently contains three separate workflows.

### Benchmark

Role:

- compare explicitly provided candidates
- show traces and hotspots
- compute speedup versus baseline

It does not do:

- search
- candidate refinement
- mutation of runtime defaults

Entry:

- [BenchmarkSession.java](./session/BenchmarkSession.java)

### Graph Autotune

Role:

- select the best `ExecutionProfile` for a specific workload
- use a search strategy
- persist the best profile and history

Entry:

- [AutotuneSession.java](./session/AutotuneSession.java)

### Platform Calibration

Role:

- tune platform runtime defaults family by family
- persist the resulting `PlatformRuntimeProfile`

Entry:

- [PlatformCalibrationSession.java](./session/PlatformCalibrationSession.java)

## Session Responsibilities

The `session` layer is the orchestrator. It combines:

- candidate generation
- validation
- measurement
- search
- progress reporting
- persistence hooks

It does not handle:

- kernel execution detail
- optimizer internals
- workload implementation detail

## Module Split

### `workload`

Defines:

- `WorkloadSpec`
- `WorkloadInstance`
- workload metadata
- standard and calibration workload catalogs

### `candidate`

Defines:

- `Candidate`
- `CandidateSpace`
- `RefinableCandidateSpace`
- `ExecutionProfileMutator`

### `measure`

Defines:

- `MeasurementPolicy`
- `MeasurementEngine`
- `MeasurementResult`

The current default measurement engine:

- compiles the graph
- prepares execution
- optionally performs a traced run
- runs warmup
- runs steady-state repeats

### `validate`

Defines:

- workload correctness checks
- baseline/reference validation

### `search`

Handles:

- candidate ordering
- refinement
- tree search
- history-aware preference/pruning

### `report`

Handles:

- text and JSON explain artifacts
- suite summaries
- candidate summaries
- calibration/tuning result renderers

### `store`

Handles:

- platform profile store
- best profile store
- history store
- hardware/workload fingerprinting
- path helpers

## Benchmark Flow

`BenchmarkSession` currently does this in practice:

1. instantiate a fresh workload for each `BenchmarkEntry`
2. run validation
3. if validation passes, measure the candidate
4. return `BenchmarkReport`

So:

- benchmark does not operate on one shared graph instance across candidates
- each candidate gets a fresh workload instance

That is correct, because compiled/prepared runtime may change graph structure and cache state.

## Autotune Flow

`DefaultAutotuneSession` currently does:

1. create `SearchContext`
2. let the search strategy select the initial batch
3. validate and measure the candidates
4. if the strategy supports refinement, continue iterating
5. sort successful candidates by steady-state median
6. select finalists
7. persist history and the best profile

Important:

- search selects candidates
- the session is what measures them
- "best" currently means the lowest steady-state median

## Platform Calibration Flow

`DefaultPlatformCalibrationSession` proceeds family by family:

1. take a seed `PlatformRuntimeProfile`
2. create a candidate space for the first family step
3. assemble runnable `ExecutionProfile` instances from candidates
4. run a benchmark suite for the selected calibration workloads
5. let the score policy choose the winner
6. use the winning runtime profile as the seed for the next family
7. after the last step, persist the final `PlatformRuntimeProfile`

That is the key difference from graph autotune:

- calibration does not search for a workload-specific winner
- calibration searches for reusable platform defaults

## Current Calibration Families

The current enum is:

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
- `CONV2D`
- `NUMERICS`

But not all families are currently used in standard presets.

### Standard Training/Inference Presets Today

`PlatformCalibrationDefaults.standardTrainingSteps(...)` and `standardInferenceSteps(...)` currently compose mainly:

- `MATMUL`
- fused threshold and ASM width families
- `ELEMENTWISE_DISPATCH`
- optionally `REDUCTION`
- optionally `ATTENTION_THRESHOLDS`
- optionally `ATTENTION_MATMUL`
- optionally `SCHEDULER`
- optionally `MATERIALIZATION`
- optionally `NUMERICS`

The important current reality:

- `FUSED_ARITHMETIC` exists in the enum, but standard presets do not use it today
- the `CONV2D` family is not yet part of the standard preset steps

The documentation needs to say this explicitly, otherwise it gives the impression that more is calibrated than actually is.

## Family Ownership

Every runtime knob needs a clear owner.

### `MATMUL`

This family currently includes for example:

- BLAS minimum work
- BLAS threads
- `f32RequireMgeK`
- `f32MaxNOverK`
- `cpu.matMulParallelMinSize`
- microkernel selection
- tile selection
- attention matmul tile/microkernel selection

### `FUSED_THRESHOLDS`

This family includes:

- `cpu.fusedCheapVectorMinSize`
- `cpu.fusedTranscendentalVectorMinSize`
- `cpu.fusedCheapParallelMinSize`
- `cpu.fusedTranscendentalParallelMinSize`

### Fused ASM Width Families

These families own width knobs for specific dispatch families:

- cheap contiguous
- cheap strided
- non-cheap contiguous
- non-cheap strided

These are no longer just "internal experimental variables". They are part of the platform calibration surface.

### `ELEMENTWISE_DISPATCH`

This family owns non-fused elementwise thresholds:

- cheap vector
- transcendental vector
- cheap parallel
- transcendental parallel

### `REDUCTION`

This family owns:

- reduction vector threshold
- reduction parallel threshold
- attention vector threshold
- attention parallel threshold
- `sumAccuracyMode`

### `SCHEDULER`

This family owns:

- target chunks per worker
- minimum chunk sizes
- common pool threshold

### `MATERIALIZATION`

This family owns:

- `cpu.contiguousMaterializeThreshold`

### `NUMERICS`

This family owns:

- `approxMode`
- `forceExactTranscendentals`

### `GRAPH_POLICY`

This group owns:

- optimizer stage order
- rewrite configs
- conv2d lowering mode

But this is not `PlatformRuntimeProfile`. This is `GraphExecutionPolicy`.

## Search And Calibration Are Different

This is one of the most common sources of confusion:

- calibration candidate space generates `PlatformRuntimeProfile` mutations
- autotune candidate space generates `ExecutionProfile` variants

The first is reusable across workloads.
The second is workload-specific search.

## Score Policy

Platform calibration uses an explicit score policy per step.

Common choices today:

- `averageMedianMs()`
- `weightedGeometricMeanWithWorstBucketPenalty(alpha)`

This matters especially for attention families, where one workload bucket should not dominate completely, while a weak worst-case bucket should still be penalized.

## Tracing Boundary

Trace data is generated by the execution layer.

Tuning only consumes it through:

- compile trace
- prepare trace
- run trace
- step trace metadata

That means:

- a tuning report can say that a candidate ran with `vectorWidth=4`
- but tuning does not compute that itself, it only reads the trace from runtime

## Persistence Boundary

Persistence is also split by workflow:

- platform calibration persists `PlatformRuntimeProfile`
- autotune persists the best `ExecutionProfile`
- history persists candidate-level evidence
- reports persist explain artifacts

More in:

- [PERSISTENCE.md](./PERSISTENCE.md)

## Example: Assemble Executable From Platform Defaults

```java
ExecutionProfile profile = ExecutionProfileAssembler.assemble(
        "abc-f64",
        "abc-f64",
        DataType.FLOAT64,
        ExecutionMode.FORWARD_BACKWARD,
        platformRuntimeProfile,
        GraphExecutionPolicy.trainingDefaults()
);
```

This is the final artifact that goes into a benchmark or an application run.

## Example: Platform Calibration Vs Autotune

Platform calibration:

- takes a training/inference seed
- mutates runtime defaults
- returns `PlatformRuntimeProfile`

Autotune:

- takes a workload
- takes a seed profile
- searches candidate `ExecutionProfile` variants
- returns the best executable profile for that workload

## Common Mistakes

- mixing runtime knobs and optimizer policy into one profile without clear ownership
- treating a platform calibration winner as a graph-specific best profile
- benchmarking candidates that are not truly runnable `ExecutionProfile` instances
- storing explain artifacts as the execute source of truth

## Related Docs

- overview: [README.md](./README.md)
- workloads: [WORKLOADS.md](./WORKLOADS.md)
- knobs: [KNOBS.md](./KNOBS.md)
- persistence: [PERSISTENCE.md](./PERSISTENCE.md)
- search: [SEARCH.md](./SEARCH.md)
- reporting: [REPORTING.md](./REPORTING.md)
