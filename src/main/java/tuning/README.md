# Tuning Package

The `tuning` package is the new benchmark and autotune layer for Synaptik.

It is designed around one central rule:

- tuning adapts to the execution surface
- the execution core does **not** adapt to benchmark/autotune orchestration

That means:

- execution candidates are represented through `ExecutionProfile`
- benchmark and autotune work above:
  - `Tensor`
  - `CompiledGraph`
  - `PreparedExecution`
  - `ExecutionProfile`
- the package is split into small modules with explicit responsibilities

## Contents

- [Architecture](./ARCHITECTURE.md)
- [Workloads](./WORKLOADS.md)
- [Search](./SEARCH.md)
- [Persistence](./PERSISTENCE.md)
- [Reporting](./REPORTING.md)
- [Legacy Benchmark Review](./LEGACY-BENCHMARK-REVIEW.md)
- [How to Add a New Scenario](#how-to-add-a-new-scenario)
- [Preset Surface](#preset-surface)
- [Benchmark Candidate Contract](#benchmark-candidate-contract)
- [End-to-End Example](#end-to-end-example)

## Package Layout

```text
tuning/
  candidate/
  measure/
  report/
  search/
  session/
  store/
  validate/
  workload/
```

## What This Package Does

- defines benchmark and autotune requests
- defines workload models
- generates execution-profile candidates
- measures compile / prepare / traced run / steady-state timings
- validates candidates against explicit references or baseline execution profiles
- searches candidate spaces with pluggable strategies
- renders text and JSON reports
- compares benchmark/tuning runs through diff renderers
- persists best profiles and tuning history

In practical terms, this is now the only benchmark/autotune architecture in the repository.
The removed legacy `benchmark` package is no longer part of the runtime or documentation surface.

## What It Does Not Do

- it does not own execution semantics
- it does not introduce hidden backend-only knobs as source of truth
- it does not compile runtime kernels directly
- it does not replace `Tensor` or `CompiledGraph`

Execution semantics still live in:

- [Tensor.java](../tensor/Tensor.java)
- [CompiledGraph.java](../graph/CompiledGraph.java)
- [PreparedExecution.java](../graph/execution/PreparedExecution.java)
- [ExecutionProfile.java](../config/profile/ExecutionProfile.java)

Profile serialization lives in:

- [ExecutionProfileIO.java](../config/profile/ExecutionProfileIO.java)

Tensor fixture/materialization helpers used by workloads live in:

- [TensorDataFactory.java](../tensor/TensorDataFactory.java)

## How to Add a New Scenario

The intended extension path is:

1. create a new `WorkloadSpec`
2. instantiate a fresh tensor graph from `WorkloadEnvironment`
3. attach stable `WorkloadMetadata`
4. attach a `ValidationReference`
5. register the spec in `StandardWorkloads` or in a local `WorkloadCatalog`

Minimal example:

```java
WorkloadSpec spec = new TensorRootWorkloadSpec(
        "custom_bias_gelu",
        WorkloadKind.GENERIC,
        environment -> {
            Tensor x = Tensor.randn(new int[]{32, 128}, environment.profile().dataType(), "x");
            Tensor w = Tensor.randn(new int[]{128, 128}, environment.profile().dataType(), "w");
            Tensor b = Tensor.randn(new int[]{128}, environment.profile().dataType(), "b");
            return x.matmul(w).add(b).gelu().sum();
        },
        environment -> ValidationReference.baselineProfile(
                WorkloadValidationProfiles.conservativeReferenceProfile(environment.profile())
        ),
        environment -> WorkloadMetadata.of(
                "custom_bias_gelu",
                WorkloadKind.GENERIC,
                java.util.Map.of("batch", 32, "hidden", 128)
        )
);
```

Input:

- profile from the benchmark/autotune session
- fresh graph roots created for that profile

Output:

- one reproducible workload scenario with:
  - graph root
  - metadata
  - validation contract

That is the key design rule:

- scenarios are graph builders, not precompiled artifacts

## Preset Surface

The intended public preset surface is:

- [TuningPreset.java](./session/TuningPreset.java)
- [TuningDefaults.java](./session/TuningDefaults.java)
- [WorkloadPresetFamily.java](./session/WorkloadPresetFamily.java)

`TuningPreset` currently provides three standard policy families:

- `QUICK`
- `BALANCED`
- `THOROUGH`

The design goal is:

- callers choose one named preset first
- low-level policy objects stay available, but are not required for normal usage

Typical benchmark usage:

```java
BenchmarkRequest request = TuningDefaults.benchmark(
        TuningPreset.BALANCED,
        StandardWorkloads.matmul("matmul_small", 1, 64, 64, 64),
        java.util.List.of(candidate)
);
```

Typical catalog usage:

```java
BenchmarkSuiteRequest request = StandardWorkloads.benchmarkSuite(
        java.util.List.of("matmul_small", "conv2d_resnet_3x3"),
        java.util.List.of(candidate),
        TuningPreset.QUICK
);
```

Typical autotune usage:

```java
AutotuneRequest request = StandardWorkloads.autotune(
        "transformer_hot_path",
        candidateSpace,
        TuningPreset.BALANCED,
        PersistencePolicy.disabled()
);
```

This gives the package a cleaner scenario UX:

- preset first
- workload selection second
- candidate/candidate-space last

On top of that, the package now exposes workload-aware recommendations:

- `WorkloadPresetFamily.benchmarkPresetFor(workload)`
- `WorkloadPresetFamily.autotunePresetFor(workload)`
- `TuningDefaults.recommendedBenchmark(...)`
- `TuningDefaults.recommendedAutotune(...)`

This is intentionally still one-layered:

- `TuningPreset` remains the real preset type
- workload-aware logic only selects a recommended preset family
- it does not introduce a second benchmark/autotune policy model

## Benchmark Candidate Contract

One benchmark compares multiple execution variants of the same logical workload.

In practice that means the candidate list should vary:

- optimizer config
- runtime config
- workload profile knobs

but should keep the same:

- dtype
- execution mode
- logical workload shape family

Why this matters:

- baselines are derived from the benchmark candidate family
- speedups are only meaningful when candidates are comparable
- validation references assume equivalent execution intent

The package does not try to silently normalize fundamentally different candidates into one comparison universe.

Typical good benchmark candidate family:

- same `WorkloadSpec`
- same dtype
- same execution mode
- same logical tensor shapes
- different:
  - optimizer stage order
  - rewrite config
  - fuse policy
  - memory policy
  - runtime kernel/blas/approximation policy

Typical bad benchmark candidate family:

- one candidate uses a transformer workload profile
- another uses a conv2d workload profile
- or one candidate changes dtype/mode while the benchmark still tries to treat the result as one comparison set

Those should be separate benchmark runs.

## End-to-End Example

## Minimal Example

```java
ExecutionProfile profile = new ExecutionProfile(
        "matmul-default",
        "matmul-default",
        DataType.FLOAT64,
        ExecutionMode.FORWARD,
        OptimizerConfig.inferenceDefaults(),
        RuntimeConfig.inferenceDefaults(),
        WorkloadProfile.none()
);

Candidate candidate = new Candidate("baseline", profile);

BenchmarkRequest request = TuningDefaults.quickBenchmark(
        StandardWorkloads.matmul("matmul_small", 1, 64, 64, 64),
        java.util.List.of(candidate)
);

BenchmarkReport report = BenchmarkSession.create(request).run();
```

Input:

- one workload: `matmul_small`
- one explicit candidate
- quick benchmark preset

Output:

- `BenchmarkReport`
  - baseline candidates added automatically
  - compile / prepare / traced run timings
  - steady-state timings
  - validation status
  - speedups vs baseline

The same structure scales to autotune:

```java
CandidateSpace space = new ProfileGridCandidateSpace(
        profile,
        ProfileMutators.transformerHotPathMutators()
);

AutotuneRequest request = TuningDefaults.balancedAutotune(
        StandardWorkloads.transformerHotPath("transformer_hot_path"),
        space,
        PersistencePolicy.disabled()
);

TuningResult result = AutotuneSession.create(request).run();
```

Input:

- one workload family
- a candidate space derived from `ExecutionProfile`
- a search/measurement/validation preset

Output:

- `TuningResult`
  - best profile
  - finalist measurements
  - structured tuning summary
  - optional persisted best-profile/history side effects

## Recommended Reading Order

1. [Architecture](./ARCHITECTURE.md)
2. [Workloads](./WORKLOADS.md)
3. [Search](./SEARCH.md)
4. [Persistence](./PERSISTENCE.md)
5. [Reporting](./REPORTING.md)
6. [Legacy Benchmark Review](./LEGACY-BENCHMARK-REVIEW.md)
