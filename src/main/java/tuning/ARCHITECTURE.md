# Tuning Architecture

## Contents

- [Core Rule](#core-rule)
- [Responsibilities](#responsibilities)
- [Execution Surface](#execution-surface)
- [Module Breakdown](#module-breakdown)
- [Benchmark Flow](#benchmark-flow)
- [Autotune Flow](#autotune-flow)
- [Baseline Model](#baseline-model)
- [Tracing Surface](#tracing-surface)
- [Architectural Review](#architectural-review)
- [Examples](#examples)

## Core Rule

The package is built around one architectural rule:

- the execution core is the source of truth
- tuning is a client of that core

That means:

- candidates are described through `ExecutionProfile`
- benchmark and autotune never invent a second execution model
- every chosen candidate must be runnable through normal execution APIs

This avoids the usual benchmark-framework failure mode:

- benchmark-specific knobs drift away from the runtime that they claim to tune

## Responsibilities

The package is split so that each part owns one decision type only.

### `workload`

Owns:

- what graph is being measured
- what metadata describes that workload
- what validation reference belongs to it

### `candidate`

Owns:

- how execution-profile candidates are represented
- how candidate spaces are generated
- how profile mutators expand profile families

### `measure`

Owns:

- timing policies
- measurement engine
- compile / prepare / traced run / steady-state timing collection

### `validate`

Owns:

- validation references
- numeric and bool comparison logic
- baseline-profile validation

### `search`

Owns:

- candidate selection strategy
- refinement
- tree search
- best-first and branch-and-bound variants

### `report`

Owns:

- benchmark and tuning report DTO
- text renderers
- JSON renderers

### `store`

Owns:

- best-profile persistence
- tuning-history persistence
- hardware and workload fingerprints

### `session`

Owns:

- orchestration of the above components
- user-facing request/result presets

## Execution Surface

Everything here is intentionally built on top of:

- [Tensor.java](../tensor/Tensor.java)
- [CompiledGraph.java](../graph/CompiledGraph.java)
- [PreparedExecution.java](../graph/execution/PreparedExecution.java)
- [ExecutionProfile.java](../config/profile/ExecutionProfile.java)

That has two consequences:

### Good consequence

- tuning results are directly reusable in real execution

### Deliberate constraint

- if execution policy is not representable through `ExecutionProfile`, tuning is not allowed to invent it silently

It must first be added to the execution surface properly.

## Module Breakdown

### Candidate model

The minimal unit is:

```java
record Candidate(
    String name,
    ExecutionProfile profile
) {}
```

This is a strong design choice:

- no parallel hidden execution model
- no “benchmark-only config object” as final source of truth

### Request model

Benchmark:

```java
BenchmarkRequest
BenchmarkSuiteRequest
```

Autotune:

```java
AutotuneRequest
```

### Result model

Benchmark:

```java
BenchmarkReport
BenchmarkSuiteReport
```

Autotune:

```java
TuningResult
TuningSummary
```

## Benchmark Flow

The benchmark flow is:

1. resolve workload
2. enrich candidate list with benchmark baselines
3. validate candidates if validation is enabled
4. measure valid candidates
5. build report

Pseudo-code:

```java
List<Candidate> candidates = withBaselines(request.candidates());
for (Candidate candidate : candidates) {
    WorkloadInstance workload = request.workload().instantiate(...);
    ValidationResult validation = validate(candidate, workload, policy);
    if (validation.valid()) {
        MeasurementResult measurement = measure(candidate, workload, policy);
        reports.add(success(candidate, validation, measurement));
    } else {
        reports.add(failure(candidate, validation, ...));
    }
}
```

## Autotune Flow

The autotune flow is:

1. search chooses initial candidates
2. candidates are validated
3. valid candidates are measured
4. if strategy supports refinement, new candidates are generated from measured frontier
5. finalists are sorted by score
6. best profile and history may be persisted

Pseudo-code:

```java
SearchResult seed = strategy.search(context);
evaluate(seed.selectedCandidates());

for (round = 1; round < maxRounds; round++) {
    SearchResult refined = strategy.refine(...);
    if (refined.selectedCandidates().isEmpty()) break;
    evaluate(refined.selectedCandidates());
}

pick finalists;
persist if configured;
return TuningResult;
```

## Baseline Model

Every benchmark can now include two explicit baseline candidates:

- `BASELINE_NO_OPT`
- `BASELINE_NO_OPT_CONSERVATIVE_RUNTIME`

These answer two different questions.

### `BASELINE_NO_OPT`

Compile-time baseline:

- `OptimizerConfig.noOptimization()`
- same runtime policy as measured candidate

This isolates:

- what optimizer stages, rewrite, fuse and memory policy changed

### `BASELINE_NO_OPT_CONSERVATIVE_RUNTIME`

Compile-time + runtime conservative baseline:

- `OptimizerConfig.noOptimization()`
- BLAS disabled
- approximation off
- exact transcendentals forced

This isolates:

- graph optimization plus conservative runtime versus tuned runtime

## Tracing Surface

Tracing is intentionally implemented in the core execution layer, not in `tuning`.

Reason:

- benchmark/autotune need observations
- they must not own execution instrumentation semantics

Current trace pipeline:

1. execution core emits `CompileTrace`
2. execution core emits `PrepareTrace`
3. execution core emits `RunTrace`
4. `tuning.measure` converts those into measurement results
5. `tuning.report` renders them for humans and JSON consumers

This keeps the layering clean:

- trace production belongs to execution
- trace interpretation belongs to tuning/reporting

## Architectural Review

Current architectural strengths:

- `ExecutionProfile` is the single execution source of truth
- workload construction is isolated from search logic
- validation is a first-class stage, not an optional afterthought
- search strategies are pluggable instead of being hardwired into one monolithic autotune loop
- reporting is separate from persistence
- baseline execution is explicit instead of being hidden in benchmark code

Current deliberate constraints:

- benchmark candidates are expected to be comparable variants of the same workload family
- persistence is optimized for usability today, not for multi-tenant fleet-scale lifecycle
- workload families are code-defined, not dynamically loaded from an external scenario DSL

Current weak spots to keep in mind:

- benchmark reports are still primarily per-run artifacts; suite-level aggregation exists, but long-horizon report comparison is still thin
- baseline derivation currently assumes the benchmark is comparing one coherent candidate family
- workload families are still code-defined Java builders rather than a higher-level declarative scenario format

None of those are blockers for the current design.
They are follow-up hardening areas.

## Examples

### Example: benchmark session layering

```java
BenchmarkRequest request = TuningDefaults.quickBenchmark(
        StandardWorkloads.matmul("matmul_small", 1, 64, 64, 64),
        java.util.List.of(new Candidate("candidate", profile))
);

BenchmarkReport report = BenchmarkSession.create(request).run();
```

Pipeline:

1. request owns policies
2. session orchestrates workload instantiation, validation and measurement
3. measurement consumes core execution traces
4. report is a pure DTO/result object

Output:

- no mutation of execution-core architecture
- no benchmark-owned hidden execution representation

Relevant classes:

- [CompileTrace.java](../graph/execution/trace/CompileTrace.java)
- [PrepareTrace.java](../graph/execution/trace/PrepareTrace.java)
- [RunTrace.java](../graph/execution/trace/RunTrace.java)
- [ExecutionStepTrace.java](../graph/execution/trace/ExecutionStepTrace.java)

That keeps the layering correct:

- execution core emits facts
- tuning consumes and reports them

The most important current trace metadata families are:

- `LayoutTraceMetadata`
- `DispatchTraceMetadata`
- `ReductionTraceMetadata`
- `MatMulTraceMetadata`
- `FusedTraceMetadata`

## Examples

### Example 1: Quick benchmark

```java
ExecutionProfile profile = new ExecutionProfile(
        "quick-matmul",
        "quick-matmul",
        DataType.FLOAT64,
        ExecutionMode.FORWARD,
        OptimizerConfig.inferenceDefaults(),
        RuntimeConfig.inferenceDefaults(),
        WorkloadProfile.none()
);

BenchmarkRequest request = TuningDefaults.quickBenchmark(
        StandardWorkloads.matmul("matmul_small", 1, 64, 64, 64),
        java.util.List.of(new Candidate("matmul", profile))
);

BenchmarkReport report = BenchmarkSession.create(request).run();
```

Input:

- workload: `matmul_small`
- one explicit candidate

Output:

- 3 benchmark candidates in practice:
  - user candidate
  - `BASELINE_NO_OPT`
  - `BASELINE_NO_OPT_CONSERVATIVE_RUNTIME`

### Example 2: Quick autotune

```java
ExecutionProfile base = new ExecutionProfile(
        "conv-base",
        "conv-base",
        DataType.FLOAT32,
        ExecutionMode.FORWARD,
        OptimizerConfig.inferenceDefaults(),
        RuntimeConfig.inferenceDefaults(),
        WorkloadProfile.none()
);

AutotuneRequest request = TuningDefaults.quickAutotune(
        StandardWorkloads.conv2d(
                "conv2d_resnet_3x3",
                2, 64, 128, 56, 56, 3, 3,
                new Conv2dOptions(1, 1, 1, 1, 1, 1, 1),
                true
        ),
        new ProfileGridCandidateSpace(
                base,
                ProfileMutators.conv2dWorkloadMutators()
        )
);

TuningResult result = AutotuneSession.create(request).run();
```

Input:

- one base profile
- workload-aware mutator family

Output:

- search-selected candidate subset
- validation results
- finalist ranking
- best `ExecutionProfile`
