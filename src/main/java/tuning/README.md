# Tuning Package

## Purpose

The `tuning` package owns everything related to measuring, comparing, searching, calibrating, and persisting execution policies.

Its job is to answer questions such as:

- which concrete profile is faster on this workload?
- which runtime knobs should be calibrated for this machine?
- how should we compare candidate profiles fairly?
- how should the chosen result be stored and reused later?

It does **not** define execution semantics.

Execution still happens through the normal runtime stack:

- [tensor/Tensor.java](../tensor/Tensor.java)
- [graph/CompiledGraph.java](../graph/CompiledGraph.java)
- [graph/execution/PreparedExecution.java](../graph/execution/PreparedExecution.java)
- [config/profile/ExecutionProfile.java](../config/profile/ExecutionProfile.java)

This is the core rule:

- tuning adapts execution policy
- tuning never becomes a shadow runtime

## Reading Guide

There are four distinct audiences for this package:

1. benchmark users
   - want direct profile-vs-profile comparisons
2. autotune users
   - want the system to search graph-specific winners
3. calibration users
   - want reusable platform runtime defaults
4. tuning implementers
   - work on candidate spaces, search, measurement, validation, and persistence

Those audiences overlap, but the package should still read as one coherent pipeline:

- workloads define what is measured
- candidate spaces define what can change
- measurement defines how timing is taken
- validation defines what must remain correct
- search defines how the candidate space is explored
- sessions orchestrate the whole workflow
- stores/reporters persist and explain the result

## Mental Model

There are three distinct workflows in this package:

1. Benchmark
2. Per-graph autotune
3. Platform calibration

They use many of the same measurement and validation utilities, but they serve different purposes.

### Benchmark

Benchmark compares an explicit set of already-defined candidates.

Typical use:

- compare current profile vs baseline profile
- inspect regressions
- compare multiple manual variants

Benchmark does **not** search the space automatically.

### Per-graph autotune

Autotune searches for a better profile for one concrete workload/graph family.

Typical use:

- tune one workload for one data type and execution mode
- start from a seed profile
- explore a bounded candidate space
- persist the winner

Autotune is workload-specific.

### Platform calibration

Calibration searches for reusable machine-specific runtime defaults over representative workload families.

Typical use:

- calibrate runtime thresholds on one CPU/JDK/hardware profile
- tune runtime defaults once and reuse them later
- keep runtime knobs separate from graph rewrite policy

Calibration is platform-oriented, not one-graph-oriented.

## Execution Source Of Truth

The final executable artifact is still:

- [config/profile/ExecutionProfile.java](../config/profile/ExecutionProfile.java)

Everything inside `tuning` eventually feeds into that.

This is important because it prevents a split-brain design:

- benchmark candidates are real execution profiles
- autotune winners are real execution profiles
- calibration results are assembled into real execution profiles

No hidden “tuning-only runtime mode” should exist.

## Policy Split

The package is organized around a conceptual separation between:

- platform runtime defaults
- graph-specific policy
- final assembled execution profile

This means:

- calibration should target runtime knobs
- graph autotune should target graph/workload-specific choices
- the final merge should stay explicit

Even when a workflow starts from a full `ExecutionProfile`, it should still respect that separation conceptually.

This distinction matters because the same machine may need:

- one reusable platform runtime profile
- many graph-specific best-profile records
- zero ambiguity about which layer each result belongs to

In practice:

- calibration should produce reusable runtime defaults
- autotune should produce workload-specific runnable winners
- benchmark should compare already-defined runnable profiles without inventing new semantics

## Package Layout

The main subpackages are:

```text
tuning/
  candidate/
  etalon/
  measure/
  report/
  search/
  session/
  store/
  validate/
  workload/
```

### `candidate`

Candidate-space definitions and profile mutation utilities.

This layer answers:

- which variants exist?
- how do we enumerate or refine them?
- how do we fingerprint them?

Representative files:

- [tuning/candidate/CandidateSpace.java](./candidate/CandidateSpace.java)
- [tuning/candidate/ProfileGridCandidateSpace.java](./candidate/ProfileGridCandidateSpace.java)
- [tuning/candidate/ProfileMutators.java](./candidate/ProfileMutators.java)

### `measure`

Measurement engine and policies.

This layer answers:

- how many warmup runs?
- how many measured runs?
- how are medians/statistics produced?

Representative files:

- [tuning/measure/MeasurementPolicy.java](./measure/MeasurementPolicy.java)
- [tuning/measure/DefaultMeasurementEngine.java](./measure/DefaultMeasurementEngine.java)

### `validate`

Correctness validation for candidate execution.

This layer answers:

- what should be compared?
- against what reference?
- with what tolerance profile?

Representative files:

- [tuning/validate/ValidationEngine.java](./validate/ValidationEngine.java)
- [tuning/validate/ValidationReference.java](./validate/ValidationReference.java)
- [tuning/validate/ValidationTarget.java](./validate/ValidationTarget.java)

### `search`

Search strategies and bound/score models used by autotune and calibration.

Representative files:

- [tuning/search/SearchStrategy.java](./search/SearchStrategy.java)
- [tuning/search/ExhaustiveSearchStrategy.java](./search/ExhaustiveSearchStrategy.java)
- [tuning/search/BestFirstTreeSearchStrategy.java](./search/BestFirstTreeSearchStrategy.java)

### `session`

Top-level orchestration entry points for benchmark, autotune, and calibration.

Representative files:

- [tuning/session/BenchmarkSession.java](./session/BenchmarkSession.java)
- [tuning/session/AutotuneSession.java](./session/AutotuneSession.java)
- [tuning/session/PlatformCalibrationSession.java](./session/PlatformCalibrationSession.java)

### `store`

Persistence of reports, profiles, winners, and calibration outputs.

This layer should be the only place that knows where artifacts live on disk.
Search and measurement should work the same way whether persistence is enabled or disabled.

### `report`

Text and JSON renderers for benchmark/tuning/calibration results.

### `workload`

Reusable workload specifications used by all tuning workflows.

This is the most important abstraction boundary:

- workloads build fresh tensor graphs from the public tensor surface
- they do not know whether they are used by benchmark, autotune, or calibration

See also:

- [tuning/WORKLOADS.md](./WORKLOADS.md)

## Benchmark, Autotune, And Calibration In More Detail

The three workflows share infrastructure, but they solve different questions.

If those questions get mixed, results become misleading:

- benchmark can accidentally turn into hidden search
- autotune can accidentally overfit what should be platform calibration
- calibration can accidentally start encoding one specific graph

### Benchmark lifecycle

1. Choose one workload.
2. Define explicit benchmark entries.
3. Instantiate a fresh graph for each candidate.
4. Validate if configured.
5. Warm up.
6. Measure.
7. Render and optionally persist the report.

Use benchmark when you want a direct apples-to-apples comparison.

Benchmark answers:

- "which already-defined runnable profile is faster here?"

Benchmark does **not** answer:

- "what is the best candidate in the whole space?"
- "what should become the machine default?"

### Autotune lifecycle

1. Choose one workload.
2. Start from a seed profile.
3. Define a candidate space.
4. Let the search strategy explore/refine that space.
5. Validate candidates as needed.
6. Measure candidates.
7. Persist the winner and history if desired.

Use autotune when you want the system to search for a better profile for one workload family.

Autotune answers:

- "for this workload family, dtype, and execution mode, which runnable profile wins?"

Autotune does **not** answer:

- "what should all graphs on this machine use by default?"

### Calibration lifecycle

1. Choose calibration steps for a platform and dtype/mode.
2. For each family, instantiate representative probe workloads.
3. Generate runtime-profile candidates for that family.
4. Measure and score them.
5. Merge winning family settings into a reusable platform profile.
6. Persist the calibrated result.

Use calibration when the goal is to improve machine defaults, not to overfit one specific graph.

Calibration answers:

- "for this hardware/JDK/platform fingerprint, which runtime defaults should become reusable?"

Calibration does **not** answer:

- "which profile is best for this single application graph?"

## Calibration Families

Calibration is intentionally split into families because different hotspots have different cost models.

The family enum is:

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

Why the split exists:

- matmul tiles and microkernels behave differently from fused thresholds
- attention-specific thresholds do not share the same optimal values as generic reductions
- materialization heuristics need non-contiguous probe workloads
- numerics choices may trade speed against approximation policy

This is better than one monolithic “CPU tuning” bucket.

A family is the unit where platform calibration is allowed to generalize.
That means a family should correspond to a real execution behavior class, not an arbitrary benchmark label.

## Workloads As The Stable Boundary

All workflows rely on the same workload contract:

- build a fresh graph
- expose a validation target and reference
- attach metadata about what is being measured

The workload itself should stay on the public modeling surface whenever possible:

- `Tensor`
- `TensorOps`
- public option/config types such as:
  - `tensor.options.Conv2dOptions`
  - `tensor.options.Pool2dOptions`
  - `tensor.options.AttentionOptions`
  - `tensor.loss.LossReduction`

This keeps workloads representative and prevents tuning code from depending on backend internals.

That boundary is the main protection against benchmark slippage.
If workloads start depending on backend-private helpers, tuning stops measuring the real public graph-building model.

## Persistence

The tuning package can persist:

- benchmark reports
- tuning results
- best-profile records
- platform runtime profiles
- calibration results
- tuning history

Persistence is deliberately a separate concern.
The measurement/search code should not care whether results are stored to disk or discarded.

There are two important persistence categories:

1. graph-level winners
   - runnable `ExecutionProfile` results for one workload fingerprint
2. platform-level runtime defaults
   - reusable runtime profiles keyed by hardware/platform identity

The preferred persisted layout is platform-versioned storage under `profiles/platform/...`.
Legacy `build/...` fallback locations may still exist for compatibility, but they are not the preferred source of truth.

Relevant files:

- [tuning/store](./store)
- [tuning/PERSISTENCE.md](./PERSISTENCE.md)

## Reporting

All major workflows have text and JSON renderers.

That matters because the same core session can feed:

- local human inspection in the terminal
- CI artifacts
- machine-readable comparison tools

Relevant files:

- [tuning/report](./report)
- [tuning/REPORTING.md](./REPORTING.md)

## Minimal Examples

These examples are intentionally short.
They are meant to show workflow shape, not every available option.

### Benchmark one workload

```java
BenchmarkRequest request = StandardWorkloads.benchmark(
        "matmul_small",
        List.of(
                BenchmarkEntry.baseline("baseline", profileA),
                BenchmarkEntry.candidate("candidate", profileB)
        ),
        TuningPreset.BALANCED
);
```

### Create a convolution workload

```java
WorkloadSpec spec = StandardWorkloads.conv2d(
        "conv2d_resnet_3x3",
        2, 64, 128, 56, 56, 3, 3,
        tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
        true
);
```

### Start autotune from a seed profile

```java
AutotuneRequest request = StandardWorkloads.autotune(
        "transformer_hot_path",
        seedProfile,
        candidateSpace,
        persistencePolicy
);
```

## Recommended Reading Order

If you are new to this package, read in this order:

1. this file
2. [tuning/WORKLOADS.md](./WORKLOADS.md)
3. [tuning/ARCHITECTURE.md](./ARCHITECTURE.md)
4. [tuning/KNOBS.md](./KNOBS.md)
5. [tuning/SEARCH.md](./SEARCH.md)
6. [tuning/PERSISTENCE.md](./PERSISTENCE.md)

That sequence goes from conceptual model to workload construction, then to deeper tuning mechanics.
