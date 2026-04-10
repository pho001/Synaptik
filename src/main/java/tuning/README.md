# Tuning Package

## Contents

- [Purpose](#purpose)
- [Top-Level Workflows](#top-level-workflows)
- [Core Architecture](#core-architecture)
- [Execution Source Of Truth](#execution-source-of-truth)
- [Family Split](#family-split)
- [Package Layout](#package-layout)
- [Public Surfaces](#public-surfaces)
- [Benchmark](#benchmark)
- [Per-Graph Autotune](#per-graph-autotune)
- [Platform Calibration](#platform-calibration)
- [Calibration Progress Reporting](#calibration-progress-reporting)
- [Scoring](#scoring)
- [Persistence](#persistence)
- [Knob Reference](#knob-reference)
- [How To Add A New Workload](#how-to-add-a-new-workload)
- [Recommended Reading Order](#recommended-reading-order)

## Purpose

The `tuning` package owns:

- benchmark orchestration
- per-graph autotune orchestration
- platform calibration orchestration
- workload catalogs
- measurement and validation integration
- search strategies
- report rendering
- tuning persistence

It does **not** own execution semantics.

The core rule is:

- tuning adapts to the execution surface
- the execution core does not adapt to tuning

That means:

- tuning never invents a hidden runtime model for actual execution
- every final execution choice must be representable through normal runtime APIs
- the only execution artifact used by real compute remains `ExecutionProfile`

## Top-Level Workflows

The package is intentionally split into three separate workflows:

- `benchmark`
- `per-graph autotune`
- `platform calibration`

They are related, but they do different jobs.

### Benchmark

Purpose:

- compare concrete runnable execution variants
- explain performance
- track regressions

Benchmark is explicit:

- explicit workload
- explicit candidate entries
- optional explicit baseline

Benchmark does not search.

### Per-graph autotune

Purpose:

- find a good execution policy for one concrete graph/workload family

Autotune owns:

- candidate ordering
- search / refinement
- validation
- measurement
- optional persistence of best profile and history

### Platform calibration

Purpose:

- calibrate platform/runtime defaults for a machine and dtype/mode pair
- do it over representative family workload sets
- produce a reusable runtime profile for later assembly

Platform calibration is not benchmark and is not graph autotune.

It optimizes:

- runtime thresholds
- runtime scheduling policy
- runtime numerics policy

It does **not** optimize:

- graph optimizer stage order
- graph rewrite policy

## Core Architecture

The tuning package now targets a three-layer execution-policy model:

1. `PlatformRuntimeProfile`
2. `GraphExecutionPolicy`
3. `ExecutionProfileAssembler`

### `PlatformRuntimeProfile`

Represents platform-specific runtime defaults.

It owns:

- matmul runtime knobs
- fused dispatch thresholds
- non-fused element-wise dispatch thresholds
- reduction runtime knobs
- scheduler policy
- materialization policy
- numerics policy

It does **not** own:

- optimizer stages
- rewrite policy
- graph-specific optimization decisions

### `GraphExecutionPolicy`

Represents graph-level policy.

Current first version:

- wrapper over `OptimizerConfig`

This is the place for:

- stage order
- rewrite policy
- CSE / fusion / memory optimizer policy

### `ExecutionProfileAssembler`

Assembler is the only place that merges:

- built-in defaults
- `PlatformRuntimeProfile`
- `GraphExecutionPolicy`
- explicit caller override, if present

into a final:

- `ExecutionProfile`

This is the key layering rule:

- runtime calibration and graph tuning stay separate until final assembly

## Execution Source Of Truth

The execution source of truth is still:

- [ExecutionProfile.java](../config/profile/ExecutionProfile.java)

Real execution still happens through:

- [Tensor.java](../tensor/Tensor.java)
- [CompiledGraph.java](../graph/CompiledGraph.java)
- [PreparedExecution.java](../graph/execution/PreparedExecution.java)

That is deliberate.

`PlatformRuntimeProfile` and `GraphExecutionPolicy` are tuning artifacts.
They are not direct execute artifacts.

## Family Split

Every tuning knob must belong to one primary family only.

Current target split:

- `MATMUL`
  - BLAS thresholds
  - BLAS threads
  - F32/BF16 shape heuristics
  - Java matmul parallel threshold

- `FUSED`
  - fused cheap vector threshold
  - fused transcendental vector threshold
  - fused cheap parallel threshold
  - fused transcendental parallel threshold

- `ELEMENTWISE_DISPATCH`
  - non-fused cheap vector threshold
  - non-fused transcendental vector threshold
  - non-fused cheap parallel threshold
  - non-fused transcendental parallel threshold

- `REDUCTION`
  - reduction vector threshold
  - reduction parallel threshold
  - sum accuracy mode

- `SCHEDULER`
  - chunk targets
  - minimum chunk sizes
  - common-pool threshold

- `MATERIALIZATION`
  - contiguous materialization threshold

- `NUMERICS`
  - approximation mode
  - exact-transcendentals switch

- `GRAPH_POLICY`
  - optimizer stages and rewrite configuration

This split matters because cost models are not the same.

Example:

- fused dispatch thresholds must not share the same family as plain unary/binary element-wise dispatch
- graph rewrite policy must not be mixed into platform calibration

## Package Layout

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

Responsibilities:

- `candidate`
  - generic execution-profile candidate spaces used mainly by per-graph autotune

- `measure`
  - measurement policies and measurement engine

- `report`
  - DTOs and text/JSON renderers

- `search`
  - search strategies and score/bound models

- `session`
  - orchestration layer
  - public request/result models
  - platform-runtime candidate model for calibration

- `store`
  - persistence helpers and fingerprinting

- `validate`
  - validation contracts and validation engine

- `workload`
  - workload catalogs and workload builders

## Public Surfaces

The most important public surfaces are:

- benchmark
  - [BenchmarkRequest.java](./session/BenchmarkRequest.java)
  - [BenchmarkSuiteRequest.java](./session/BenchmarkSuiteRequest.java)
  - [BenchmarkSession.java](./session/BenchmarkSession.java)
  - [BenchmarkSuiteSession.java](./session/BenchmarkSuiteSession.java)

- autotune
  - [AutotuneRequest.java](./session/AutotuneRequest.java)
  - [AutotuneSession.java](./session/AutotuneSession.java)
  - [TuningResult.java](./session/TuningResult.java)

- platform calibration
  - [PlatformCalibrationRequest.java](./session/PlatformCalibrationRequest.java)
  - [PlatformCalibrationSession.java](./session/PlatformCalibrationSession.java)
  - [PlatformCalibrationResult.java](./session/PlatformCalibrationResult.java)

- execution-policy model
  - [PlatformRuntimeProfile.java](../config/profile/PlatformRuntimeProfile.java)
  - [GraphExecutionPolicy.java](../config/profile/GraphExecutionPolicy.java)
  - [ExecutionProfileAssembler.java](../config/profile/ExecutionProfileAssembler.java)

## Benchmark

Benchmark is intentionally simple.

Input:

- one workload
- explicit benchmark entries
- optional explicit baseline

Important contract:

- benchmark entries must represent comparable variants of the same logical workload family

Good comparison:

- same workload
- same dtype
- same execution mode
- same logical shape family
- different optimizer/runtime configuration

Bad comparison:

- mixing conv2d and transformer in one benchmark
- mixing dtype/mode changes in one comparison set

Typical usage:

```java
ExecutionProfile profile = new ExecutionProfile(
        "matmul-default",
        "matmul-default",
        tensor.DataType.FLOAT64,
        backend.runtime.ExecutionMode.FORWARD,
        config.optimizer.OptimizerConfig.inferenceDefaults(),
        config.runtime.RuntimeConfig.inferenceDefaults(),
        config.profile.WorkloadProfile.none()
);

BenchmarkRequest request = TuningDefaults.quickBenchmark(
        tuning.workload.StandardWorkloads.matmul("matmul_small", 1, 64, 64, 64),
        java.util.List.of(
                BenchmarkEntry.baseline("baseline", profile),
                BenchmarkEntry.candidate("candidate", profile)
        )
);
```

## Per-Graph Autotune

Per-graph autotune now carries the same architectural split as the rest of tuning.

`AutotuneRequest` explicitly carries:

- `GraphExecutionPolicy`
- `PlatformRuntimeProfile`
- workload
- candidate space
- measurement policy
- validation policy
- search policy

This means:

- the request knows which platform runtime defaults the graph tuning starts from
- final candidate execution is still measured as ordinary `ExecutionProfile`
- but the seed model is no longer conceptually one monolithic opaque profile

Important current practical detail:

- generic graph autotune still evaluates `ExecutionProfile` candidates
- but the request model now already reflects the target architecture

Typical usage:

```java
ExecutionProfile seed = new ExecutionProfile(
        "conv-base",
        "conv-base",
        tensor.DataType.FLOAT32,
        backend.runtime.ExecutionMode.FORWARD,
        config.optimizer.OptimizerConfig.inferenceDefaults(),
        config.runtime.RuntimeConfig.inferenceDefaults(),
        config.profile.WorkloadProfile.none()
);

tuning.candidate.CandidateSpace space = new tuning.candidate.ProfileGridCandidateSpace(
        seed,
        tuning.candidate.ProfileMutators.conv2dWorkloadMutators()
);

AutotuneRequest request = TuningDefaults.quickAutotune(
        tuning.workload.StandardWorkloads.conv2d(
                "conv2d_resnet_3x3",
                2, 64, 128, 56, 56, 3, 3,
                tensor.Conv2dOptions.defaults().withPadding(1, 1),
                true
        ),
        seed,
        space
);
```

## Platform Calibration

Platform calibration now follows the new model fully.

It starts from:

- a seed `ExecutionProfile`

and immediately derives:

- `GraphExecutionPolicy`
- `PlatformRuntimeProfile`

From then on:

- candidate generation mutates `PlatformRuntimeProfile`
- not generic `ExecutionProfile`

Only right before benchmark execution does calibration assemble a candidate `ExecutionProfile`.

That is the clean layer boundary we wanted.

### Calibration step model

Each calibration step owns:

- one family
- one workload set
- one runtime-profile candidate-space factory
- one score policy

Flow:

1. start from current `PlatformRuntimeProfile`
2. generate runtime-profile candidates
3. assemble executable `ExecutionProfile` candidates
4. benchmark them over the whole step workload set
5. score candidates with the family score policy
6. choose the winner
7. carry the winning runtime profile into the next step

Typical one-call usage:

```java
ExecutionProfile seed = new ExecutionProfile(
        "platform-seed",
        "platform-seed",
        tensor.DataType.FLOAT64,
        backend.runtime.ExecutionMode.FORWARD,
        config.optimizer.OptimizerConfig.inferenceDefaults(),
        config.runtime.RuntimeConfig.inferenceDefaults(),
        config.profile.WorkloadProfile.none()
);

PlatformCalibrationResult result = PlatformCalibrationRunner.runBalancedInference(
        java.nio.file.Path.of("build", "platform-calibration"),
        seed
);
```

The final persisted runtime artifact is:

- `PlatformRuntimeProfile`

not a platform-default `ExecutionProfile`.

## Calibration Progress Reporting

Platform calibration should expose a live progress-reporting layer.

Reason:

- calibration is multi-family
- each family evaluates workload sets
- each workload set evaluates many candidates

Without live progress, long calibration runs are hard to observe and debug.

The intended progress hierarchy is:

- family
- scenario / workload
- candidate
- phase

Minimum information that should be emitted:

- current family
- current family step index / total
- current workload/scenario name
- current workload index / total in the family step
- current candidate id
- current candidate index / total in the family step
- current phase:
  - generating
  - validating
  - measuring
  - scoring
  - completed
  - failed
- current family leader, if already known

Expected textual shape:

```text
family=MATMUL step=1/5
scenario=matmul_square_128 workload=2/4
candidate=blasThreads=4 candidate=7/24
phase=measuring
currentLeader=blasThreads=2 score=1.238
```

Architectural rule:

- session orchestration emits structured progress events
- logger / renderer consumes them
- measurement and score logic stay pure and do not own console reporting

## Scoring

Scoring is now a strategy layer.

It is not hardcoded inside session orchestration.

Core types:

- [PlatformCalibrationScorePolicy.java](./session/PlatformCalibrationScorePolicy.java)
- [PlatformCalibrationScore.java](./session/PlatformCalibrationScore.java)
- [PlatformCalibrationCandidateSummary.java](./session/PlatformCalibrationCandidateSummary.java)

Current properties:

- score policy is swappable
- score policy is reportable
- score policy can carry structured breakdown
- calibration result stores candidate-level score summaries

The current default implementation is still simple:

- `averageMedianMs`

but the architecture already supports:

- weighted geometric mean
- worst-bucket penalties
- validation-gated numerics scoring
- variance-aware scheduler scoring

## Persistence

Persistence is now split by artifact role.

### Platform layer

Persisted artifact:

- `PlatformRuntimeProfile`

Store:

- [PlatformRuntimeProfileIO.java](../config/profile/PlatformRuntimeProfileIO.java)
- [JsonFilePlatformRuntimeProfileStore.java](./store/JsonFilePlatformRuntimeProfileStore.java)

### Graph autotune layer

Persisted artifacts:

- best `ExecutionProfile`
- tuning history

Stores:

- [BestProfileStore.java](./store/BestProfileStore.java)
- [TuningHistoryStore.java](./store/TuningHistoryStore.java)

### Explain artifacts

Persisted separately:

- benchmark JSON/text reports
- autotune JSON/text reports
- platform calibration JSON/text reports

Important rule:

- explain artifacts are not execution source of truth

## Knob Reference

Detailed knob reference lives in:

- [KNOBS.md](./KNOBS.md)

That document explains:

- family ownership
- meaning of each knob
- which workflow uses it
- recommended candidate values

## How To Add A New Workload

The extension path is:

1. create a `WorkloadSpec`
2. build a fresh tensor graph from `WorkloadEnvironment`
3. attach stable `WorkloadMetadata`
4. choose validation target
5. choose validation reference
6. register it in `StandardWorkloads` or a local `WorkloadCatalog`

Minimal example:

```java
tuning.workload.WorkloadSpec spec = new tuning.workload.TensorRootWorkloadSpec(
        "custom_bias_gelu",
        tuning.workload.WorkloadKind.GENERIC,
        environment -> {
            tensor.Tensor x = tensor.Tensor.randn(new int[]{32, 128}, environment.profile().dataType(), "x");
            tensor.Tensor w = tensor.Tensor.randn(new int[]{128, 128}, environment.profile().dataType(), "w");
            tensor.Tensor b = tensor.Tensor.randn(new int[]{128}, environment.profile().dataType(), "b");
            tensor.Tensor y = x.matmul(w).add(b).gelu();
            y.setLabel("custom_bias_gelu_output");
            return y.sum();
        },
        environment -> tuning.validate.ValidationReference.none(),
        environment -> tuning.workload.WorkloadMetadata.of("custom_bias_gelu", tuning.workload.WorkloadKind.GENERIC)
);
```

Design rule:

- workloads are graph builders
- not precompiled artifacts
- not benchmark scripts

## Recommended Reading Order

1. [Architecture](./ARCHITECTURE.md)
2. [Knobs](./KNOBS.md)
3. [Persistence](./PERSISTENCE.md)
4. [Workloads](./WORKLOADS.md)
5. [Search](./SEARCH.md)
6. [Reporting](./REPORTING.md)
