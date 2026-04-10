# Tuning Architecture

## Contents

- [Core Rule](#core-rule)
- [Execution Layers](#execution-layers)
- [Workflow Split](#workflow-split)
- [Module Responsibilities](#module-responsibilities)
- [Family Ownership](#family-ownership)
- [Assembly Rules](#assembly-rules)
- [Benchmark Flow](#benchmark-flow)
- [Per-Graph Autotune Flow](#per-graph-autotune-flow)
- [Platform Calibration Flow](#platform-calibration-flow)
- [Calibration Progress Reporting](#calibration-progress-reporting)
- [Score Policy Model](#score-policy-model)
- [Tracing Boundary](#tracing-boundary)
- [Current State](#current-state)

## Core Rule

The package is built around one rule:

- execution owns execution semantics
- tuning consumes execution semantics

Tuning is not allowed to invent a hidden execute-only model.

Final execution still happens through:

- `ExecutionProfile`
- `Tensor`
- `CompiledGraph`
- `PreparedExecution`

This prevents the common failure mode where a benchmark/autotune framework drifts away from the runtime it claims to tune.

## Execution Layers

The target architecture has three layers.

### 1. `PlatformRuntimeProfile`

Represents machine-specific runtime defaults.

This layer owns:

- runtime thresholds
- runtime scheduling knobs
- runtime numerics policy

It does not own:

- graph optimizer stage order
- rewrite policy

### 2. `GraphExecutionPolicy`

Represents graph-level policy.

Current first version:

- `OptimizerConfig`

Future direction:

- explicit per-subsystem graph policy records

### 3. `ExecutionProfileAssembler`

Assembler merges:

- built-in defaults
- `PlatformRuntimeProfile`
- `GraphExecutionPolicy`
- explicit caller override

into final:

- `ExecutionProfile`

This is the only allowed merge point.

## Workflow Split

Three workflows exist and must stay separate:

### Benchmark

Role:

- compare concrete runnable variants
- explain speed
- track regressions

Benchmark does not tune platform defaults and does not search graph policy.

### Per-graph autotune

Role:

- search graph-level execution variants for one workload family

Autotune can persist:

- best profile
- history

But it still evaluates runnable `ExecutionProfile` candidates.

### Platform calibration

Role:

- calibrate reusable runtime defaults for one platform and dtype/mode pair

Platform calibration does not tune:

- graph optimizer stage order
- graph rewrite policy

It tunes runtime families only.

## Module Responsibilities

### `workload`

Owns:

- graph construction contracts
- workload metadata
- validation references

### `candidate`

Owns:

- generic `ExecutionProfile` candidate spaces
- mainly for per-graph autotune

### `measure`

Owns:

- measurement policy
- measurement engine
- compile / prepare / traced run / steady-state collection

### `validate`

Owns:

- validation policies
- tensor comparisons
- baseline/reference-driven correctness checks

### `search`

Owns:

- candidate ordering
- refinement strategies
- tree search
- score/bound models for graph autotune search

### `report`

Owns:

- benchmark/tuning/calibration report DTOs
- text renderers
- JSON renderers

### `store`

Owns:

- persistence helpers
- best-profile store
- tuning history
- hardware/workload fingerprints

### `session`

Owns orchestration.

This is where:

- requests are interpreted
- measurement and validation are combined
- reports are built
- persistence hooks are triggered

It also owns the platform-calibration runtime-profile candidate layer:

- `PlatformRuntimeCandidateSpace`
- `PlatformRuntimeProfileMutator`

## Family Ownership

Every runtime knob must belong to exactly one family.

### `MATMUL`

Owns:

- BLAS min-work threshold
- BLAS threads
- F32/BF16 shape heuristics
- Java matmul parallel threshold

### `FUSED`

Owns:

- fused cheap vector threshold
- fused transcendental vector threshold
- fused cheap parallel threshold
- fused transcendental parallel threshold

Important:

- these are dispatch thresholds
- not backend-selection knobs

### `ELEMENTWISE_DISPATCH`

Owns:

- non-fused cheap vector threshold
- non-fused transcendental vector threshold
- non-fused cheap parallel threshold
- non-fused transcendental parallel threshold

This family is intentionally separate from `FUSED`.

### `REDUCTION`

Owns:

- reduction vector threshold
- reduction parallel threshold
- sum accuracy mode

### `SCHEDULER`

Owns:

- chunk-target knobs
- minimum chunk-size knobs
- common-pool threshold

### `MATERIALIZATION`

Owns:

- contiguous materialization threshold

### `NUMERICS`

Owns:

- approximation mode
- exact transcendentals flag

Important:

- these are policy knobs
- not pure hardware threshold knobs

### `GRAPH_POLICY`

Owns:

- optimizer stage order
- rewrite policy
- graph optimizer sub-configs

## Assembly Rules

Assembler must obey strict precedence:

1. built-in defaults
2. `PlatformRuntimeProfile`
3. `GraphExecutionPolicy`
4. explicit caller override

It must also obey ownership rules:

- platform runtime profile never writes graph policy
- graph policy never writes runtime family knobs
- if two families want to write the same knob, the design is wrong

## Benchmark Flow

Benchmark flow:

1. instantiate workload for each entry profile
2. validate candidate if validation is enabled
3. measure valid candidates
4. build report

Important properties:

- benchmark has explicit entries
- benchmark may have one explicit baseline
- benchmark does not enrich candidates implicitly

## Per-Graph Autotune Flow

Per-graph autotune flow:

1. build `AutotuneRequest`
2. attach:
   - workload
   - seed `PlatformRuntimeProfile`
   - seed `GraphExecutionPolicy`
   - generic `ExecutionProfile` candidate space
3. search strategy chooses a batch
4. session validates and measures candidates
5. if search supports refinement, repeat
6. finalists are ranked
7. best `ExecutionProfile` may be persisted

Current important detail:

- graph autotune still evaluates `ExecutionProfile` candidates
- but the request model already carries explicit platform/runtime and graph-policy layers

That means the public surface already matches the target architecture, even though some graph candidate generation is still implemented through generic profile mutators.

## Platform Calibration Flow

Platform calibration flow:

1. start from seed `ExecutionProfile`
2. derive:
   - `PlatformRuntimeProfile`
   - `GraphExecutionPolicy`
3. for each calibration step:
   - take current runtime profile
   - generate runtime-profile candidates
   - assemble executable `ExecutionProfile` candidates
   - benchmark them across the step workload set
   - score them through the step score policy
   - choose winner
   - carry winning runtime profile forward
4. persist final `PlatformRuntimeProfile`
5. persist JSON/text explain reports

This is the key improvement over the old model:

- platform calibration no longer mutates generic execution profiles directly
- runtime family tuning is separated from graph policy tuning

## Calibration Progress Reporting

Platform calibration should have its own explicit progress-event layer.

It should not blindly reuse graph-autotune phases, because calibration has a different hierarchy:

- family step
- workload / scenario inside the family
- runtime-profile candidate inside the family candidate set

The correct event model is:

- run started
- family started
- workload started
- candidate validating
- candidate measuring
- candidate scored
- family leader updated
- family completed
- run completed
- run failed

Required event payload should include at least:

- platform id
- family name
- family step index / total family steps
- workload name
- workload index / total workloads in family
- candidate id
- candidate index / total candidates in family
- current phase
- current best candidate id, if known
- current best score, if known
- message / failure reason

Architectural rule:

- event emission belongs to session orchestration
- score policy and measurement engine stay pure
- reporting/logging consumes the emitted structured events

## Score Policy Model

Calibration score is a strategy layer.

It is not hardcoded inside session orchestration.

Core types:

- `PlatformCalibrationScorePolicy`
- `PlatformCalibrationScore`
- `PlatformCalibrationCandidateSummary`

Required properties:

- swappable
- reportable
- versionable
- family-specific

Current default score implementation is still simple:

- `averageMedianMs`

But the architecture already supports richer family-specific policies such as:

- weighted geometric mean
- worst-bucket penalty
- validation-gated numerics scoring
- variance-aware scheduler scoring

## Tracing Boundary

Tracing belongs to execution, not to tuning.

Execution produces facts:

- compile trace
- prepare trace
- run trace

Tuning consumes those facts:

- measurement interprets them
- reporting renders them

This keeps the layering clean:

- execution produces
- tuning observes

## Current State

### What is already in place

- `PlatformRuntimeProfile`
- `GraphExecutionPolicy`
- `ExecutionProfileAssembler`
- platform calibration returns and persists runtime profiles
- platform calibration candidate generation uses runtime-profile mutators
- calibration scoring is a strategy layer with candidate-level breakdown
- autotune request surface carries platform/runtime + graph-policy seed model

### What is still transitional

- generic per-graph autotune candidate generation is still execution-profile based
- tuning documentation outside these core docs may still contain stale examples
- family-specific calibration score policies beyond the simple default are not all implemented yet

These are cleanup and hardening tasks, not architectural blockers.
