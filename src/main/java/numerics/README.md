# Numerics Harness

`numerics` is a small standalone A/B harness for comparing numeric drift between two `ExecutionProfile` variants. It is not the benchmark subsystem and it is not the autotuner.

Its job is straightforward:

- take two executable profile variants
- run them on the same deterministic inputs
- compare outputs and selected gradients
- return a human-readable verdict

## Reading Guide

Use this module if you want to:

- quickly check that a new optimizer stage order did not break numerics
- compare baseline vs a more aggressive rewrite/fusion variant
- check drift between `FLOAT32` and `FLOAT64` policy variants
- verify that approximation policy does not cause unacceptable spread

Do not use it as:

- a performance benchmark
- a replacement for unit tests
- proof of complete numerical correctness of the whole framework

## Main Components

- CLI
  - [NumericsCli.java](./NumericsCli.java)
- harness orchestration
  - [NumericsHarness.java](./NumericsHarness.java)
- scenarios / graph recipes
  - [NumericsGraphFactory.java](./NumericsGraphFactory.java)
- metrics
  - [NumericsMetrics.java](./NumericsMetrics.java)
- tolerance policy
  - [NumericsPolicy.java](./NumericsPolicy.java)
- report
  - [NumericsReport.java](./NumericsReport.java)

## What It Measures

The harness currently compares five signals:

- `out`
  - forward output of the benchmark-like graph
- `gradA`
- `gradB`
- `gradC`
  - gradients of three leaf inputs in the benchmark-like graph
- `broadcast`
  - forward output of a separate broadcast-heavy graph

For each signal it computes:

- `maxAbs`
- `avgAbs`
- `maxRel`
- `maxUlp`
- `p50Ulp`
- `p95Ulp`
- `invalidCount`

It then aggregates them and applies `NumericsPolicy`.

## What It Actually Runs

`NumericsHarness` does not run a single graph. It runs two scenarios:

### 1. Optimizer-like training graph

In [NumericsGraphFactory.java](./NumericsGraphFactory.java), it is built from:

- repeated elementwise blocks over `A`, `B`, `C`
- several `linear(...)` layers
- a final scalar reduction

This graph runs in:

- `FORWARD_BACKWARD`

and the harness collects from it:

- forward output
- gradients of `A`, `B`, and `C`

### 2. Broadcast-heavy forward graph

A separate graph:

- `a.add(b).mul(c).add(a).sigmoid()`

over broadcast-compatible shapes.

It runs in:

- `FORWARD`

and is meant to expose drift in broadcast/elementwise paths that the optimizer-like graph alone might not catch.

## Why Two Graphs

This is intentional.

One graph alone typically does not capture both:

- numerics in a broader training-style graph
- numerics in broadcast-heavy shape/layout situations

So the harness combines:

- one "deeper" training-like scenario
- one "flatter" broadcast scenario

## Determinism And Input Policy

Inputs are deterministic:

- the seed is controlled by `numerics.seed`
- input arrays are generated once
- both candidate profiles receive the same data

This is essential. Without that, the results would say nothing about numeric drift between candidates and would instead reflect different inputs.

## Candidate Model

The CLI currently builds two candidates through:

- dtype
- variant name
- stage order

`NumericsHarness.profile(...)` builds `ExecutionProfile` like this:

- `OptimizerConfig.trainingDefaults().withStageOrder(...)`
- `RuntimeConfig.trainingDefaults()`

That means:

- the harness primarily compares graph-policy variants today
- runtime policy remains fixed

If you want to compare runtime policy as well, you need to build candidates programmatically outside the basic CLI.

## CLI Usage

Main class:

- `numerics.NumericsCli`

Example:

```bash
java --add-modules jdk.incubator.vector \
  -Dnumerics.dtype=FLOAT32 \
  -Dnumerics.stageA=NONE \
  -Dnumerics.stageB=AR,CSE,FUSE \
  -Dnumerics.nameA=baseline \
  -Dnumerics.nameB=optimized \
  -Dnumerics.size=200000 \
  -Dnumerics.graphBlocks=6 \
  -Dnumerics.broadcastB0=128 \
  -Dnumerics.broadcastB1=8 \
  -Dnumerics.broadcastF=128 \
  -Dnumerics.seed=42 \
  -cp build/classes/java/main \
  numerics.NumericsCli
```

## Properties

- `numerics.dtype`
  - `FLOAT32` or `FLOAT64`
  - default `FLOAT32`
- `numerics.stageA`
  - comma- or `+`-separated stage list
  - `NONE` means an empty stage order
- `numerics.stageB`
  - same as `stageA`
- `numerics.nameA`
  - label of candidate A
- `numerics.nameB`
  - label of candidate B
- `numerics.size`
  - size of the flat training input arrays
- `numerics.graphBlocks`
  - how many repeated optimizer-like blocks the graph contains
- `numerics.broadcastB0`
- `numerics.broadcastB1`
- `numerics.broadcastF`
  - shape parameters for the broadcast scenario
- `numerics.seed`
  - RNG seed

## Stage Syntax

`NumericsHarness.parseStages(...)` accepts:

- `NONE`
- `AR`
- `AR,CSE`
- `AR+CSE+FUSE`

This is useful especially for quick A/B checks:

- no optimization vs rewrite-only
- rewrite-only vs rewrite+CSE
- inference-like stage order vs training-like stage order

## Tolerance Policy

Default policy is chosen by dtype:

- `FLOAT64`
  - `absTol = 1e-12`
  - `relTol = 1e-12`
  - `maxUlpTol = 16`
- `FLOAT32`
  - `absTol = 1e-5`
  - `relTol = 1e-5`
  - `maxUlpTol = 128`

Verdict can be:

- `SAFE`
- `BORDERLINE`
- `UNSAFE`

### Meaning Of Verdicts

- `SAFE`
  - everything stayed within abs/rel and ULP tolerance
- `BORDERLINE`
  - some metrics exceeded the main tolerance but stayed within an acceptable ULP band
  - or ULP drift exceeded the bound while absolute error remained small
- `UNSAFE`
  - invalid values occurred
  - or tolerance was exceeded significantly without a reasonable explanation

This is not a mathematical proof of correctness. It is a pragmatic guardrail for fast regression checks.

## Example Output

Report output looks roughly like this:

```text
Numerics Report
scenario=benchmark-like, A=baseline, B=optimized
out: maxAbs=1.234e-06, avgAbs=2.100e-08, maxRel=7.000e-07, maxUlp=5, p50Ulp=0, p95Ulp=1, invalid=0
gradA: ...
gradB: ...
gradC: ...
broadcast: ...
aggregate: maxAbs=1.234e-06, maxRel=7.000e-07, maxUlp=5, invalid=0
verdict=SAFE (within abs/rel and ulp tolerance)
```

## Real Usage Patterns

### 1. Verifying a new optimizer stage combination

Use:

- `stageA=AR,CSE`
- `stageB=AR,CSE,FUSE`

Why:

- you can quickly see whether enabling fusion worsened numerics beyond a reasonable bound

### 2. Rewrite regression check

Use:

- `stageA=NONE`
- `stageB=AR`

Why:

- you validate that the rewrite family did not damage forward outputs or gradients

### 3. Broadcast audit

Increase:

- `numerics.broadcastB0`
- `numerics.broadcastB1`
- `numerics.broadcastF`

Why:

- you force more weight onto the broadcast-heavy scenario

## What The Harness Does Not Guarantee

It does not guarantee:

- coverage of all operation families
- coverage of all dtype/layout corner cases
- detection of performance regressions
- detection of all long-tail NaN/Inf problems in deep networks

It is a fast smoke/regression harness, not a formal numerics certification layer.

## Extending The Harness

If you want to add a new scenario:

1. add a deterministic graph recipe to `NumericsGraphFactory`
2. decide which signals to collect from it
3. extend `OutputSet`
4. extend metrics and the report
5. keep scenarios small and reproducible

Do not turn this into:

- a benchmark suite
- a workload zoo with twenty configurations
- a second autotune framework

## Common Mistakes

- treating `SAFE` as proof of absolute correctness
- comparing candidates with different inputs
- mixing the numerics harness with performance benchmarking
- adding synthetic scenarios without real diagnostic value

## Related Modules

- graph: [../graph/README.md](../graph/README.md)
- tuning: [../tuning/README.md](../tuning/README.md)
