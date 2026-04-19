# Numerics Harness

The `numerics` module is a standalone A/B harness for numerical drift checks.

Its purpose is:

- run two executable profile variants on the same deterministic inputs
- compare outputs and selected gradients
- summarize numeric drift with practical metrics and verdicts

It is not:

- a performance benchmark
- a replacement for tests
- a proof of full mathematical correctness

## Main Components

- CLI:
  - [NumericsCli.java](./NumericsCli.java)
- orchestration:
  - [NumericsHarness.java](./NumericsHarness.java)
- graph recipes:
  - [NumericsGraphFactory.java](./NumericsGraphFactory.java)
- metrics:
  - [NumericsMetrics.java](./NumericsMetrics.java)
- policy:
  - [NumericsPolicy.java](./NumericsPolicy.java)
- report:
  - [NumericsReport.java](./NumericsReport.java)

## What It Measures

The harness currently compares five signals:

- `out`
- `gradA`
- `gradB`
- `gradC`
- `broadcast`

For each signal it computes statistics such as:

- `maxAbs`
- `avgAbs`
- `maxRel`
- `maxUlp`
- `p50Ulp`
- `p95Ulp`
- `invalidCount`

## What It Actually Runs

The harness currently uses two scenario types.

### 1. Training-style graph

Built from repeated elementwise and linear-style blocks.

Used to compare:

- forward output
- gradients of several leaf inputs

This scenario runs in:

- `FORWARD_BACKWARD`

### 2. Broadcast-heavy graph

A simpler broadcast-heavy forward graph used to stress:

- shape/broadcast behavior
- elementwise paths

This scenario runs in:

- `FORWARD`

## Why Two Scenarios

One graph alone often misses important drift patterns.

The split intentionally covers both:

- deeper training-style behavior
- flatter broadcast-heavy behavior

## Determinism

Inputs are deterministic:

- seed controlled by `numerics.seed`
- both candidates see identical arrays

Without this, the harness would compare different random inputs instead of different execution policies.

## Candidate Model

The basic CLI primarily compares graph-policy variants.

Typical CLI setup:

- same dtype
- different stage orders
- same runtime defaults

So numerics CLI is best read as:

- "did this optimizer policy change drift numerically?"

not:

- "which runtime threshold is faster?"

## CLI Example

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

## Tolerance Policy

Default tolerance depends on dtype.

### `FLOAT64`

- `absTol = 1e-12`
- `relTol = 1e-12`
- `maxUlpTol = 16`

### `FLOAT32`

- `absTol = 1e-5`
- `relTol = 1e-5`
- `maxUlpTol = 128`

Possible verdicts:

- `SAFE`
- `BORDERLINE`
- `UNSAFE`

## Worked Example

Suppose candidate A and B produce for one signal:

- `maxAbs = 2.0e-6`
- `maxRel = 8.0e-7`
- `maxUlp = 4`
- `invalidCount = 0`

For `FLOAT32`, that is comfortably inside the default tolerance band and is likely reported as `SAFE`.

If instead:

- `invalidCount > 0`
- or ULP drift explodes without tiny absolute error

the verdict can move to `BORDERLINE` or `UNSAFE`.
