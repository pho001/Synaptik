# Numerics Harness

Lightweight A/B harness for numerical-stability checks independent of the benchmark/autotuner flow.

This module is now a standalone numerics comparison harness.

It can be used directly to compare two `ExecutionProfile` stage configurations over:

- one optimizer-like graph
- one broadcast-heavy graph
- forward outputs and selected gradients

## What It Compares
- forward output (`out`)
- gradients (`gradA`, `gradB`, `gradC`)
- broadcast expression output (`broadcast`)

For each signal:
- `maxAbs`, `avgAbs`, `maxRel`
- `maxUlp`, `p50Ulp`, `p95Ulp`
- `invalidCount` (`NaN`/`Inf`)

## CLI
Main class: `numerics.NumericsCli`

The CLI compares two `ExecutionProfile` stage selections over the same deterministic inputs.

Today it varies:

- dtype
- stage lists `numerics.stageA` / `numerics.stageB`
- graph size parameters

It keeps fixed:

- the benchmark-like graph recipe
- the broadcast-heavy graph recipe
- runtime config defaults

That makes it useful for:

- quick optimizer rewrite sanity checks
- numerics drift checks between two stage orders
- local investigation without involving the full tuning/search stack

Example:

```bash
java --add-modules jdk.incubator.vector \
  -Dnumerics.dtype=FLOAT32 \
  -Dnumerics.stageA=NONE \
  -Dnumerics.stageB=AR \
  -Dnumerics.size=200000 \
  -Dnumerics.graphBlocks=6 \
  -cp build/classes/java/main \
  numerics.NumericsCli
```

## Properties
- `numerics.dtype` = `FLOAT32|FLOAT64` (default `FLOAT32`)
- `numerics.stageA` (default `NONE`)
- `numerics.stageB` (default `AR`)
- `numerics.nameA` (default `A`)
- `numerics.nameB` (default `B`)
- `numerics.size` (default `200000`)
- `numerics.graphBlocks` (default `6`)
- `numerics.broadcastB0` (default `128`)
- `numerics.broadcastB1` (default `8`)
- `numerics.broadcastF` (default `128`)
- `numerics.seed` (default `42`)

## Default Policy

- `FLOAT64`: `absTol=1e-12`, `relTol=1e-12`, `maxUlpTol=16`
- `FLOAT32`: `absTol=1e-5`, `relTol=1e-5`, `maxUlpTol=128`

Verdicts:

- `SAFE`
- `BORDERLINE`
- `UNSAFE`
