# Numerics Harness

Lightweight A/B harness for numerical-stability checks independent of the benchmark/autotuner flow.

This module is also used by benchmark autotune finalists post-check when:

- `-Dbenchmark.autotuneNumericsPostcheck=true`
- reports are exported to `build/numerics/autotune-postcheck-<dtype>-<timestamp>.tsv`

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
