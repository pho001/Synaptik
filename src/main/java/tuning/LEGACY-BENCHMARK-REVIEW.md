# Legacy Benchmark Review

This file is a historical note, not a roadmap back to the old design.

Its purpose is to preserve what was worth learning from the earlier benchmark/autotune layer while making clear what should not return.

## What Was Worth Keeping

### Workload knowledge

This was genuinely valuable:

- realistic matmul scenarios
- conv2d scenarios
- transformer/attention hotspot scenarios

Scenario knowledge survives architectural rewrites because it is expensive to rediscover.

### Useful measurement ideas

The older layer was right to separate:

- compile
- prepare
- traced run
- steady-state

The current tuning layer keeps that distinction.

### Some data factories and scenario builders

If a helper captures realistic shapes and graph structure well, it remains valuable even if orchestration changes completely.

## What Became Legacy

### Benchmark-owned candidate universe

The old pattern where benchmark had its own candidate abstraction separate from executable profiles should not be the main path anymore.

Today the correct source of truth is:

- `ExecutionProfile`

### Monolithic orchestration

One large class that tries to:

- generate candidates
- validate
- measure
- search
- persist
- report

is hard to maintain and hard to reason about.

The current architecture intentionally splits those concerns.

### Benchmark-specific persistence

Persistence tied only to a benchmark runner is the wrong abstraction because:

- it is not reusable
- it mixes explain artifacts with runtime source of truth

## What Must Not Return

These are still anti-patterns today:

- a hidden execution model next to `ExecutionProfile`
- synthetic benchmark-only knob universes
- reports used as runtime source of truth
- compile/runtime decisions stored only as ad hoc benchmark metadata

## What Replaced The Old Model

Today:

- benchmark measures explicit `ExecutionProfile` candidates
- autotune searches explicit `ExecutionProfile` candidates
- calibration produces explicit `PlatformRuntimeProfile`
- persistence distinguishes:
  - runtime defaults
  - best profiles
  - history
  - explain artifacts

That is the main architectural improvement over the older benchmark-first design.
