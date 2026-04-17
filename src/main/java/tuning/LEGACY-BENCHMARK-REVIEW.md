# Legacy Benchmark Review

This document serves as a historical note: what was worth keeping from the original benchmark/autotune layer, and what should no longer be treated as an architectural model today.

It is not a roadmap for returning to the old design. It is mainly an explanation of why the current `tuning` package looks the way it does.

## Historical Problem

The older benchmark-first approach typically suffered from the fact that:

- benchmark had its own candidate model
- runtime had a different execution model
- persistence stored something different from what was actually run

The result:

- difficult maintenance
- drift between benchmark and runtime
- hard-to-interpret winners

## What Was Worth Keeping

Even from the old layer, some things were worth preserving.

### 1. Practical workload knowledge

For example:

- matmul scenarios
- conv2d scenarios
- transformer hot-path workloads

This is valuable, because workload know-how is expensive and should not be discarded just because orchestration changes.

### 2. Some data factories and scenario helpers

If they capture real workload shapes well, they remain valuable even after an orchestration rewrite.

### 3. Some measurement ideas

For example, separating:

- compile
- prepare
- traced run
- steady-state

This was a good direction, and the current tuning layer keeps it.

## What Became Legacy

### 1. Benchmark-Owned Candidate Universe

The old model of a "benchmark candidate" separate from `ExecutionProfile` should no longer be the main path.

Today the source of truth is:

- `ExecutionProfile`

### 2. Monolithic Autotune Flow

One huge class that:

- generates candidates
- measures
- validates
- stores
- decides strategy
- renders results

is a bad design.

The current tuning package splits this into:

- `candidate`
- `measure`
- `validate`
- `search`
- `store`
- `report`
- `session`

### 3. Benchmark-Specific Persistence

Persistence tied only to the old benchmark runner did not make sense, because:

- it was not reuse-friendly
- it mixed source of truth with explain data

## What Must Not Return

These anti-patterns should not come back under a new name:

- a second hidden execution model next to `ExecutionProfile`
- a benchmark-only knob universe
- storing a report as the source of truth
- synthetic candidate models that runtime will never run directly

## What The New Architecture Replaced It With

The current state:

- benchmark measures explicit `ExecutionProfile` candidates
- autotune searches explicit `ExecutionProfile` candidates
- platform calibration mutates explicit `PlatformRuntimeProfile`
- persistence distinguishes:
  - runtime defaults
  - best profile
  - history
  - explain artifacts

That is much cleaner than the benchmark-first architecture.

## Keep / Freeze / Retire

### Keep

- workload know-how
- sensible scenario builders
- useful measurement patterns

### Freeze

- historical compatibility fallbacks
- old path layouts under `build/...`

### Retire

- the old benchmark-first candidate mindset
- old documentation that presents benchmark as the main runtime architecture

## Practical Rule For New Work

When you add new tuning functionality today, ask:

- can this be expressed as `ExecutionProfile` or `PlatformRuntimeProfile`?

If not, there is a high chance that you are reintroducing the old problem.

## Why This Document Still Exists

Because it helps explain:

- why tuning separates platform defaults from workload winners
- why `ExecutionProfile` is the only execution source of truth
- why reports and history are not runtime artifacts

These are not just stylistic preferences. They are defensive mechanisms against regressing the architecture back into benchmark-first chaos.
