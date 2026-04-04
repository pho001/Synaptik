# Legacy Benchmark Review

## Contents

- [Goal of This Review](#goal-of-this-review)
- [What Still Has Value](#what-still-has-value)
- [What Is Now Legacy](#what-is-now-legacy)
- [What Should Not Be Carried Forward](#what-should-not-be-carried-forward)
- [Recommended Keep / Freeze / Retire Plan](#recommended-keep--freeze--retire-plan)

## Goal of This Review

This document answers one question:

- after building the new `tuning` package, what from the old `benchmark` package is still worth keeping?

The answer is intentionally pragmatic.

## What Still Has Value

These parts still carry real value and can continue to live for some time:

### 1. Existing benchmark entry programs

Examples:

- [MatMulBenchmark.java](../benchmark/MatMulBenchmark.java)
- [Conv2dLoweringBenchmark.java](../benchmark/Conv2dLoweringBenchmark.java)
- [TransformerHotPathBenchmark.java](../benchmark/TransformerHotPathBenchmark.java)

Reason:

- they are useful operational entry points today
- they still encode practical workload knowledge

### 2. Some scenario/data factory pieces

Examples:

- `ScenarioTensorFactory`
- `BenchmarkGraphRecipes`

Reason:

- they already capture some useful graph construction patterns
- parts of them can be migrated or reused gradually

### 3. Some low-level measurement utilities

Examples:

- timing helpers
- tier policy helpers

Reason:

- utility logic may still be worth extracting or copying if clean enough

## What Is Now Legacy

These areas are now conceptually legacy because the new `tuning` package supersedes them:

### 1. Old autotune orchestration

Everything under:

- `benchmark.autotune.*`

is now legacy from an architecture perspective.

Reason:

- the new `tuning` package now has:
  - candidate generation
  - benchmark sessions
  - validation
  - persistence
  - tree search
  - history-aware search

### 2. Old benchmark-first candidate model

Examples:

- `OptimizerCandidate`
- `TuningKnobs`
- parts of `OptimizerCandidateFactory`

These were useful for bootstrapping.
But the long-term correct source of truth is now:

- `ExecutionProfile`

### 3. Old benchmark-owned persistence chain

Examples:

- old best-profile workflow
- old unsafe-history workflow
- old benchmark-specific search support

These should not grow further as the main system.

## What Should Not Be Carried Forward

These patterns should be explicitly avoided:

### 1. Parallel execution model outside `ExecutionProfile`

We should not keep investing in:

- a second benchmark-only knob universe

### 2. Benchmark-first architecture

We should not keep:

- systems where benchmark requirements dictate runtime architecture

### 3. Large monolithic autotune flow classes

We should not continue growing:

- one giant benchmark/autotune framework class with many orthogonal responsibilities

## Recommended Keep / Freeze / Retire Plan

### Keep for now

- benchmark CLI-style entry classes
- scenario/data helpers that still provide practical value
- utility measurement helpers if still used by live workflows

### Freeze

- old autotune orchestration classes
- old candidate search support
- old benchmark-owned persistence logic

Meaning:

- do not extend them further except for compatibility or bug fixes

### Retire gradually

- old candidate and search models once equivalent `tuning` workflows are in active use
- old persistence once new stores cover the same operational use-cases
- old benchmark README sections that present legacy flow as the primary path

## Practical Decision

Right now the old `benchmark` package should still be kept as:

- operational tooling
- compatibility tooling
- migration reference

But not as:

- the future source of truth for benchmark/autotune architecture

The new `tuning` package is now the architectural destination.
