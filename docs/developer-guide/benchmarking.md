# Benchmarking

## What you will learn

This guide defines the evidence expected from future Synaptik benchmarks and separates
benchmarking from model autotuning, planning cost, and runtime profiling. The
`tools/benchmarks` project exists structurally, but no harness, `BenchmarkReport`, or workload is
implemented.

## Prerequisites and terms

Read the [performance evidence and tuning boundary](../architecture/performance-evidence-and-tuning.md)
and the [performance glossary entries](../glossary.md#benchmark-report--benchmarking) before
designing a harness.

- **Benchmarking** runs a fixed reproducible workload and reports measurements without changing
  production settings.
- A **benchmark report** is the rich immutable evidence from one recorded benchmark run.
- **Model autotuning** explicitly reuses or measures actual model workloads, then compares a
  bounded set of complete valid model plans.
- **Runtime profiling** passively observes actual prepared execution.

## Mental model

```text
correctness tests -> is the result valid?
benchmark report  -> how did this fixed workload behave here?
model autotuning  -> which compatible local results and complete plan best meet the objective?
runtime profiling -> what happened during actual prepared execution?
```

A benchmark answers only the second question. It never substitutes for unit, conformance, or
integration tests and never installs the fastest measured setting.

## Benchmark workloads and reports

Future benchmark suites may contain four fixed workload levels:

- one operation occurrence;
- a representative operation-family workload;
- a complete model workload; and
- end-to-end compile, prepare, and run workloads.

For every report, record the exact commit, model or workload identity, JDK, operating system,
hardware, native-library versions, warmup policy, measurement iterations, input data types and
shapes, backend and supplied route/configuration, thread settings, lifecycle boundary, and summary
statistics. Keep raw samples or equivalent distribution evidence when the selected reporting
policy requires it.

Measurement evidence remains richer than any later compact planning-cost profile, workload tuning
cache, or model-plan record. A production artifact may retain selected values and compatibility
identity; the report retains the environment, workload, candidates, samples, and statistics that
justify them.

## Complete conceptual example

### Inputs and initial state

Assume a future operation benchmark measures matrix multiplication `[64, 128] × [128, 32]` on one
recorded CPU configuration. The fixed logical work is:

```text
output values:          64 × 32 = 2,048
multiply contributions: 64 × 128 × 32 = 262,144
```

### Procedure and intermediate evidence

The harness warms up according to the report policy, measures only the chosen lifecycle stage,
and writes a report containing the fixed inputs, environment, supplied CPU route parameters, raw
or aggregated samples, and summary statistic.

### Result and interpretation

Running the same report definition on two commits permits a performance comparison. The faster
result does not update a CPU route threshold, vector strategy, thread count, workload tuning
cache, or model plan. A separate explicit model-autotuning workflow may consume compatible
measurement machinery, run a declared bounded search, and produce its own explicit cache and plan
artifacts.

## Model-autotuning candidate spaces

Model autotuning belongs to `tools/tuning`, not the benchmark suite. A concrete backend owns typed,
version-controlled candidate generators beside its routes. An operation family selects the
appropriate generator, but cache reuse uses a canonical workload signature that also includes
semantics and attributes, data types, shapes, layouts, relevant policies, and target
compatibility. A single family-wide thread or vector value is not a valid general cache contract.

Hardware and supported JDK Vector API species constrain physical vector lanes. A CPU candidate
generator may return complete valid species/strategy, unroll, tile, parallelism, and OpenBLAS
thread configurations. It cannot promise an arbitrary lane count or expose private knobs through
a generic parameter map.

## Current validation

```bash
./gradlew :tools:benchmarks:build
```

Today this validates only the empty project structure. The [benchmarks master plan](../planning/tools/benchmarks/master-plan.md)
governs later harness and reporting work.

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|
| A result cannot be reproduced | Environment or workload inputs were omitted. | Record the complete workload and environment. |
| Compile time is mixed into run time | Lifecycle boundaries were not isolated. | Report compile, prepare, and run measurements separately. |
| A benchmark changes a later run's settings | Reporting was confused with tuning. | Remove side effects; use the explicit model-autotuning workflow. |
| One vector or thread value is applied everywhere | Operation family was incorrectly used as a universal cache key. | Key reuse by canonical workload signature and keep candidate vocabulary backend-owned. |
| Faster code changes numerical behavior | Performance was accepted without correctness evidence. | Run reference and conformance tests before interpreting speed. |

## Limitations and boundaries

Benchmark, model-autotuning, planning-cost, and runtime-profile contracts remain planned.
`modules/config` may later store immutable declarative outputs, but it will not own the runner,
search algorithm, live discovery, or mutable evidence. No benchmark runs in the runtime hot path.

## Related documentation

- [Performance discipline in `AGENTS.md`](../../AGENTS.md)
- [Performance evidence and tuning](../architecture/performance-evidence-and-tuning.md)
- [CPU kernel strategy](../design/notes/cpu-kernel-strategy.md)
- [Benchmark master plan](../planning/tools/benchmarks/master-plan.md)
- [Tuning master plan](../planning/tools/tuning/master-plan.md)
