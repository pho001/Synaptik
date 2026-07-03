# Benchmarking

## What you will learn

This guide defines the evidence expected from future Synaptik performance measurements. The `tools/benchmarks` module exists structurally but has no benchmark harness or workload implemented yet.

## Mental model

Correctness tests answer whether a result is valid. Benchmarks answer how a fixed workload behaves under a recorded environment. A benchmark never substitutes for unit, conformance, or integration tests.

## Required measurement record

For each future benchmark, record the exact commit, JDK, operating system, hardware, native-library versions, warmup policy, measurement iterations, input data types and shapes, backend and route, thread settings, and summary statistic. Keep setup work separate from the measured operation where the distinction matters.

A useful workload describes its arithmetic. For matrix multiplication `[64, 128] × [128, 32]`, record 2,048 outputs and 262,144 multiply contributions. This lets reviewers compare runs using the same logical work rather than only a benchmark name.

## Current validation

```bash
./gradlew :tools:benchmarks:build
```

Today this only validates the empty project structure. The [benchmarks master plan](../planning/tools/benchmarks/master-plan.md) governs later harness and reporting work.

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|
| A result cannot be reproduced | Environment or workload inputs were omitted. | Record all required measurement facts. |
| Compile time is mixed into run time | Lifecycle boundaries were not isolated. | Report compile, prepare, and run measurements separately. |
| Faster code changes numerical behavior | Performance was accepted without correctness evidence. | Run reference and conformance tests before interpreting speed. |

## Related documentation

- [Performance discipline in `AGENTS.md`](../../AGENTS.md)
- [CPU kernel strategy](../design/notes/cpu-kernel-strategy.md)
- [Benchmark master plan](../planning/tools/benchmarks/master-plan.md)
