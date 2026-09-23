# 0001 — Reproducible CPU lifecycle baseline

Status: Incomplete
Follow-up required: Supply a runnable current CPU capability/native environment and resolve mandatory workload failures; smoke must produce timings/checksums for every workload.

Depends on: Operational Engine lifecycle and stable public Tensor contracts (satisfied at common base `83f28005fbfcb6518af3fbd8760297c90ea12e1e`).
Conflicts with: None.
Parallel group: Tools/Benchmarks 0001; isolated worktree required.
Common base revision: `83f28005fbfcb6518af3fbd8760297c90ea12e1e`.
Integration order: Integrate after CPU lifecycle baseline review; no shared root-document edits.
Integration validation: `./gradlew :tools:benchmarks:build`; smoke command below.
Shared-document integration owner: Integrator; this task changes only benchmark guide and benchmark planning files.

## Goal

Provide a runnable report-only CLI baseline over the public `Engine.standard()` lifecycle. Fixed deterministic MATMUL, Conv2d, Conv3d, reduction, and normalization graphs record compile, prepare, repeated prepared runs, and one-shot compute timing with explicit host materialization checksums.

## Acceptance

- `./gradlew :tools:benchmarks:benchmark -Pprofile=smoke` emits one machine-readable JSON report.
- `smoke` and `baseline` select bounded warmup/measurement counts; no production settings or caches are changed.
- Environment, shapes, lifecycle timings, raw repeated samples, and checksum evidence are reported; unavailable backend routes are explicit rather than fabricated.
- No JMH, external dependency, correctness-test substitution, or placeholder remains.

## Limitations

Timing uses `System.nanoTime()` and is a microbenchmark observation, not a release gate. The current environment may report unavailable CPU ownership when its backend capability/native prerequisites are absent; the report retains that failure text. Results are not correctness evidence.

## Exact commands

```bash
./gradlew :tools:benchmarks:build
./gradlew :tools:benchmarks:benchmark -Pprofile=smoke
./gradlew :tools:benchmarks:benchmark -Pprofile=baseline
```
