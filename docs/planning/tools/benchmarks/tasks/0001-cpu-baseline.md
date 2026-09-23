# 0001 — Reproducible CPU lifecycle baseline

Status: Complete
Follow-up required: None for the fixed CPU lifecycle baseline. Broader operation-family and model suites remain owned by tasks 0002 and 0003.

Depends on: Operational Engine lifecycle and stable public Tensor contracts (satisfied).
Conflicts with: None.
Parallel group: Tools/Benchmarks 0001; isolated worktree required.

## Goal

Provide a runnable report-only CLI baseline over the public `Engine.standard()` lifecycle. Fixed deterministic workloads record compile, prepare, repeated prepared runs, one-shot compute timing, and explicit host materialization checksums.

## Acceptance

- `./gradlew :tools:benchmarks:benchmark -Pprofile=smoke` emits one machine-readable JSON report and exits zero.
- Mandatory pointwise, MATMUL, reduction, normalization, Conv2d, and Conv3d workloads all compile, prepare, and run through the public Engine lifecycle.
- `smoke` and `baseline` select bounded warmup/measurement counts; no production settings or caches are changed.
- Environment, shapes, lifecycle timings, raw repeated samples, and checksum evidence are reported; unavailable backend routes remain explicit rather than fabricated.
- No JMH, external dependency, correctness-test substitution, or placeholder remains.

## Reproducible evidence

Command, from repository root:

```bash
./gradlew :tools:benchmarks:benchmark -Pprofile=smoke
```

Environment reported by the run: Java `26.0.1`, Mac OS X `aarch64`; profile `smoke`; one warmup and two measurement iterations. The command completed with `BUILD SUCCESSFUL` and exit code 0. The report schema was `1` and reported elapsed time `395300625` ns.

| Workload | Shape | Compile ns | Prepare ns | Repeated run ns | One-shot compute total ns | FNV-1a checksum |
|---|---|---:|---:|---|---:|---:|
| pointwise | `Shape[2, 3]` | 24206208 | 78331833 | `[422500, 449875]` | 8400541 | 2673275336379922148 |
| matmul | `Shape[2, 4]` | 2408000 | 13607291 | `[234584, 197708]` | 7737084 | -8400416335202307302 |
| conv2d | `Shape[1, 3, 6, 6]` | 1143625 | 7241250 | `[688125, 641500]` | 5773750 | 7597499385349807496 |
| conv3d | `Shape[1, 3, 3, 3, 3]` | 1140416 | 6229584 | `[719000, 674208]` | 5429876 | -1899628152893975126 |
| reduction | `Shape[]` | 1229834 | 14395792 | `[401542, 361792]` | 5147708 | 4904312788079628870 |
| normalization | `Shape[4, 16]` | 789583 | 15777625 | `[405709, 319958]` | 5291875 | 5131093269821768582 |

Every mandatory family has non-empty compile and prepare timing, two repeated prepared-run samples, one-shot compute timing, and a checksum. The report contains no unavailable rows.

## Limitations

Timing uses `System.nanoTime()` and is a microbenchmark observation, not a release gate. Results are sensitive to JVM warmup, scheduler, thermal, and operating-system noise and are not correctness evidence. Checksums prove that host materialization was observed, not numerical conformance. The smoke profile is intentionally bounded; `baseline` provides three warmups and ten measurements for a longer comparison. Benchmarking is report-only and does not select routes, tune settings, write caches, or substitute for correctness/conformance tests.

## Verification

- `./gradlew :tools:benchmarks:build` — passed.
- `./gradlew :tools:benchmarks:test` — passed; the benchmark module currently has no test sources, so all applicable tasks were up to date.
- `./gradlew :tools:benchmarks:benchmark -Pprofile=smoke` — passed; all six mandatory workloads completed as recorded above.
