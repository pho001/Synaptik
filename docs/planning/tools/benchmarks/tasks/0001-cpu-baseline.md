# 0001 — Reproducible [CPU](../../../../glossary.md#term-cpu) [lifecycle](../../../../glossary.md#lifecycle) baseline

## Editorial reading key: interpreting the recorded baseline

In this task the harness, not a user-facing neural-network layer, calls `Engine.standard()` and
the compile/prepare/run lifecycle on deterministic inputs. It observes a checksum after an
explicit host copy. That checksum makes the result observable; it does not prove every numerical
edge case or justify a production route change. The historical `InferenceSession` wording below
does not name the current prepared handle; consult the
[Engine lifecycle and migration boundary](../../../../reference/engine/lifecycle.md).

The table's durations are nanoseconds, not per-element costs. As a unit-reading example only,
$422500\,\mathrm{ns}=0.4225\,\mathrm{ms}$ because one millisecond contains $10^6$ nanoseconds.
This converts the first recorded pointwise run sample; it neither reruns that workload nor
compares it with a later machine. Read
[benchmark sample interpretation](../../../../developer-guide/benchmarking.md#prerequisites-and-terms)
for warmup, sample and median meanings. All table values and original verification claims below
remain unchanged.


Status: Complete
Follow-up required: None for the fixed CPU lifecycle baseline. Broader [operation](../../../../glossary.md#operation)-family and [model](../../../../glossary.md#term-model) suites remain owned by tasks 0002 and 0003.

Depends on: Operational [Engine](../../../../glossary.md#term-engine) lifecycle and stable public [Tensor](../../../../glossary.md#tensor) contracts (satisfied).
Conflicts with: None.
Parallel group: Tools/Benchmarks 0001; isolated worktree required.

## Goal

Provide a runnable report-only [CLI](../../../../glossary.md#term-cli) baseline over the public `Engine.standard()` [lifecycle](../../../../glossary.md#lifecycle). Fixed deterministic workloads record [compile](../../../../glossary.md#compile), [prepare](../../../../glossary.md#prepare), repeated prepared runs, one-shot compute timing, and explicit [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization) checksums.

## Acceptance

- `./gradlew :tools:benchmarks:benchmark -Pprofile=smoke` emits one machine-readable [JSON](../../../../glossary.md#term-json) report and exits zero.
- Mandatory [pointwise](../../../../glossary.md#term-pointwise), [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), reduction, [normalization](../../../../glossary.md#term-normalization), [Conv2d](../../../../glossary.md#convolution--conv2d), and [Conv3d](../../../../glossary.md#conv3d--ncdhw) workloads all [compile](../../../../glossary.md#compile), [prepare](../../../../glossary.md#prepare), and [run](../../../../glossary.md#term-run) through the public [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle).
- [`smoke`](../../../../glossary.md#term-smoke-test) and `baseline` select bounded [warmup](../../../../glossary.md#term-warmup)/measurement counts; no production settings or caches are changed.
- Environment, [shapes](../../../../glossary.md#shape), lifecycle timings, raw repeated [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), and [checksum](../../../../glossary.md#term-checksum) evidence are reported; unavailable [backend](../../../../glossary.md#backend) routes remain explicit rather than fabricated.
- No [JMH](../../../../glossary.md#term-java-microbenchmark-harness-jmh), external [dependency](../../../../glossary.md#term-dependency), correctness-test substitution, or placeholder remains.

## Reproducible evidence

Command, from repository root:

```bash
./gradlew :tools:benchmarks:benchmark -Pprofile=smoke
```

Environment reported by the [run](../../../../glossary.md#term-run): Java `26.0.1`, Mac OS X [`aarch64`](../../../../glossary.md#term-aarch64); profile [`smoke`](../../../../glossary.md#term-smoke-test); one [warmup](../../../../glossary.md#term-warmup) and two measurement iterations. The command completed with `BUILD SUCCESSFUL` and exit code 0. The report [schema](../../../../glossary.md#term-schema) was `1` and reported elapsed time `395300625` ns.

| Workload | [Shape](../../../../glossary.md#shape) | [Compile](../../../../glossary.md#compile) ns | [Prepare](../../../../glossary.md#prepare) ns | Repeated run ns | One-shot compute total ns | [FNV-1a](../../../../glossary.md#term-fnv-1a) [checksum](../../../../glossary.md#term-checksum) |
|---|---|---:|---:|---|---:|---:|
| [pointwise](../../../../glossary.md#term-pointwise) | `Shape[2, 3]` | 24206208 | 78331833 | `[422500, 449875]` | 8400541 | 2673275336379922148 |
| matmul | `Shape[2, 4]` | 2408000 | 13607291 | `[234584, 197708]` | 7737084 | -8400416335202307302 |
| [conv2d](../../../../glossary.md#convolution--conv2d) | `Shape[1, 3, 6, 6]` | 1143625 | 7241250 | `[688125, 641500]` | 5773750 | 7597499385349807496 |
| [conv3d](../../../../glossary.md#conv3d--ncdhw) | `Shape[1, 3, 3, 3, 3]` | 1140416 | 6229584 | `[719000, 674208]` | 5429876 | -1899628152893975126 |
| reduction | `Shape[]` | 1229834 | 14395792 | `[401542, 361792]` | 5147708 | 4904312788079628870 |
| [normalization](../../../../glossary.md#term-normalization) | `Shape[4, 16]` | 789583 | 15777625 | `[405709, 319958]` | 5291875 | 5131093269821768582 |

Every mandatory family has non-empty compile and prepare timing, two repeated prepared-run [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), one-shot compute timing, and a checksum. The report contains no unavailable rows.

## Limitations

Timing uses `System.nanoTime()` and is a [microbenchmark](../../../../glossary.md#term-microbenchmark) observation, not a [release](../../../../glossary.md#term-release) gate. Results are sensitive to [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [warmup](../../../../glossary.md#term-warmup), scheduler, thermal, and operating-system noise and are not correctness evidence. Checksums prove that [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization) was observed, not numerical [conformance](../../../../glossary.md#term-conformance). The [smoke](../../../../glossary.md#term-smoke-test) profile is intentionally bounded; `baseline` provides three warmups and ten measurements for a longer comparison. [Benchmarking](../../../../glossary.md#term-benchmarking) is report-only and does not select routes, tune settings, write caches, or substitute for correctness/conformance tests.

## [CPU](../../../../glossary.md#term-cpu)-default evidence extension

[CPU task 0010M](../../../backends/cpu/tasks/0010m-evidence-backed-production-defaults.md)
extended the completed harness without changing its report-only boundary. Report [schema](../../../../glossary.md#term-schema) 3 uses a
reusable public [`InferenceSession`](../../../../glossary.md#former-inference-session--inferencesession), separates timed repeated runs from untimed [host](../../../../glossary.md#term-host)
[materialization](../../../../glossary.md#materialization)/checksums, retains per-[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) iteration counts plus raw normalized and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)
samples, and adds a strict `evidence` profile. Every retained repeated batch accumulates all timed
executions until it reaches 25 ms; the profile fixes both [heap](../../../../glossary.md#term-heap) bounds and [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) flags and requires
source, configuration, [fork](../../../../glossary.md#term-fork-meanings)/order, CPU, species, worker, materialization, and [OpenBLAS](../../../../glossary.md#term-openblas) [metadata](../../../../glossary.md#term-metadata).

The original [scalar](../../../../glossary.md#term-scalar)/[vector](../../../../glossary.md#term-vector) attempt was invalidated and removed. Five regenerated scalar reports
validate the hardened floor and metadata behavior only; they explicitly lack private selected-plan
facts and are ineligible to authorize a production change. The exact commands and [hashes](../../../../glossary.md#term-hash) plus a
minimum future-comparison acceptance framework are retained under
`backends/cpu/evidence/cpu-0010m-defaults/`; the framework is not measurement-[ready](../../../../glossary.md#term-ready). This benchmark
task still neither selects nor mutates production settings.

## Verification

- `./gradlew :tools:benchmarks:build` — passed.
- `./gradlew :tools:benchmarks:test` — passed; the benchmark [module](../../../../glossary.md#term-module) currently has no test sources, so all applicable tasks were up to date.
- `./gradlew :tools:benchmarks:benchmark -Pprofile=smoke` — passed; all six mandatory workloads completed as recorded above.
