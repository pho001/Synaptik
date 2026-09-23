# Task 0010M: Evidence-backed CPU production defaults

## Status

Complete

Readiness verification: the user authorized this isolated execution from exact clean base
`48354c47e93a4b9aa10bee329dd619d3d84f0019`; CPU 0009G1 and 0010L, Engine 0017, and benchmark
0001 are Complete. No concurrent write task owns these files.

## Change class

Class C — the investigation evaluates generated CPU hot-path strategy and concurrency defaults.
The accepted result changes no production behavior, public API, numerical contract, module edge,
Runtime policy, or native ABI. It hardens observational benchmark tooling. A fresh independent
targeted review was required because an earlier candidate and its evidence were blocked.

## Goal

Inventory every current production CPU tuning default, establish a reproducible evidence protocol,
and change only defaults supported across the complete enabled policy scope and representative
target/JIT strata. Prefer an explicit evidence-backed no-change to a machine-local overclaim.

## Contracts

- [`backend-execution.md`, Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  keeps benchmarking report-only and route/default selection in CPU preparation.
- [`backend-execution.md`, CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  keeps scalar, Vector API, generated, specialized, fused, and OpenBLAS routes behind one CPU owner.
- [`runtime-prepare-engine.md`, Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  requires route/resource choice before Runtime execution.
- [`runtime-prepare-engine.md`, Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)
  forbids runtime search and preserves fixed prepared execution.

Architecture impact: none. Existing contracts already authorize backend-private safe heuristics;
this task found insufficient evidence to change one.

## Dependencies and integration

- Depends on: CPU 0009G1, CPU 0010L, Engine 0017, benchmarks 0001
- Conflicts with: CPU production composition/defaults and shared CPU/benchmark planning or guidance
- Parallel group: None
- Common base revision: `48354c47e93a4b9aa10bee329dd619d3d84f0019`
- Integration order: Any
- Integration validation: affected CPU/benchmark builds, public lifecycle smoke/evidence runs,
  documentation/evidence validation, repository-wide tests once, fresh independent Class C review
- Shared-document integration owner: this task

## Production inventory and retained decision

| Decision | Current production/default | Guardrail or fallback |
|---|---|---|
| Ordinary portable compute | `SCALAR` | `CpuPartitionAnalysisInputs.DEFAULT`; unchanged |
| Portable parallelism | configured/available `1`; minimum elements/worker `1` | Budget capacity `1`; no worker group |
| Vector species | preferred hardware species only when an explicit typed input requests vector-if-eligible | Complete family/type/access/lane eligibility; scalar fallback |
| Pointwise vector eligibility | at least one preferred lane; supported homogeneous lane type or BOOL; vector-capable opcodes; no cross-type cast; eligible access | BFLOAT16, direct scalar POW, unsupported/mixed forms, short extents, and general odometer stay scalar |
| Convolution vector eligibility | direct Conv2d/Conv3d, homogeneous FLOAT32/FLOAT64, dense boundaries, width stride/dilation `1`, full interior lane block; Conv2d epilogue `NONE` | Incompatible compositions, edges, layouts, types, and widths stay scalar |
| Loss vector eligibility | dense homogeneous FLOAT32/FLOAT64 MSE `NONE` with one lane | Every other loss form stays scalar |
| Pooling and attention | scalar | No production vector threshold |
| Reductions/normalization | scalar whole-cell; BatchNorm channel range at parallelism `1` | Partial recipe unreachable without trusted evidence |
| Movement/indexing/scatter/fold/order/random/scan/affine | scalar | No production vector threshold |
| MATMUL | scalar direct for zero work, matrix-vector, or fewer than 4,096 contributions; scalar 2x2 at 16,384; explicit vector candidates can use direct-N or 2x2 at 65,536 | Complete route facts agree or fail closed |
| OpenBLAS | optional qualified single-thread FLOAT32/FLOAT64 MATMUL | Portable cost `1000 + 10*outputs + 100*macs`; native `1 + outputs + macs`; margins `1`; portable fallback |
| Materialization | disabled | No speculative production copy |
| Fusion | margin `32`; profitable code/live ceilings `48`/`12`; decision facts `384` | Canonical split on uncertainty/tie |
| Decomposition | nodes `8`, attempts `28`, boundaries `16`, live `16`, indexing `32`, code `64`, candidates `64`, enumeration attempts `256` | Bounded fail closed |
| Recognition | attempts `24`, facts `8`, members `6`, boundaries `10`, units `2`, epilogue ops `2` | Reject over-budget recognition |
| Representation | eligible sources `8`, copy choices `36`, variants `37` | Production materialization remains disabled |
| Specialization | candidates `4`, artifacts `1`, fixed/unrolled variants `0` | No Runtime specialization |
| Preparation identity | manifest/persistence off; derived exact segment carriers; exact numerics; unspecified hardware; ordinary cohort | Typed explicit inputs only |

No row changes. A proposed global `VECTOR_IF_ELIGIBLE` escalation was reverted after independent
review showed that its workload matrix covered only dense FLOAT32 ADD and convolution while the
policy also enabled MATMUL, MSE, floating transcendental, integral, BOOL/mask, broadcast, and fused
pointwise forms.

## Invalidated attempt and historical search

The original 13 scalar/vector/parallel files were removed. They omitted source, command, CPU,
species, and selected-plan identities; ran all scalar processes before all vector processes; pooled
within-fork samples as independent observations; and had 81 of 420 repeated batches below the
claimed 25 ms floor. Their aggregates are not retained or cited.

Targeted sealed-artifact search found generated-versus-direct parity and machine-local macOS
AArch64 observations, not compatible scalar-versus-vector evidence spanning the current policy
matrix or x86-64. Generated-versus-direct parity is not permission to change scalar/vector policy.
A universal default therefore has no adequate cross-target basis.

## Hardened report protocol

Schema 3 evidence runs require exact base revision, Git tree object format/OID, configuration, and
SHA-256 harness/production hashes; fork/order facts; CPU identity/features; portable values;
worker/materialization/OpenBLAS policy; and `-Xms1g -Xmx1g -XX:-TieredCompilation -Xbatch`.
Gradle normalizes the profile with `Locale.ROOT`, so mixed-case `EVIDENCE` receives the same flags
and requirements.

Each retained sample accumulates timed executions until its own elapsed time reaches 25 ms; no
execution, sample, fork, retry, or outlier is discarded. Reports retain per-sample iteration counts,
batch durations, normalized times, summaries, one-shot observations, checksums, preferred species,
VM arguments, and a hash of loaded harness class bytes.

Public Engine deliberately hides backend-private selected-plan facts. Schema 3 reports record that
limitation and set `eligibleForProductionDecision=false`; they cannot authorize a route/default
change. The manifest's minimum future acceptance framework requires a later comparison to fail
when actual route, strategy, species, ranges, workers, materializations, or OpenBLAS selection are
unavailable.

`backends/cpu/evidence/cpu-0010m-defaults/manifest.json` retains the exact scalar-baseline commands
and hashes. Its future section is not a measurement-ready protocol: a separately reviewed task
must seal exact CPU/OS/JDK/species strata; every operation/topology/type/shape/layout/access/carrier/
data case; randomized interleaved order and seed; family aggregation and weighting; and the exact
bootstrap resample count, PRNG, sidedness, and quantile convention before either configuration is
measured. The retained framework supplies minimum floor, fork-unit, correctness, regression,
improvement, policy-scope, and selected-plan requirements only.

## Regenerated scalar baseline

Five fresh JVMs validate the hardened protocol against unchanged scalar production on Apple M3
Max `Mac15,9`, macOS 26.6.2 AArch64, OpenJDK 26.0.1+8-34, and preferred 128-bit species. All 210
retained repeated batches exceed 25 ms; minimum is 25,170,042 ns. Every within/across-fork checksum
is stable. Median-of-fork-medians in nanoseconds is: pointwise `12,951,333`, MATMUL `266,271`,
Conv2d `28,719,791`, Conv3d `37,864,125`, reduction `15,876,833`, normalization `29,217,000`.

These values are a machine-local scalar baseline and harness proof, not a candidate comparison,
speedup claim, selected-route proof, or production-decision artifact.

## Acceptance criteria

- Production composition remains scalar/single-threaded with every existing fallback and threshold.
- Evidence profile spelling cannot bypass evidence JVM flags or required metadata.
- Every retained repeated batch meets its floor; reports retain exact source/config/environment and
  explicitly refuse production-decision eligibility without actual selected-plan facts.
- Invalid evidence and aggregates are absent; the manifest is hash-valid and explicitly prevents
  future measurement until a reviewed task seals every currently open protocol choice.
- CPU/benchmark tests, builds, Javadocs, smoke/evidence execution, evidence/docs validation,
  repository-wide tests, and fresh independent Class C review pass before status becomes Complete.

## Validation and result

Completed worker checks after review remediation:

- Affected CPU/benchmark tests, builds, and Javadocs — passed via `:backends:cpu:test`,
  `:backends:cpu:javadoc`, `:tools:benchmarks:build`, and `:tools:benchmarks:javadoc`.
- Final `-Pprofile=EVIDENCE` lifecycle execution — passed with evidence JVM flags and all six
  workloads, proving case-normalized profile handling.
- Five scalar evidence JVMs — passed; 210/210 retained batches meet the floor and all checksums are
  stable. A deterministic integrity check rehashed every report/source artifact and verified exact
  metadata, JVM flags, orders, production-decision ineligibility, checksums, and aggregate facts.
- Ten changed Markdown files passed local-link, fence, final-newline, and trailing-whitespace
  validation; `git diff --check` passed.
- Narrow independent Class C re-review — approved after verifying honest Git object naming, exact
  order vocabulary, and the non-measurement-ready future acceptance framework.
- Final unchanged-worktree `./gradlew test` — `BUILD SUCCESSFUL`; 72 tasks: 3 executed, 37 from
  cache, and 32 up-to-date.
- Final unchanged-worktree `./gradlew build` — `BUILD SUCCESSFUL`; 84 tasks: 11 executed and 73
  up-to-date.
- `git diff --check` passed before and after the repository-wide validation; worktree status was
  identical before and after.

All acceptance criteria passed. The evidence-backed conclusion is no production-default change:
ordinary CPU execution remains scalar with configured/available parallelism `1`/`1`, minimum
elements per worker `1`, no worker group, and every existing fallback and threshold retained. The
benchmark protocol is hardened; any future candidate comparison still requires a separately
reviewed, fully sealed matrix before measurement.
