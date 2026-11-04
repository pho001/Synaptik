# Task 0010M: Evidence-backed [CPU](../../../../glossary.md#term-cpu) production defaults

## Status

Complete

Readiness verification: the user authorized this isolated execution from exact clean base
`48354c47e93a4b9aa10bee329dd619d3d84f0019`; [CPU](../../../../glossary.md#term-cpu) 0009G1 and 0010L, [Engine](../../../../glossary.md#term-engine) 0017, and benchmark
0001 are Complete. No concurrent write task owns these files.

## Change class

Class C — the investigation evaluates generated [CPU](../../../../glossary.md#term-cpu) hot-path strategy and [concurrency](../../../../glossary.md#term-concurrency) defaults.
The accepted result changes no production behavior, public [API](../../../../glossary.md#term-application-programming-interface-api), numerical contract, [module](../../../../glossary.md#term-module) edge,
[Runtime](../../../../glossary.md#term-runtime) policy, or native [ABI](../../../../glossary.md#term-application-binary-interface-abi). It hardens observational benchmark tooling. A fresh independent
targeted review was required because an earlier candidate and its evidence were blocked.

## Goal

Inventory every current production [CPU](../../../../glossary.md#term-cpu) tuning default, establish a reproducible evidence protocol,
and change only defaults supported across the complete enabled policy [scope](../../../../glossary.md#term-scope) and representative
[target](../../../../glossary.md#term-target)/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit) strata. Prefer an explicit evidence-backed no-change to a machine-local overclaim.

## Contracts

- [`backend-execution.md`, Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  keeps benchmarking report-only and route/default selection in CPU preparation.<span data-glossary-navigation="true"> ([benchmarking glossary](../../../../glossary.md#term-benchmarking); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- [`backend-execution.md`, CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  keeps scalar, Vector API, generated, specialized, fused, and OpenBLAS routes behind one CPU owner.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Vector API glossary](../../../../glossary.md#term-vector-api); [fused glossary](../../../../glossary.md#specialization-and-fusion); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [owner glossary](../../../../glossary.md#term-owner))</span>
- [`runtime-prepare-engine.md`, Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  requires route/resource choice before Runtime execution.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`runtime-prepare-engine.md`, Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>
  forbids runtime search and preserves fixed prepared execution.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>

Architecture impact: none. Existing contracts already authorize [backend](../../../../glossary.md#backend)-private safe heuristics;
this task found insufficient evidence to change one.

## Dependencies and integration

- Depends on: [CPU](../../../../glossary.md#term-cpu) 0009G1, CPU 0010L, [Engine](../../../../glossary.md#term-engine) 0017, benchmarks 0001
- Conflicts with: CPU production composition/defaults and shared CPU/benchmark [planning](../../../../glossary.md#term-planning) or guidance
- Parallel group: None
- Common base revision: `48354c47e93a4b9aa10bee329dd619d3d84f0019`
- Integration order: Any
- Integration validation: affected CPU/benchmark builds, public lifecycle smoke/evidence runs,
  documentation/evidence validation, repository-wide tests once, fresh independent Class C review<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle); [smoke glossary](../../../../glossary.md#term-smoke-test))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): this task

## Production inventory and retained decision

| Decision | Current production/default | Guardrail or fallback |
|---|---|---|
| Ordinary portable compute | [`SCALAR`](../../../../glossary.md#term-scalar) | `CpuPartitionAnalysisInputs.DEFAULT`; unchanged |
| Portable parallelism | configured/available `1`; minimum [elements](../../../../glossary.md#term-element)/worker `1` | Budget capacity `1`; no worker group |
| [Vector species](../../../../glossary.md#term-vector-species) | preferred hardware species only when an explicit typed input requests [vector](../../../../glossary.md#term-vector)-if-eligible | Complete family/type/access/[lane](../../../../glossary.md#term-lane) eligibility; scalar fallback |
| [Pointwise](../../../../glossary.md#term-pointwise) vector eligibility | at least one preferred lane; supported [homogeneous](../../../../glossary.md#term-homogeneous) lane type or [BOOL](../../../../glossary.md#data-type--datatype); vector-capable opcodes; no cross-type cast; eligible access | [BFLOAT16](../../../../glossary.md#term-bfloat16), direct scalar POW, unsupported/mixed forms, short extents, and general odometer stay scalar |
| [Convolution](../../../../glossary.md#term-channels-first-convolution-layers) vector eligibility | direct [Conv2d](../../../../glossary.md#convolution--conv2d)/[Conv3d](../../../../glossary.md#conv3d--ncdhw), homogeneous [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64), dense boundaries, width [stride](../../../../glossary.md#term-stride)/[dilation](../../../../glossary.md#term-dilation) `1`, full interior lane block; Conv2d [epilogue](../../../../glossary.md#term-epilogue) `NONE` | Incompatible compositions, edges, [layouts](../../../../glossary.md#layout), types, and widths stay scalar |
| [Loss](../../../../glossary.md#loss) vector eligibility | dense homogeneous FLOAT32/FLOAT64 [MSE](../../../../glossary.md#mean-squared-error--mse) `NONE` with one lane | Every other loss form stays scalar |
| [Pooling](../../../../glossary.md#term-pooling) and attention | scalar | No production vector threshold |
| Reductions/[normalization](../../../../glossary.md#term-normalization) | scalar whole-cell; BatchNorm [channel](../../../../glossary.md#term-channel) range at parallelism `1` | Partial recipe unreachable without trusted evidence |
| Movement/indexing/scatter/fold/order/random/[scan](../../../../glossary.md#term-cumulative-scan)/affine | scalar | No production vector threshold |
| [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) | scalar direct for zero work, [matrix](../../../../glossary.md#term-matrix-meanings)-vector, or fewer than 4,096 contributions; scalar 2x2 at 16,384; explicit vector candidates can use direct-N or 2x2 at 65,536 | Complete route facts agree or [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) |
| [OpenBLAS](../../../../glossary.md#term-openblas) | optional qualified single-thread FLOAT32/FLOAT64 MATMUL | Portable cost `1000 + 10*outputs + 100*macs`; native `1 + outputs + macs`; margins `1`; portable fallback |
| [Materialization](../../../../glossary.md#materialization) | disabled | No speculative production copy |
| [Fusion](../../../../glossary.md#term-fusion) | margin `32`; profitable code/live ceilings `48`/`12`; decision facts `384` | [Canonical](../../../../glossary.md#term-canonical-form) split on uncertainty/tie |
| [Decomposition](../../../../glossary.md#term-decomposition) | [nodes](../../../../glossary.md#node) `8`, attempts `28`, boundaries `16`, live `16`, indexing `32`, code `64`, candidates `64`, enumeration attempts `256` | Bounded fail closed |
| Recognition | attempts `24`, facts `8`, members `6`, boundaries `10`, units `2`, epilogue ops `2` | Reject over-budget recognition |
| Representation | eligible sources `8`, copy choices `36`, variants `37` | Production materialization remains disabled |
| [Specialization](../../../../glossary.md#term-specialization) | candidates `4`, [artifacts](../../../../glossary.md#term-artifact) `1`, fixed/[unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) variants `0` | No [Runtime](../../../../glossary.md#term-runtime) specialization |
| Preparation [identity](../../../../glossary.md#term-identity-meanings) | [manifest](../../../../glossary.md#term-manifest)/[persistence](../../../../glossary.md#term-persistence) off; derived exact segment carriers; exact numerics; unspecified hardware; ordinary cohort | Typed explicit inputs only |

No row changes. A proposed global `VECTOR_IF_ELIGIBLE` escalation was reverted after independent
review showed that its workload matrix covered only dense FLOAT32 ADD and convolution while the
policy also enabled MATMUL, MSE, floating transcendental, integral, BOOL/[mask](../../../../glossary.md#term-boolean-mask), broadcast, and [fused](../../../../glossary.md#specialization-and-fusion)
pointwise forms.

## Invalidated attempt and historical search

The original 13 [scalar](../../../../glossary.md#term-scalar)/[vector](../../../../glossary.md#term-vector)/parallel files were removed. They omitted source, command, [CPU](../../../../glossary.md#term-cpu),
species, and selected-plan [identities](../../../../glossary.md#term-identity-meanings); ran all scalar processes before all vector processes; pooled
within-[fork](../../../../glossary.md#term-fork-meanings) [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) as independent observations; and had 81 of 420 repeated [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) below the
claimed 25 ms floor. Their aggregates are not retained or cited.

Targeted [sealed](../../../../glossary.md#term-sealed)-[artifact](../../../../glossary.md#term-artifact) search found generated-versus-direct parity and machine-local macOS
[AArch64](../../../../glossary.md#term-aarch64) observations, not compatible scalar-versus-vector evidence spanning the current policy
[matrix](../../../../glossary.md#term-matrix-meanings) or [x86-64](../../../../glossary.md#term-x86-64). Generated-versus-direct parity is not permission to change scalar/vector policy.
A universal default therefore has no adequate cross-[target](../../../../glossary.md#term-target) basis.

## Hardened report protocol

[Schema](../../../../glossary.md#term-schema) 3 evidence runs require exact base revision, Git tree object format/OID, configuration, and
[SHA-256](../../../../glossary.md#term-sha-256) harness/production [hashes](../../../../glossary.md#term-hash); [fork](../../../../glossary.md#term-fork-meanings)/order facts; [CPU](../../../../glossary.md#term-cpu) [identity](../../../../glossary.md#term-identity-meanings)/[features](../../../../glossary.md#term-software-feature); portable values;
worker/[materialization](../../../../glossary.md#materialization)/[OpenBLAS](../../../../glossary.md#term-openblas) policy; and `-Xms1g -Xmx1g -XX:-TieredCompilation -Xbatch`.
[Gradle](../../../../glossary.md#term-gradle) normalizes the profile with `Locale.ROOT`, so mixed-case `EVIDENCE` receives the same flags
and requirements.

Each retained [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) accumulates timed executions until its own elapsed time reaches 25 ms; no
execution, sample, fork, retry, or [outlier](../../../../glossary.md#term-outlier) is discarded. Reports retain per-sample iteration counts,
[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) durations, normalized times, summaries, one-shot observations, checksums, preferred species,
VM [arguments](../../../../glossary.md#term-argument), and a [hash](../../../../glossary.md#term-hash) of loaded harness class bytes.

Public [Engine](../../../../glossary.md#term-engine) deliberately hides [backend](../../../../glossary.md#backend)-private selected-plan facts. Schema 3 reports record that
limitation and set `eligibleForProductionDecision=false`; they cannot authorize a route/default
change. The [manifest](../../../../glossary.md#term-manifest)'s minimum future acceptance framework requires a later comparison to fail
when actual route, strategy, species, ranges, workers, materializations, or OpenBLAS selection are
unavailable.

`backends/cpu/evidence/cpu-0010m-defaults/manifest.json` retains the exact [scalar](../../../../glossary.md#term-scalar)-baseline commands
and hashes. Its future section is not a measurement-[ready](../../../../glossary.md#term-ready) protocol: a separately reviewed task
must seal exact CPU/OS/[JDK](../../../../glossary.md#term-java-development-kit-jdk)/species strata; every [operation](../../../../glossary.md#operation)/[topology](../../../../glossary.md#term-topology)/type/[shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/access/[carrier](../../../../glossary.md#term-storage-carrier)/
data case; randomized interleaved order and [seed](../../../../glossary.md#term-seed); family aggregation and weighting; and the exact
bootstrap resample count, [PRNG](../../../../glossary.md#graph-rng-state--graphrngstate), sidedness, and quantile convention before either configuration is
measured. The retained framework supplies minimum floor, fork-unit, correctness, [regression](../../../../glossary.md#term-regression),
improvement, policy-[scope](../../../../glossary.md#term-scope), and selected-plan requirements only.

## Regenerated [scalar](../../../../glossary.md#term-scalar) baseline

Five fresh JVMs validate the hardened protocol against unchanged scalar production on Apple M3
Max `Mac15,9`, macOS 26.6.2 [AArch64](../../../../glossary.md#term-aarch64), OpenJDK 26.0.1+8-34, and preferred 128-bit species. All 210
retained repeated [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) exceed 25 ms; minimum is 25,170,042 ns. Every within/across-[fork](../../../../glossary.md#term-fork-meanings) [checksum](../../../../glossary.md#term-checksum)
is stable. [Median](../../../../glossary.md#term-median)-of-fork-medians in nanoseconds is: [pointwise](../../../../glossary.md#term-pointwise) `12,951,333`, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) `266,271`,
[Conv2d](../../../../glossary.md#convolution--conv2d) `28,719,791`, [Conv3d](../../../../glossary.md#conv3d--ncdhw) `37,864,125`, reduction `15,876,833`, [normalization](../../../../glossary.md#term-normalization) `29,217,000`.

These values are a machine-local scalar baseline and harness proof, not a candidate comparison,
speedup claim, selected-route proof, or production-decision [artifact](../../../../glossary.md#term-artifact).

## Acceptance criteria

- Production composition remains [scalar](../../../../glossary.md#term-scalar)/single-threaded with every existing fallback and threshold.
- Evidence profile spelling cannot bypass evidence [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) flags or required [metadata](../../../../glossary.md#term-metadata).
- Every retained repeated batch meets its floor; reports retain exact source/config/environment and
  explicitly refuse production-decision eligibility without actual selected-plan facts.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Invalid evidence and aggregates are absent; the manifest is hash-valid and explicitly prevents
  future measurement until a reviewed task seals every currently open protocol choice.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [hash glossary](../../../../glossary.md#term-hash))</span>
- CPU/benchmark tests, builds, Javadocs, smoke/evidence execution, evidence/docs validation,
  repository-wide tests, and fresh independent Class C review pass before status becomes Complete.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [smoke glossary](../../../../glossary.md#term-smoke-test); [validation glossary](../../../../glossary.md#term-validation))</span>

## [Validation](../../../../glossary.md#term-validation) and result

Completed worker checks after review remediation:

- Affected [CPU](../../../../glossary.md#term-cpu)/benchmark tests, builds, and Javadocs — passed via `:backends:cpu:test`,
  `:backends:cpu:javadoc`, `:tools:benchmarks:build`, and `:tools:benchmarks:javadoc`.
- Final `-Pprofile=EVIDENCE` lifecycle execution — passed with evidence JVM flags and all six
  workloads, proving case-normalized profile handling.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- Five scalar evidence JVMs — passed; 210/210 retained batches meet the floor and all checksums are
  stable. A deterministic integrity check rehashed every report/source artifact and verified exact
  metadata, JVM flags, orders, production-decision ineligibility, checksums, and aggregate facts.<span data-glossary-navigation="true"> ([batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [integrity glossary](../../../../glossary.md#term-integrity); [artifact glossary](../../../../glossary.md#term-artifact); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Ten changed Markdown files passed local-link, fence, final-newline, and trailing-whitespace
  validation; `git diff --check` passed.
- Narrow independent Class C re-review — approved after verifying honest Git object naming, exact
  order vocabulary, and the non-measurement-ready future acceptance framework.<span data-glossary-navigation="true"> ([ready glossary](../../../../glossary.md#term-ready))</span>
- Final unchanged-worktree `./gradlew test` — `BUILD SUCCESSFUL`; 72 tasks: 3 executed, 37 from
  cache, and 32 up-to-date.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Final unchanged-worktree `./gradlew build` — `BUILD SUCCESSFUL`; 84 tasks: 11 executed and 73
  up-to-date.
- `git diff --check` passed before and after the repository-wide validation; worktree status was
  identical before and after.

All acceptance criteria passed. The evidence-backed conclusion is no production-default change:
ordinary CPU execution remains [scalar](../../../../glossary.md#term-scalar) with configured/available parallelism `1`/`1`, minimum
[elements](../../../../glossary.md#term-element) per worker `1`, no worker group, and every existing fallback and threshold retained. The
benchmark protocol is hardened; any future candidate comparison still requires a separately
reviewed, fully [sealed](../../../../glossary.md#term-sealed) [matrix](../../../../glossary.md#term-matrix-meanings) before measurement.
