# [CPU](../../../../glossary.md#term-cpu) Task 0008P: Deterministic Modular Partial-Reduction Parallelism

## Status

Complete

## Goal

Add one [CPU](../../../../glossary.md#term-cpu)-private, prepared partial-reduction execution architecture for a large ordinary integral `SUM` or `PROD` domain. It splits a domain into fixed disjoint partial ranges, retains one modular state for every partial, combines states in fixed ordinal order, and publishes only after a successful deterministic-result combine. This is a new partial-body/partial-state/combine realization, not an extension of the current output-cell range emitter.

## [Scope](../../../../glossary.md#term-scope)

Implement exactly this first admitted realization:

| Fact | Admitted value |
|---|---|
| [Model](../../../../glossary.md#term-model) family | ordinary `AggregateReductionKind.SUM` and `PROD` only |
| [CPU](../../../../glossary.md#term-cpu) [IR](../../../../glossary.md#term-intermediate-representation-ir) | one `CpuAggregateIr` ordinary `FULL`, `SINGLE_AXIS`, or `MULTI_AXIS`; never `SUM_TO_SHAPE` |
| represented/result type | same-typed [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype) only |
| [layout](../../../../glossary.md#layout)/carriers | fully static resolved dense-[linear](../../../../glossary.md#term-linear-meanings) input and [injective](../../../../glossary.md#term-injective) dense-linear output; primitive [heap](../../../../glossary.md#term-heap) `int[]`/`long[]` boundaries only |
| [topology](../../../../glossary.md#term-topology) | one direct aggregate computation unit, one input and one output, no [fusion](../../../../glossary.md#term-fusion)/[materialization](../../../../glossary.md#materialization) |
| modes | existing [scalar](../../../../glossary.md#term-scalar) and caller-parallel complete-output-cell modes, plus selected `PARTIAL_REDUCTION_PARALLEL` only |
| partial eligibility | non-empty domain, at least one output cell, `P` exactly 2 or 4 (`4` is the maximum admitted partial count), every partial meets the existing `CpuPartitionAnalysisInputs.minimumElementsPerWorker()` fact, checked workspace geometry fits, and the matching frozen performance row passed |

The selector is cold, deterministic, and [fail-closed](../../../../glossary.md#term-fail-closed). Every zero-domain [identity](../../../../glossary.md#term-identity-meanings), zero-output [Shape](../../../../glossary.md#shape), small domain, non-dense access, segment/mixed [carrier](../../../../glossary.md#term-storage-carrier), [BFLOAT16](../../../../glossary.md#term-bfloat16), or unlisted [occurrence](../../../../glossary.md#term-occurrence) keeps the current whole-cell path. A one-cell output is eligible when its selected domain supplies the required partial work.

For output cell `c` and prepared partial ordinal `p`, fixed quotient/remainder partitioning creates one non-empty selected-domain interval `[begin(c,p), end(c,p))`. The worker phase invokes the generated partial body once per `(c,p)` and writes only `state(c,p)`. After one [synchronous](../../../../glossary.md#term-synchronous) worker-group join, the invoking thread runs the generated combine body for output cells in ascending logical order, consumes partial states in ascending `p`, writes the output, then follows normal [publication](../../../../glossary.md#term-publication). No partial body writes output; no atomics, changing state ordinal, or early publication exists.

Use one same-width [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) modular state: `int`/`iadd` or `imul` for `INT32`, and `long`/`ladd` or `lmul` for `INT64`. Each partial begins with the existing identity (zero for `SUM`, one for `PROD`); combine uses the same [operation](../../../../glossary.md#operation), so every selected value is included once and [overflow](../../../../glossary.md#term-overflow) wraps modulo `2^32` or `2^64`. No floating state, [rounding](../../../../glossary.md#term-rounding), [tolerance](../../../../glossary.md#term-tolerance), or special-value claim exists. The frozen optimal clean-Java [oracle](../../../../glossary.md#term-test-oracle) must use the same [partition](../../../../glossary.md#partition) formula, state transition, combine order, and direct array [signatures](../../../../glossary.md#term-signature). It is test/performance-only and unreachable from production execution.

### Source-backed eligibility boundary

[Model](../../../../glossary.md#term-model) 0018V permits floating [reassociation](../../../../glossary.md#term-reassociation) or parallelization only when observable results satisfy a *future* [conformance](../../../../glossary.md#term-conformance) [tolerance](../../../../glossary.md#term-tolerance); it explicitly does not define that tolerance. [CPU](../../../../glossary.md#term-cpu) 0008O likewise stopped floating reassociation where permission was not current. The current exact floating state cannot manufacture that missing conformance permission, even if it can represent an exact sum before one [rounding](../../../../glossary.md#term-rounding). Model 0018U1 independently and currently permits exact-type modular reassociation for integral ordinary `SUM`/`PROD`. That is semantic permission, not CPU [admission](../../../../glossary.md#term-admission).

CPU 0008P consumes only ordinary [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) `SUM`/`PROD` above. It does not claim a general `CpuAggregateIr` is partial-capable. Floating `SUM`/[`MEAN`](../../../../glossary.md#term-mean)/`PROD`, [extrema](../../../../glossary.md#saturation-pole-and-extremum), [BOOL](../../../../glossary.md#data-type--datatype), masked reductions, `SUM_TO_SHAPE`, and advanced `LOG_SUM_EXP`/[variance](../../../../glossary.md#term-variance)/standard-deviation/[L1](../../../../glossary.md#term-l1)/[L2](../../../../glossary.md#term-l2) remain unchanged future candidates requiring their own Model eligibility and state/combine proof. Cumulative scans remain whole-[slice](../../../../glossary.md#slice): prefix order is observable. [Softmax](../../../../glossary.md#softmax--log-softmax)/[log-softmax](../../../../glossary.md#softmax--log-softmax), attention rows, Layer/RMS and [batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training), [MSE](../../../../glossary.md#mean-squared-error--mse)/categorical [losses](../../../../glossary.md#loss), and their stable/multi-pass reductions remain caller-parallel only. CPU 0008O's cancelled candidate-2 `KEEP_SCALAR` evidence grants no [SIMD](../../../../glossary.md#simd), [lane](../../../../glossary.md#term-lane)-tree, or floating-reassociation permission.

The result is deterministic because modular `SUM`/`PROD` reassociation produces the same same-width bit pattern for a fixed input, regardless of the frozen `P` or worker count. The prepared range formula and ordinal combine make the recipe deterministic; they do not promise a deterministic worker schedule, worker assignment, timing, or cross-[backend](../../../../glossary.md#backend) execution behavior.

## Out of [scope](../../../../glossary.md#term-scope)

- [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend)-contract, public [API](../../../../glossary.md#term-application-programming-interface-api), architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle), [conformance](../../../../glossary.md#term-conformance), or integration change.
- [SIMD](../../../../glossary.md#simd)/[vector](../../../../glossary.md#term-vector) partial bodies, [lane](../../../../glossary.md#term-lane)/tree reduction, floating [tolerance](../../../../glossary.md#term-tolerance) or [reassociation](../../../../glossary.md#term-reassociation), a deterministic-schedule/timing/cross-backend promise, or use of 0008O as permission.
- Partial paths for scans, floating/masked/advanced/statistical/[norm](../../../../glossary.md#term-norm) reductions, [softmax](../../../../glossary.md#softmax--log-softmax), [normalization](../../../../glossary.md#term-normalization), attention, [losses](../../../../glossary.md#loss), matmul/[convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[pooling](../../../../glossary.md#term-pooling), [BOOL](../../../../glossary.md#data-type--datatype)/[extrema](../../../../glossary.md#saturation-pole-and-extremum) aggregates, integral [`MEAN`](../../../../glossary.md#term-mean), or `SUM_TO_SHAPE`.
- Segments, mixed carriers, strided/zero-strided/broadcast [layouts](../../../../glossary.md#layout), dynamic [Shapes](../../../../glossary.md#shape), [fusion](../../../../glossary.md#term-fusion), [materialization](../../../../glossary.md#materialization), nested submission, per-worker allocation, persistent/thread-local scratch, atomics, locks in generated loops, or Runtime selection.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, including staged preparation, [RunState](../../../../glossary.md#run-state--runstate), workspace [ownership](../../../../glossary.md#term-ownership-meanings), and [runtime](../../../../glossary.md#term-runtime) hot-path rules.
- [Current architecture navigation](../../../../architecture/current-architecture-plan.md), [planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>, and [CPU master plan](../master-plan.md).
- [CPU](../../../../glossary.md#term-cpu) [ordinary aggregates](0007a1-portable-ordinary-numerical-aggregate-reductions.md), [masked reductions](0007c-portable-masked-reduction-coverage.md), [advanced reductions](0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md), [scans](0007-portable-cumulative-scan-coverage.md), [softmax](0007e-portable-stable-softmax-and-log-softmax-coverage.md)<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>, [normalization](0007f-portable-layer-and-rms-normalization-coverage.md)<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>, [losses](0008i-portable-loss-family-execution.md)<span data-glossary-navigation="true"> ([losses glossary](../../../../glossary.md#loss))</span>, [attention](0008h-portable-scaled-dot-product-attention-execution.md), and cancelled [0008O](0008o-stable-reduction-vector-numerical-spike.md).
- [Model](../../../../glossary.md#term-model) [integral aggregate](../../../modules/model/tasks/0018u1-integral-reductions-and-arg-min-normalization.md), [multi-axis/statistical](../../../modules/model/tasks/0018v-multi-axis-and-statistical-reductions.md)<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>, [scan](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>, [softmax](../../../modules/model/tasks/0016j-softmax-tensor-expressions.md), [attention](../../../modules/model/tasks/0019e-scaled-dot-product-attention.md), [normalization](../../../modules/model/tasks/0021-layer-normalization-semantics-and-tensor-expressions.md), [batch normalization](../../../modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md)<span data-glossary-navigation="true"> ([batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training))</span>, and [loss](../../../modules/model/tasks/0022-mean-squared-error-loss.md).

## Architecture constraints

- [Model](../../../../glossary.md#term-model) owns mathematics and [reassociation](../../../../glossary.md#term-reassociation) permission; [CPU](../../../../glossary.md#term-cpu) owns truthful [lowering](../../../../glossary.md#lowering), cold selection, generated realization, workspace, and execution. Never manufacture floating permission or a [tolerance](../../../../glossary.md#term-tolerance).
- Analysis declares one exact workspace before shared assignment. Finalization uses that slot without changing route, partial count, or size. [Runtime](../../../../glossary.md#term-runtime) only binds/[releases](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) [run](../../../../glossary.md#term-run)-owned workspace and executes the prepared recipe.
- [Prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) remains [immutable](../../../../glossary.md#term-immutable)/reusable; each [RunState](../../../../glossary.md#run-state--runstate) gets distinct workspace. Inputs are [borrowed](../../../../glossary.md#ownership-borrowing-and-lease), workspace is run-owned, and failure cleanup releases only still-run-owned resources.
- Freeze partial count `P`, range formula, output order, partial order, state [layout](../../../../glossary.md#layout), and combine algorithm in preparation. `P` is exactly `2` or `4`, and `4` is the maximum admitted count; it is also at most worker capacity and valid partial-work count. The recipe fixes result construction, but does not fix worker scheduling; modular semantics make the admitted result bit-identical across valid `P` and worker counts.
- Borrow [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) once for one flat indexed partial-work list. No worker submits, closes, or waits on it. Existing failure ordering, [cancellation](../../../../glossary.md#term-cancellation), interruption, accessibility, and join rules remain authoritative. Failure prevents combine/[publication](../../../../glossary.md#term-publication).
- Let `C` be output cells, `P` partial count, and `S = 8` be one aligned state [slice](../../../../glossary.md#slice): the [`INT32`](../../../../glossary.md#data-type--datatype) [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) occupies its low 4 bytes and [`INT64`](../../../../glossary.md#data-type--datatype) its 8-byte payload. Declare `alignUpExact(C * P * S, 8)` with checked multiplication; no partial workspace unless this route is selected. Each aligned `state(c,p)` is non-overlapping and lives from [binding](../../../../glossary.md#term-binding) through combine completion. Validate span/[alignment](../../../../glossary.md#term-alignment)/accessibility and workspace/input/output non-overlap before submission.

## Package impact

Existing packages used:

- `...backend.cpu.internal.ir` — separate partial-reduction [IR](../../../../glossary.md#term-intermediate-representation-ir) and aggregate facts; do not overload whole-cell range meaning.
- `...internal.lowering` — eligibility and cold geometry.
- `...internal.codegen.emit` — dedicated typed partial/combine emitters.
- `...internal.prepare` and `...internal.executable` — staged resource declaration, finalization, [binding](../../../../glossary.md#term-binding), orchestration, and [publication](../../../../glossary.md#term-publication).
- `...internal.reference` — frozen test-only clean-Java [oracle](../../../../glossary.md#term-test-oracle).

Packages added or changed: none. Expected placement is `CpuPartialReductionIr`, its [lowering](../../../../glossary.md#lowering) geometry, and dedicated emitter/reference test types in those packages; refine names only by updating this task and the master-plan package map before implementation.

## Affected files

Expected implementation paths (at most 18 [CPU](../../../../glossary.md#term-cpu) source/test paths) include aggregate [lowering](../../../../glossary.md#lowering),
[partition](../../../../glossary.md#partition) lowering/[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag)/preparer/finalizer [handoff](../../../../glossary.md#term-handoff), [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable), generated
emitter/generator, workspace [binding](../../../../glossary.md#term-binding), a new partial [IR](../../../../glossary.md#term-intermediate-representation-ir)/emitter/reference, and focused tests
mirroring their [owners](../../../../glossary.md#term-owner). `CpuAggregateIr` continues to mean a complete output-cell range; the
separate partial IR and its generated partial/combine route therefore cannot truthfully be folded
into the existing ordinary-emitter path merely to meet a smaller file count.

The exactly seven permitted focused CPU test paths are:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPartialReductionGeneratedKernelTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPartialReductionExecutionTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPartialReductionIrTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartialReductionLoweringTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`; and
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`.

The last path is mandatory rather than discretionary: its exact authorized-production-file set
must name each of this task's four new internal production types for the CPU [module](../../../../glossary.md#term-module) test suite to
admit them.

[Planning](../../../../glossary.md#term-planning) paths are exactly:

- this task;
- `docs/planning/backends/cpu/master-plan.md`; and
- `docs/planning/roadmap.md`.

The implementation evidence root is one caller-supplied, initially empty directory outside the
repository, named `synaptik-cpu-0008p-<run-id>`. It is not a repository path and counts as zero of
the 20-path repository budget; its required files are nevertheless part of the acceptance
inventory below. No evidence [artifact](../../../../glossary.md#term-artifact) may be silently added under a source, test, or planning
directory. If retained evidence must be tracked in the repository, stop and re-plan the path
budget before writing it.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 21 paths: these three [planning](../../../../glossary.md#term-planning) paths, eleven production [CPU](../../../../glossary.md#term-cpu) paths, and exactly seven
focused CPU test paths. The eleven production paths are the minimum coherent current-[topology](../../../../glossary.md#term-topology)
budget: the separate `CpuPartialReductionIr` and generated partial/combine route require their
own path in addition to the existing cold [lowering](../../../../glossary.md#lowering), preparation/finalization, and execution
[handoff](../../../../glossary.md#term-handoff) [owners](../../../../glossary.md#term-owner). The seventh test path is the mandatory CPU internal-package inventory update;
without it, the full CPU [validation](../../../../glossary.md#term-validation) rejects the four required new production types. This is the
minimum coherent correction and does not broaden implementation, semantic, or architectural
scope. If a shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) contract, second workspace type, [Model](../../../../glossary.md#term-model) change, another family,
or a 22nd path is needed, stop and create a separately scoped follow-up.

## Acceptance criteria

1. A separate partial-reduction [IR](../../../../glossary.md#term-intermediate-representation-ir) expresses partial body/state/combine/fixed [partition](../../../../glossary.md#partition)/final [publication](../../../../glossary.md#term-publication); `CpuAggregateIr` retains complete output-cell range meaning.
2. Lowering selects it only for the exact scope table and the complete passing profitability
   evidence for the matching form and <span data-glossary-navigation="true"> ([Lowering glossary](../../../../glossary.md#lowering); [scope glossary](../../../../glossary.md#term-scope))</span>`P` (`P=2` or `P=4`, the prepared maximum in the four-worker
   admission configuration); all exclusions, overflow, absent worker proof, or incomplete/failed
   evidence retain scalar/caller-parallel whole-cell execution.
3. Analysis/finalization declares/validates exact aligned `C * P * S` workspace; partials own disjoint state, combine owns writes, publication follows successful ordered combines.
4. Generated partial/combine and frozen clean-Java [oracle](../../../../glossary.md#term-test-oracle) are equivalent direct typed algorithms. Generated/decompiled inspection proves no helper/[dispatch](../../../../glossary.md#term-dispatch), allocation, boxing, reflection, [carrier](../../../../glossary.md#term-storage-carrier) [lookup](../../../../glossary.md#term-lookup), route selection, or worker submission in hot loops.
5. Focused semantics cover both [identities](../../../../glossary.md#term-identity-meanings), modular [overflow](../../../../glossary.md#term-overflow), negative operands, empty/zero output fallback, full/single/multi forms, P=2/maximum P, coverage/no overlap, [scalar](../../../../glossary.md#term-scalar)/whole-cell/partial bit [identity](../../../../glossary.md#term-identity-meanings) across valid `P` and worker counts, replay, [cancellation](../../../../glossary.md#term-cancellation)/failure/no output publication, interruption, and concurrent [RunState](../../../../glossary.md#run-state--runstate) isolation.
6. A named [finite](../../../../glossary.md#term-finite) generated-identity [matrix](../../../../glossary.md#term-matrix-meanings) covers both types and kinds, forms, dense offsets where proof permits, `P=2`/maximum, scalar/whole-cell/partial strategy, and all admitted array [signatures](../../../../glossary.md#term-signature). The timed [admission](../../../../glossary.md#term-admission) matrix and its five-[fork](../../../../glossary.md#term-fork-meanings) protocol below are frozen before the first fork; it is the only profitability evidence permitted for production selection.
7. A separate documentation-focused agent finalizes Javadocs, [CPU](../../../../glossary.md#term-cpu) guide/glossary impact, task/master/roadmap, and reasoned no-change conclusions in the same change.

### Frozen partial-reduction profitability protocol

This protocol answers one narrow question: whether the exact generated primitive-array partial
body plus ordinal combine is profitable enough to replace the existing generated complete-output-
cell route for the covered prepared facts. It does not add semantic coverage; ordinary focused
tests own [identities](../../../../glossary.md#term-identity-meanings), [overflow](../../../../glossary.md#term-overflow), [cancellation](../../../../glossary.md#term-cancellation), range coverage, `P` boundaries, and [RunState](../../../../glossary.md#run-state--runstate)
isolation.

Before timing, encode and assert this exact 24-row inventory. `C` is output-cell count and `D` is
selected-domain [elements](../../../../glossary.md#term-element) per output cell. Every row has one fully static dense `int[]` or `long[]`
input and [injective](../../../../glossary.md#term-injective) dense output, zero base offsets, at least four available worker slots, and
both fixed partial counts shown. The [test fixture](../../../../glossary.md#adr-ci-and-qualification) must also derive a positive-offset [identity](../../../../glossary.md#term-identity-meanings) row
for every type/kind/form outside timing; offset is not a timed profitability variable.

| Form | Input [shape](../../../../glossary.md#shape) and [axes](../../../../glossary.md#term-axis) | `C` | `D` | `P` values |
|---|---|---:|---:|---|
| `FULL` | `[524288]`, all axes | 1 | 524288 | 2, 4 |
| `SINGLE_AXIS` | `[64, 8192]`, axis `1` | 64 | 8192 | 2, 4 |
| `MULTI_AXIS` | `[4, 16, 2048]`, axes `1,2` | 4 | 32768 | 2, 4 |

```text
2 types (INT32, INT64) * 2 kinds (SUM, PROD) * 3 forms * 2 P values = 24 rows
```

For each row, the harness uses four independently named, shape-polymorphic implementations with
identical typed array [signatures](../../../../glossary.md#term-signature), input values, output geometry, fixed range formula, `P`, result
consumption, and cold facts:

- `Wg`: the current generated whole-cell route. It is the production baseline.
- `Dw`: frozen optimal clean Java for that same whole-cell algorithm. `Wg/Dw` proves that the
  baseline itself has not acquired generated overhead which could make an apparent partial gain
  meaningless.
- `Gp`: the candidate generated partial-body plus generated ordered-combine route. It is the only
  implementation whose selection is being considered.
- `Dp`: frozen optimal clean Java for the same partial ranges, modular state transition, ordinal
  combine, and direct array signatures as `Gp`. `Gp/Dp` proves generated partial/combine parity.

`Gp/Wg` proves profitability of partial execution including combine and worker-group overhead,
not merely an isolated body. `Gp/Dp <= 1.15x` and `Wg/Dw <= 1.15x` are generated-versus-optimal-
Java parity controls; `Gp/Wg <= 0.90x` requires at least a ten-percent end-to-end gain. Ten percent
is deliberately above ordinary [run](../../../../glossary.md#term-run)-to-run noise and compensates for the route's workspace and
join complexity; 15% permits normal generated-class versus `javac` variation but rejects a hidden
algorithmic or [dispatch](../../../../glossary.md#term-dispatch) penalty. Every individual retained pair, every per-[fork](../../../../glossary.md#term-fork-meanings) [median](../../../../glossary.md#term-median), and the
median of five fork medians must meet its applicable gate. A missing full `P=4` worker proof, a
failed parity gate, or any missed profitability gate is `KEEP_WHOLE_CELL` for that exact row.
For one symmetric pair, its ratio is the sum of the two `A` side durations divided by the sum of
the two `B` side durations in execution order; the fork [statistic](../../../../glossary.md#term-statistic) is the median of its nine pair
ratios, and the row aggregate is the median of its five fork medians. No [mean](../../../../glossary.md#term-mean), best [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), or
cross-row aggregate participates in [admission](../../../../glossary.md#term-admission).

Use exactly five fresh Java 26 child [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [forks](../../../../glossary.md#term-jvm-fork) per row, with
`-Xms1g -Xmx1g -XX:-TieredCompilation -Xbatch`. Each fork runs five randomized symmetric [warmup](../../../../glossary.md#term-warmup)
pairs for each of `Gp/Wg`, `Gp/Dp`, and `Wg/Dw`; each pair is either `A-B-B-A` or `B-A-A-B`.
After warmup, double one shared iteration count until two invocations of each of `Wg`, `Dw`, `Gp`,
and `Dp` take at least 50 ms. Retain exactly nine randomized symmetric pairs for each comparison;
all four timed sides of every pair must individually take at least 25 ms. Thus the [immutable](../../../../glossary.md#term-immutable)
inventory is:

```text
24 rows * 5 forks * 9 pairs * 3 comparisons = 3,240 retained pairs
3,240 pairs * 4 timed sides = 12,960 timings
24 rows * 5 forks * 3 comparisons = 360 fork medians
24 rows * 3 comparisons = 72 median-of-fork-medians aggregates
```

The order generator is [SplitMix64](../../../../glossary.md#graph-rng-state--graphrngstate) with base [seed](../../../../glossary.md#term-seed) `0x0000000000080050L`; derive the row seed from
the [canonical](../../../../glossary.md#term-canonical-form) row identifier and the fork seed from `(row seed, fork ordinal)`. Generate input
values from that seed with a fixed documented mapping to the signed values `[-3, 3]`; record the
[SHA-256](../../../../glossary.md#term-sha-256) of each canonical [little-endian](../../../../glossary.md#text-encoding-and-file-boundaries) typed input byte sequence and of every consumed output
sequence. The implementation source [snapshot](../../../../glossary.md#term-snapshot), generated class bytes, compiled `Dp`/`Dw` class
bytes, and exact command lines are hashed before fork zero. The [fixture](../../../../glossary.md#term-fixture) must reject a changed
seed, class/source [hash](../../../../glossary.md#term-hash), row inventory, JVM flags, gates, calibration rule, sample count, or input
hash before it launches a timed fork.

The evidence root contains immutable `protocol.json`, append-only `forks/row-<id>-fork-<0..4>.csv`,
`environment.json`, `classes.json`, `inputs.json`, `summary.json`, and `SHA256SUMS`. `protocol.json`
has [schema](../../../../glossary.md#term-schema) `synaptik.cpu.partial-reduction-performance.v1` and records the 24 canonical row IDs,
comparison definitions, gates, warmup/measured counts, flags, seed derivation, and file schemas.
Each CSV has one row per timed side with `row_id,fork,comparison,pair,side,order,iterations,ns,
input_sha256,output_sha256`; it is written only in prescribed pair/side order. `environment.json`
records OS/version/architecture, [CPU](../../../../glossary.md#term-cpu) [model](../../../../glossary.md#term-model) and logical processors, [JDK](../../../../glossary.md#term-java-development-kit-jdk) vendor/version/build,
JVM flags, [heap](../../../../glossary.md#term-heap), available workers, affinity/governor facts when available (otherwise explicit
`unavailable`), and wall-clock timestamps. `classes.json` records all source/class hashes and
direct [descriptors](../../../../glossary.md#term-descriptor); `inputs.json` records canonical geometry, `P`, seed, mapping, and hashes;
`summary.json` records every pair ratio, fork median, aggregate, gate result, and row decision.
`SHA256SUMS` covers every completed file. The harness verifies an existing root before append and
refuses changed or missing prior hashes.

There is one one-shot run: rows/forks execute in canonical row then fork order; no retry, discard,
replacement, [outlier](../../../../glossary.md#term-outlier) filtering, rerun-to-pass, fixture change, post-hoc batching, threshold change,
or partial-fork substitution is allowed. An interruption, timing-floor failure, calibration failure,
missing [metadata](../../../../glossary.md#term-metadata)/hash, corrupted [artifact](../../../../glossary.md#term-artifact), or 30-minute whole-protocol wall-clock limit records
the precise failure and makes every unfinished or unverifiable row `KEEP_WHOLE_CELL`. Results from
another root never supplement this run. Production admission is [fail-closed](../../../../glossary.md#term-fail-closed): only a complete,
hash-valid root in which all 24 rows pass every stated gate permits this route's selector for the
exact covered [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) `SUM`/`PROD`, form, and `P` facts. Otherwise the selector must keep
the existing whole-cell route; it may not extrapolate one row's pass to another form, type, kind,
or partial count.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused [CPU](../../../../glossary.md#term-cpu) [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/emitter/executable tests, then:

```bash
./gradlew :backends:cpu:test
./gradlew :backends:cpu:javadoc
git diff --check
```

Add generated Class-File/[decompilation](../../../../glossary.md#term-decompilation) and import/manual scans, checked resource/range [matrix](../../../../glossary.md#term-matrix-meanings),
and the frozen 24-row/5-[fork](../../../../glossary.md#term-fork-meanings) evidence protocol above. The performance command must require the
empty external evidence root and fail before timing if its protocol seal cannot be created.
Repository-wide validation is deferred to CPU 0009/[CI](../../../../glossary.md#term-continuous-integration-ci) because no shared contract changes.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0007A1 ordinary aggregate modular [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) route and its generated-[operation](../../../../glossary.md#operation) conventions.
- CPU 0008B [partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) finalization and existing [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) [lifecycle](../../../../glossary.md#lifecycle).
- CPU 0008O cancelled [scalar](../../../../glossary.md#term-scalar) evidence, only as [SIMD](../../../../glossary.md#simd) prohibition.
- Current [Model](../../../../glossary.md#term-model) 0018U1 modular-[reassociation](../../../../glossary.md#term-reassociation) contract; Model 0018V is a floating non-eligibility boundary, not a [dependency](../../../../glossary.md#term-dependency) granting permission.

## Follow-up tasks

- Another family needs its own [Model](../../../../glossary.md#term-model) eligibility, state/combine, resource, [publication](../../../../glossary.md#term-publication), and [oracle](../../../../glossary.md#term-test-oracle) proof.
- [CPU](../../../../glossary.md#term-cpu) 0009 remains the later generated-coverage [checkpoint](../../../../glossary.md#term-checkpoint); do not detail it here.

## Architecture impact

Expected impact: None. The architecture already authorizes [backend](../../../../glossary.md#backend)-private [lowering](../../../../glossary.md#lowering)/route selection, staged exact workspace declaration, [immutable](../../../../glossary.md#term-immutable) prepared recipes, isolated [RunState](../../../../glossary.md#run-state--runstate) resources, and prepared [Runtime](../../../../glossary.md#term-runtime) schedules. No [ADR](../../../../glossary.md#term-architecture-decision-record-adr) or [architecture test](../../../../glossary.md#term-architecture-test) is expected because no [dependency](../../../../glossary.md#term-dependency)/[ownership](../../../../glossary.md#term-ownership-meanings) rule changes. If implementation needs a shared partial-body contract, changes workspace-slot ownership, lets Runtime choose a route, or requires [Model](../../../../glossary.md#term-model) permission beyond the cited contracts, stop for an explicit architecture decision.

## Implementation prompt

```text
You are working in the Synaptik repository. Read AGENTS.md, ARCHITECTURE.md,
docs/planning/planning-guide.md, docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0008p-deterministic-partial-reduction-parallelism.md.
Implement CPU 0008P exactly as specified. Do not broaden family/carrier/layout scope or alter
architecture/shared contracts. Stop on architecture or Model-permission conflict. Before any timed
fork, implement and seal the exact 24-row, three-comparison, five-fork protocol in this task; do
not substitute an open-ended benchmark or reuse evidence from another root. Admit only covered
form/`P` facts after every hash-valid gate passes; otherwise keep the whole-cell route. Run the
specified CPU validation and hand the final diff plus exact evidence to a separate clean
documentation-focused agent under docs/developer-guide/documentation-rules.md. Do not commit or
push. Update evidence and status only after that pass completes.
```

## Local decisions

- The first partial route uses an aligned eight-byte state slice for both admitted types. This keeps
  every state address eight-byte aligned while preserving direct four-byte <span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>[`INT32`](../../../../glossary.md#data-type--datatype) and eight-byte
  [`INT64`](../../../../glossary.md#data-type--datatype) arithmetic; it does not create a reusable numerical abstraction.
- A scalar output cell is admitted when its domain is large enough. Requiring two output cells
  would exclude the main full-reduction case while adding no safety property, because work is
  partitioned within each complete output-cell domain.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- `P` and worker capacity are prepared facts. Modular arithmetic makes result bits independent of
  valid <span data-glossary-navigation="true"> ([Modular arithmetic glossary](../../../../glossary.md#term-modular-arithmetic))</span>`P`, but no schedule, assignment, timing, or cross-[backend identity](../../../../glossary.md#backend-identity--backendid) is part of the route.
- The profitability protocol deliberately times `P=2` and `P=4` only. The fixture supplies four
  worker slots, making <span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>`P=4` its maximum; production must fail closed rather than infer admission
  for another prepared partial count from these measurements.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [admission glossary](../../../../glossary.md#term-admission))</span>

## Known limitations

- Only fully static dense primitive-array INT32/INT64 ordinary SUM/PROD domains are eligible.
  All other layouts, carriers, types, forms, and reduction families retain the existing route.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [layouts glossary](../../../../glossary.md#layout))</span>
- This task deliberately has no floating route: Model 0018V leaves its conformance tolerance
  future, and no exact-state implementation may replace that missing semantic permission.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [conformance glossary](../../../../glossary.md#term-conformance); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- Profitability is unproved until the required generated-versus-direct five-fork evidence passes;
  the selector uses the existing prepared minimum-work fact and must retain the whole-cell fallback
  whenever that fact, worker capacity, or measured acceptance gates do not support the route.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Documentation-finalization context: this mandatory independent clean-context pass. Read
  `AGENTS.md`, `ARCHITECTURE.md`, the current architecture navigation, documentation rules and
  General/Java-Javadoc/Planning profiles, the planning guide, roadmap, CPU master plan, this
  task, changed production/test sources, relevant CPU package documentation and CPU guide, the
  glossary, and the complete actual diff. The direct implementation and focused tests establish
  the separate partial IR, quotient/remainder ranges, aligned run-owned state, generated partial
  fold plus ascending caller combine, and the fail-closed prepared selector. The generated-code
  oracle result is structural/semantic parity with the specified direct typed algorithm; it does
  not claim JIT assembly identity or a measured speed result.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [oracle glossary](../../../../glossary.md#term-test-oracle); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The sole allowed evidence root remains
  `/private/tmp/synaptik-cpu-0008p-20260906-L4XZed`. It contains only `protocol.json` and the
  initial `environment.json`; it failed before sealing, forks, timings, or summaries because the
  then-current harness used a Gradle-working-directory-relative source path. It is immutable,
  non-passing, and may not be retried, appended, replaced, or supplemented. The later harness
  correction is for future work only. Accordingly, every production selection is deliberately<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [immutable glossary](../../../../glossary.md#term-immutable))</span>
  `KEEP_WHOLE_CELL`, including when a caller forges a diagnostic `PartialReductionEvidence`.
- Reused implementation evidence: focused IR, lowering, generated-kernel, execution, preparation,
  prepared-executable, and inventory tests passed after final code changes. An earlier full CPU
  run had three failures: the now-corrected required inventory omission, the existing<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel))</span>
  `CpuPartitionDagGeneratedEvidenceTest` fork-2 no-accepted-sample timing failure, and the
  existing <span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>`CpuWorkerGroupTest` close-race failure. This task does not claim a final full CPU
  suite pass; that validation is separately pending after this documentation pass.<span data-glossary-navigation="true"> ([race glossary](../../../../glossary.md#term-race))</span>
- Documentation decisions: finalized affected production Javadocs document the CPU-private,
  cold/prepared boundary, typed partial invocation, and non-admitted route. The CPU guide and
  relevant package documentation remain accurate because no optimization is admitted and this
  task may not edit those paths. The glossary needs no change: partial-reduction protocol labels
  and `KEEP_WHOLE_CELL` are task-local, not reusable project terminology.
- Documentation validation: `./gradlew :backends:cpu:javadoc` completed successfully after the
  final Javadoc edits. It retained 94 pre-existing Javadoc warnings in unrelated broad records
  plus incubating-Vector notices; it reported no warning from this task's added documentation.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector))</span>
  `git diff --check`, a trailing-whitespace scan of all three permitted planning files, the
  planning-link/anchor review, and the exact-path inventory check passed; the inventory is exactly
  21 paths (3 planning, 11 production, and 7 tests).<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>

- Planning/documentation review context: this independent clean-context correction. Read the
  governing architecture, planning, and documentation contracts; inspected the current diff and
  `CpuInternalPackageInventoryTest`. Its exact production-file allowlist omits
  `CpuPartialReductionEmitter`, `CpuPartialReductionExecution`, `CpuPartialReductionIr`, and
  `CpuPartialReductionLowering`, so `:backends:cpu:test` requires the explicitly authorized
  seventh test path. The scope therefore changes only from 20 to 21 paths and from six to exactly
  seven focused test paths; Java was not edited in this context.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Frozen evidence outcome: the sole authorized root
  `/private/tmp/synaptik-cpu-0008p-20260906-L4XZed` contains only `protocol.json` and the initial
  `environment.json`. Its parent failed before the protocol seal, child forks, timings, or
  summaries because the then-current harness resolved its source hash through a
  Gradle-working-directory-relative path. This root is incomplete and non-passing; it must not be
  deleted, appended, retried, replaced, or treated as passing. The harness source-location logic
  has since been corrected for future work, but CPU 0008P authorizes no new evidence run. The
  actual production admission decision remains <span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash); [admission glossary](../../../../glossary.md#term-admission))</span>`KEEP_WHOLE_CELL`.
- Glossary impact: none. This correction changes neither reusable project terminology nor a term
  boundary; `KEEP_WHOLE_CELL` and the evidence-root names remain task-local protocol labels.

- [Planning](../../../../glossary.md#term-planning)/documentation review context: `01a07813-7d52-7873-b11a-57f95d169ba9` (independent
  clean-context path-budget review).
- Re-read the architecture contract, current architecture navigation, planning guide, roadmap,
  CPU master plan, this task, documentation rules, and General and Planning profiles. Inspected
  the current CPU aggregate IR, aggregate lowering, partition-lowering/DAG/preparation/finalization
  handoff, prepared executable, generated-kernel generator, and schema-specialization topology.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [handoff glossary](../../../../glossary.md#term-handoff); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [schema glossary](../../../../glossary.md#term-schema); [specialization glossary](../../../../glossary.md#term-specialization); [topology glossary](../../../../glossary.md#term-topology))</span>
  `CpuAggregateIr` fixes complete output-cell ranges and the aggregate emitter owns that existing
  route; a partial route consequently requires a separate partial IR plus a distinct generated
  partial/combine path and the owners that carry its cold geometry, workspace, and execution
  recipe. The existing ten-production-path ceiling therefore prevented the task's mandatory
  coherent implementation before editing. Eleven is the smallest truthful production-path budget;
  no scope, family, carrier, semantic, performance, shared-contract, or architecture constraint
  was broadened or weakened.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Glossary impact: none. `partial reduction`, workspace, and generated route are task-local
  planning descriptions here; this correction creates no new reusable project term or changes an
  existing term's meaning.
- Documentation validation in this context passed: `git diff --check`; a local relative-link
  target scan of this task; and <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`rg -n "[[:blank:]]+$"
  docs/planning/backends/cpu/tasks/0008p-deterministic-partial-reduction-parallelism.md` (no
  output). No Java or Javadoc command applies because this pass changes planning text only.
- Planning/documentation review context: `01a0780a-0b6e-7153-afdb-0c3d6c3037e6` (independent clean-context review).
- Read and checked the architecture contract, current architecture navigation, planning guide,
  CPU master plan/roadmap, Model 0018U1 and 0018V, CPU 0007A1/0008B/0008O, current aggregate IR
  and lowering contracts, documentation rules, and General and Planning profiles.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Verified that Model 0018V says floating reassociation requires a future conformance tolerance
  and defines none; Model 0018U1 permits same-width modular integral SUM/PROD reassociation.
  Verified <span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [conformance glossary](../../../../glossary.md#term-conformance); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>`CpuAggregateIr` currently retains complete output-cell ranges and [CPU](../../../../glossary.md#term-cpu) 0007A1 uses
  `iadd`/`imul` and `ladd`/`lmul` modular semantics, so a separate partial [IR](../../../../glossary.md#term-intermediate-representation-ir) remains required.
- `git diff --check` passed; a local relative-link scan found no missing task-link target. No Java
  or Javadoc command was run because this review changes planning files only.
- Protocol-finalization review (this separate clean documentation/planning context): read the ADR
  and retained CPU 0008M, 0008N, 0008N1, and 0008O evidence conventions in addition to the
  governing files above. The new protocol freezes a finite matrix, three explicit comparisons,
  warmup/calibration/sample arithmetic, numeric gates, external immutable-evidence schemas,
  seed/hash identity, one-shot rules, and fail-closed selection before timing. Glossary impact is
  none: these are task-local evidence terms, not a changed reusable domain term.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [finite glossary](../../../../glossary.md#term-finite); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [warmup glossary](../../../../glossary.md#term-warmup); [seed glossary](../../../../glossary.md#term-seed))</span>
- Protocol-finalization validation passed: `git diff --check`; `rg -n '[[:blank:]]+$'
  docs/planning/backends/cpu/tasks/0008p-deterministic-partial-reduction-parallelism.md` (no
  output); and a `test -f` scan of every local Markdown target on lines 48--51 (all targets
  exist). The arithmetic is explicitly `2*2*3*2=24`, `24*5*9*3=3240`, `3240*4=12960`,
  `24*5*3=360`, and `24*3=72`. No Java or [Javadoc](../../../../glossary.md#term-javadoc) command applies to this planning-only edit.

## Implementation notes

Implementation completed within the 21-path budget: three [planning](../../../../glossary.md#term-planning) paths, eleven production [CPU](../../../../glossary.md#term-cpu)
paths, and seven focused test paths. The generated partial/combine machinery is present and tested,
but the sole authorized profitability root is non-passing; the prepared production selector is
therefore intentionally unreachable and retains the existing whole-cell execution route.

## Completion summary

[CPU](../../../../glossary.md#term-cpu) 0008P is complete as a bounded [fail-closed](../../../../glossary.md#term-fail-closed) outcome. It delivers the private generated
partial/combine architecture and its direct tests while admitting no optimization: the [immutable](../../../../glossary.md#term-immutable)
non-passing profitability record mandates `KEEP_WHOLE_CELL`. Full CPU [validation](../../../../glossary.md#term-validation) is not claimed
here; its three earlier failures are recorded above for the separately scheduled final validation.

Status: Complete
