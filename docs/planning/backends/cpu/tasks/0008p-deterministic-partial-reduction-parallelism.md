# CPU Task 0008P: Deterministic Modular Partial-Reduction Parallelism

## Status

Complete

## Goal

Add one CPU-private, prepared partial-reduction execution architecture for a large ordinary integral `SUM` or `PROD` domain. It splits a domain into fixed disjoint partial ranges, retains one modular state for every partial, combines states in fixed ordinal order, and publishes only after a successful deterministic-result combine. This is a new partial-body/partial-state/combine realization, not an extension of the current output-cell range emitter.

## Scope

Implement exactly this first admitted realization:

| Fact | Admitted value |
|---|---|
| Model family | ordinary `AggregateReductionKind.SUM` and `PROD` only |
| CPU IR | one `CpuAggregateIr` ordinary `FULL`, `SINGLE_AXIS`, or `MULTI_AXIS`; never `SUM_TO_SHAPE` |
| represented/result type | same-typed `INT32` or `INT64` only |
| layout/carriers | fully static resolved dense-linear input and injective dense-linear output; primitive heap `int[]`/`long[]` boundaries only |
| topology | one direct aggregate computation unit, one input and one output, no fusion/materialization |
| modes | existing scalar and caller-parallel complete-output-cell modes, plus selected `PARTIAL_REDUCTION_PARALLEL` only |
| partial eligibility | non-empty domain, at least one output cell, `P` exactly 2 or 4 (`4` is the maximum admitted partial count), every partial meets the existing `CpuPartitionAnalysisInputs.minimumElementsPerWorker()` fact, checked workspace geometry fits, and the matching frozen performance row passed |

The selector is cold, deterministic, and fail-closed. Every zero-domain identity, zero-output Shape, small domain, non-dense access, segment/mixed carrier, BFLOAT16, or unlisted occurrence keeps the current whole-cell path. A one-cell output is eligible when its selected domain supplies the required partial work.

For output cell `c` and prepared partial ordinal `p`, fixed quotient/remainder partitioning creates one non-empty selected-domain interval `[begin(c,p), end(c,p))`. The worker phase invokes the generated partial body once per `(c,p)` and writes only `state(c,p)`. After one synchronous worker-group join, the invoking thread runs the generated combine body for output cells in ascending logical order, consumes partial states in ascending `p`, writes the output, then follows normal publication. No partial body writes output; no atomics, changing state ordinal, or early publication exists.

Use one same-width two's-complement modular state: `int`/`iadd` or `imul` for `INT32`, and `long`/`ladd` or `lmul` for `INT64`. Each partial begins with the existing identity (zero for `SUM`, one for `PROD`); combine uses the same operation, so every selected value is included once and overflow wraps modulo `2^32` or `2^64`. No floating state, rounding, tolerance, or special-value claim exists. The frozen optimal clean-Java oracle must use the same partition formula, state transition, combine order, and direct array signatures. It is test/performance-only and unreachable from production execution.

### Source-backed eligibility boundary

Model 0018V permits floating reassociation or parallelization only when observable results satisfy a *future* conformance tolerance; it explicitly does not define that tolerance. CPU 0008O likewise stopped floating reassociation where permission was not current. The current exact floating state cannot manufacture that missing conformance permission, even if it can represent an exact sum before one rounding. Model 0018U1 independently and currently permits exact-type modular reassociation for integral ordinary `SUM`/`PROD`. That is semantic permission, not CPU admission.

CPU 0008P consumes only ordinary INT32/INT64 `SUM`/`PROD` above. It does not claim a general `CpuAggregateIr` is partial-capable. Floating `SUM`/`MEAN`/`PROD`, extrema, BOOL, masked reductions, `SUM_TO_SHAPE`, and advanced `LOG_SUM_EXP`/variance/standard-deviation/L1/L2 remain unchanged future candidates requiring their own Model eligibility and state/combine proof. Cumulative scans remain whole-slice: prefix order is observable. Softmax/log-softmax, attention rows, Layer/RMS and batch normalization, MSE/categorical losses, and their stable/multi-pass reductions remain caller-parallel only. CPU 0008O's cancelled candidate-2 `KEEP_SCALAR` evidence grants no SIMD, lane-tree, or floating-reassociation permission.

The result is deterministic because modular `SUM`/`PROD` reassociation produces the same same-width bit pattern for a fixed input, regardless of the frozen `P` or worker count. The prepared range formula and ordinal combine make the recipe deterministic; they do not promise a deterministic worker schedule, worker assignment, timing, or cross-backend execution behavior.

## Out of scope

- Model, Compiler, Prepare, Runtime, backend-contract, public API, architecture, ADR, dependency, Gradle, conformance, or integration change.
- SIMD/vector partial bodies, lane/tree reduction, floating tolerance or reassociation, a deterministic-schedule/timing/cross-backend promise, or use of 0008O as permission.
- Partial paths for scans, floating/masked/advanced/statistical/norm reductions, softmax, normalization, attention, losses, matmul/convolution/pooling, BOOL/extrema aggregates, integral `MEAN`, or `SUM_TO_SHAPE`.
- Segments, mixed carriers, strided/zero-strided/broadcast layouts, dynamic Shapes, fusion, materialization, nested submission, per-worker allocation, persistent/thread-local scratch, atomics, locks in generated loops, or Runtime selection.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), including staged preparation, RunState, workspace ownership, and runtime hot-path rules.
- [Current architecture navigation](../../../../architecture/current-architecture-plan.md), [planning guide](../../../planning-guide.md), and [CPU master plan](../master-plan.md).
- CPU [ordinary aggregates](0007a1-portable-ordinary-numerical-aggregate-reductions.md), [masked reductions](0007c-portable-masked-reduction-coverage.md), [advanced reductions](0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md), [scans](0007-portable-cumulative-scan-coverage.md), [softmax](0007e-portable-stable-softmax-and-log-softmax-coverage.md), [normalization](0007f-portable-layer-and-rms-normalization-coverage.md), [losses](0008i-portable-loss-family-execution.md), [attention](0008h-portable-scaled-dot-product-attention-execution.md), and cancelled [0008O](0008o-stable-reduction-vector-numerical-spike.md).
- Model [integral aggregate](../../../modules/model/tasks/0018u1-integral-reductions-and-arg-min-normalization.md), [multi-axis/statistical](../../../modules/model/tasks/0018v-multi-axis-and-statistical-reductions.md), [scan](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md), [softmax](../../../modules/model/tasks/0016j-softmax-tensor-expressions.md), [attention](../../../modules/model/tasks/0019e-scaled-dot-product-attention.md), [normalization](../../../modules/model/tasks/0021-layer-normalization-semantics-and-tensor-expressions.md), [batch normalization](../../../modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md), and [loss](../../../modules/model/tasks/0022-mean-squared-error-loss.md).

## Architecture constraints

- Model owns mathematics and reassociation permission; CPU owns truthful lowering, cold selection, generated realization, workspace, and execution. Never manufacture floating permission or a tolerance.
- Analysis declares one exact workspace before shared assignment. Finalization uses that slot without changing route, partial count, or size. Runtime only binds/releases run-owned workspace and executes the prepared recipe.
- Prepared execution remains immutable/reusable; each RunState gets distinct workspace. Inputs are borrowed, workspace is run-owned, and failure cleanup releases only still-run-owned resources.
- Freeze partial count `P`, range formula, output order, partial order, state layout, and combine algorithm in preparation. `P` is exactly `2` or `4`, and `4` is the maximum admitted count; it is also at most worker capacity and valid partial-work count. The recipe fixes result construction, but does not fix worker scheduling; modular semantics make the admitted result bit-identical across valid `P` and worker counts.
- Borrow `CpuWorkerGroup` once for one flat indexed partial-work list. No worker submits, closes, or waits on it. Existing failure ordering, cancellation, interruption, accessibility, and join rules remain authoritative. Failure prevents combine/publication.
- Let `C` be output cells, `P` partial count, and `S = 8` be one aligned state slice: the `INT32` payload occupies its low 4 bytes and `INT64` its 8-byte payload. Declare `alignUpExact(C * P * S, 8)` with checked multiplication; no partial workspace unless this route is selected. Each aligned `state(c,p)` is non-overlapping and lives from binding through combine completion. Validate span/alignment/accessibility and workspace/input/output non-overlap before submission.

## Package impact

Existing packages used:

- `...backend.cpu.internal.ir` — separate partial-reduction IR and aggregate facts; do not overload whole-cell range meaning.
- `...internal.lowering` — eligibility and cold geometry.
- `...internal.codegen.emit` — dedicated typed partial/combine emitters.
- `...internal.prepare` and `...internal.executable` — staged resource declaration, finalization, binding, orchestration, and publication.
- `...internal.reference` — frozen test-only clean-Java oracle.

Packages added or changed: none. Expected placement is `CpuPartialReductionIr`, its lowering geometry, and dedicated emitter/reference test types in those packages; refine names only by updating this task and the master-plan package map before implementation.

## Affected files

Expected implementation paths (at most 18 CPU source/test paths) include aggregate lowering,
partition lowering/DAG/preparer/finalizer handoff, prepared executable, generated
emitter/generator, workspace binding, a new partial IR/emitter/reference, and focused tests
mirroring their owners. `CpuAggregateIr` continues to mean a complete output-cell range; the
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
must name each of this task's four new internal production types for the CPU module test suite to
admit them.

Planning paths are exactly:

- this task;
- `docs/planning/backends/cpu/master-plan.md`; and
- `docs/planning/roadmap.md`.

The implementation evidence root is one caller-supplied, initially empty directory outside the
repository, named `synaptik-cpu-0008p-<run-id>`. It is not a repository path and counts as zero of
the 20-path repository budget; its required files are nevertheless part of the acceptance
inventory below. No evidence artifact may be silently added under a source, test, or planning
directory. If retained evidence must be tracked in the repository, stop and re-plan the path
budget before writing it.

## Maximum scope

At most 21 paths: these three planning paths, eleven production CPU paths, and exactly seven
focused CPU test paths. The eleven production paths are the minimum coherent current-topology
budget: the separate `CpuPartialReductionIr` and generated partial/combine route require their
own path in addition to the existing cold lowering, preparation/finalization, and execution
handoff owners. The seventh test path is the mandatory CPU internal-package inventory update;
without it, the full CPU validation rejects the four required new production types. This is the
minimum coherent correction and does not broaden implementation, semantic, or architectural
scope. If a shared Prepare/Runtime contract, second workspace type, Model change, another family,
or a 22nd path is needed, stop and create a separately scoped follow-up.

## Acceptance criteria

1. A separate partial-reduction IR expresses partial body/state/combine/fixed partition/final publication; `CpuAggregateIr` retains complete output-cell range meaning.
2. Lowering selects it only for the exact scope table and the complete passing profitability
   evidence for the matching form and `P` (`P=2` or `P=4`, the prepared maximum in the four-worker
   admission configuration); all exclusions, overflow, absent worker proof, or incomplete/failed
   evidence retain scalar/caller-parallel whole-cell execution.
3. Analysis/finalization declares/validates exact aligned `C * P * S` workspace; partials own disjoint state, combine owns writes, publication follows successful ordered combines.
4. Generated partial/combine and frozen clean-Java oracle are equivalent direct typed algorithms. Generated/decompiled inspection proves no helper/dispatch, allocation, boxing, reflection, carrier lookup, route selection, or worker submission in hot loops.
5. Focused semantics cover both identities, modular overflow, negative operands, empty/zero output fallback, full/single/multi forms, P=2/maximum P, coverage/no overlap, scalar/whole-cell/partial bit identity across valid `P` and worker counts, replay, cancellation/failure/no output publication, interruption, and concurrent RunState isolation.
6. A named finite generated-identity matrix covers both types and kinds, forms, dense offsets where proof permits, `P=2`/maximum, scalar/whole-cell/partial strategy, and all admitted array signatures. The timed admission matrix and its five-fork protocol below are frozen before the first fork; it is the only profitability evidence permitted for production selection.
7. A separate documentation-focused agent finalizes Javadocs, CPU guide/glossary impact, task/master/roadmap, and reasoned no-change conclusions in the same change.

### Frozen partial-reduction profitability protocol

This protocol answers one narrow question: whether the exact generated primitive-array partial
body plus ordinal combine is profitable enough to replace the existing generated complete-output-
cell route for the covered prepared facts. It does not add semantic coverage; ordinary focused
tests own identities, overflow, cancellation, range coverage, `P` boundaries, and RunState
isolation.

Before timing, encode and assert this exact 24-row inventory. `C` is output-cell count and `D` is
selected-domain elements per output cell. Every row has one fully static dense `int[]` or `long[]`
input and injective dense output, zero base offsets, at least four available worker slots, and
both fixed partial counts shown. The test fixture must also derive a positive-offset identity row
for every type/kind/form outside timing; offset is not a timed profitability variable.

| Form | Input shape and axes | `C` | `D` | `P` values |
|---|---|---:|---:|---|
| `FULL` | `[524288]`, all axes | 1 | 524288 | 2, 4 |
| `SINGLE_AXIS` | `[64, 8192]`, axis `1` | 64 | 8192 | 2, 4 |
| `MULTI_AXIS` | `[4, 16, 2048]`, axes `1,2` | 4 | 32768 | 2, 4 |

```text
2 types (INT32, INT64) * 2 kinds (SUM, PROD) * 3 forms * 2 P values = 24 rows
```

For each row, the harness uses four independently named, shape-polymorphic implementations with
identical typed array signatures, input values, output geometry, fixed range formula, `P`, result
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
is deliberately above ordinary run-to-run noise and compensates for the route's workspace and
join complexity; 15% permits normal generated-class versus `javac` variation but rejects a hidden
algorithmic or dispatch penalty. Every individual retained pair, every per-fork median, and the
median of five fork medians must meet its applicable gate. A missing full `P=4` worker proof, a
failed parity gate, or any missed profitability gate is `KEEP_WHOLE_CELL` for that exact row.
For one symmetric pair, its ratio is the sum of the two `A` side durations divided by the sum of
the two `B` side durations in execution order; the fork statistic is the median of its nine pair
ratios, and the row aggregate is the median of its five fork medians. No mean, best sample, or
cross-row aggregate participates in admission.

Use exactly five fresh Java 26 child JVM forks per row, with
`-Xms1g -Xmx1g -XX:-TieredCompilation -Xbatch`. Each fork runs five randomized symmetric warmup
pairs for each of `Gp/Wg`, `Gp/Dp`, and `Wg/Dw`; each pair is either `A-B-B-A` or `B-A-A-B`.
After warmup, double one shared iteration count until two invocations of each of `Wg`, `Dw`, `Gp`,
and `Dp` take at least 50 ms. Retain exactly nine randomized symmetric pairs for each comparison;
all four timed sides of every pair must individually take at least 25 ms. Thus the immutable
inventory is:

```text
24 rows * 5 forks * 9 pairs * 3 comparisons = 3,240 retained pairs
3,240 pairs * 4 timed sides = 12,960 timings
24 rows * 5 forks * 3 comparisons = 360 fork medians
24 rows * 3 comparisons = 72 median-of-fork-medians aggregates
```

The order generator is SplitMix64 with base seed `0x0000000000080050L`; derive the row seed from
the canonical row identifier and the fork seed from `(row seed, fork ordinal)`. Generate input
values from that seed with a fixed documented mapping to the signed values `[-3, 3]`; record the
SHA-256 of each canonical little-endian typed input byte sequence and of every consumed output
sequence. The implementation source snapshot, generated class bytes, compiled `Dp`/`Dw` class
bytes, and exact command lines are hashed before fork zero. The fixture must reject a changed
seed, class/source hash, row inventory, JVM flags, gates, calibration rule, sample count, or input
hash before it launches a timed fork.

The evidence root contains immutable `protocol.json`, append-only `forks/row-<id>-fork-<0..4>.csv`,
`environment.json`, `classes.json`, `inputs.json`, `summary.json`, and `SHA256SUMS`. `protocol.json`
has schema `synaptik.cpu.partial-reduction-performance.v1` and records the 24 canonical row IDs,
comparison definitions, gates, warmup/measured counts, flags, seed derivation, and file schemas.
Each CSV has one row per timed side with `row_id,fork,comparison,pair,side,order,iterations,ns,
input_sha256,output_sha256`; it is written only in prescribed pair/side order. `environment.json`
records OS/version/architecture, CPU model and logical processors, JDK vendor/version/build,
JVM flags, heap, available workers, affinity/governor facts when available (otherwise explicit
`unavailable`), and wall-clock timestamps. `classes.json` records all source/class hashes and
direct descriptors; `inputs.json` records canonical geometry, `P`, seed, mapping, and hashes;
`summary.json` records every pair ratio, fork median, aggregate, gate result, and row decision.
`SHA256SUMS` covers every completed file. The harness verifies an existing root before append and
refuses changed or missing prior hashes.

There is one one-shot run: rows/forks execute in canonical row then fork order; no retry, discard,
replacement, outlier filtering, rerun-to-pass, fixture change, post-hoc batching, threshold change,
or partial-fork substitution is allowed. An interruption, timing-floor failure, calibration failure,
missing metadata/hash, corrupted artifact, or 30-minute whole-protocol wall-clock limit records
the precise failure and makes every unfinished or unverifiable row `KEEP_WHOLE_CELL`. Results from
another root never supplement this run. Production admission is fail-closed: only a complete,
hash-valid root in which all 24 rows pass every stated gate permits this route's selector for the
exact covered `INT32`/`INT64` `SUM`/`PROD`, form, and `P` facts. Otherwise the selector must keep
the existing whole-cell route; it may not extrapolate one row's pass to another form, type, kind,
or partial count.

## Tests / validation

Run focused CPU IR/lowering/emitter/executable tests, then:

```bash
./gradlew :backends:cpu:test
./gradlew :backends:cpu:javadoc
git diff --check
```

Add generated Class-File/decompilation and import/manual scans, checked resource/range matrix,
and the frozen 24-row/5-fork evidence protocol above. The performance command must require the
empty external evidence root and fail before timing if its protocol seal cannot be created.
Repository-wide validation is deferred to CPU 0009/CI because no shared contract changes.

## Dependencies

- CPU 0007A1 ordinary aggregate modular `INT32`/`INT64` route and its generated-operation conventions.
- CPU 0008B partition-DAG finalization and existing `CpuWorkerGroup` lifecycle.
- CPU 0008O cancelled scalar evidence, only as SIMD prohibition.
- Current Model 0018U1 modular-reassociation contract; Model 0018V is a floating non-eligibility boundary, not a dependency granting permission.

## Follow-up tasks

- Another family needs its own Model eligibility, state/combine, resource, publication, and oracle proof.
- CPU 0009 remains the later generated-coverage checkpoint; do not detail it here.

## Architecture impact

Expected impact: None. The architecture already authorizes backend-private lowering/route selection, staged exact workspace declaration, immutable prepared recipes, isolated RunState resources, and prepared Runtime schedules. No ADR or architecture test is expected because no dependency/ownership rule changes. If implementation needs a shared partial-body contract, changes workspace-slot ownership, lets Runtime choose a route, or requires Model permission beyond the cited contracts, stop for an explicit architecture decision.

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
  every state address eight-byte aligned while preserving direct four-byte `INT32` and eight-byte
  `INT64` arithmetic; it does not create a reusable numerical abstraction.
- A scalar output cell is admitted when its domain is large enough. Requiring two output cells
  would exclude the main full-reduction case while adding no safety property, because work is
  partitioned within each complete output-cell domain.
- `P` and worker capacity are prepared facts. Modular arithmetic makes result bits independent of
  valid `P`, but no schedule, assignment, timing, or cross-backend identity is part of the route.
- The profitability protocol deliberately times `P=2` and `P=4` only. The fixture supplies four
  worker slots, making `P=4` its maximum; production must fail closed rather than infer admission
  for another prepared partial count from these measurements.

## Known limitations

- Only fully static dense primitive-array INT32/INT64 ordinary SUM/PROD domains are eligible.
  All other layouts, carriers, types, forms, and reduction families retain the existing route.
- This task deliberately has no floating route: Model 0018V leaves its conformance tolerance
  future, and no exact-state implementation may replace that missing semantic permission.
- Profitability is unproved until the required generated-versus-direct five-fork evidence passes;
  the selector uses the existing prepared minimum-work fact and must retain the whole-cell fallback
  whenever that fact, worker capacity, or measured acceptance gates do not support the route.

## Validation evidence

- Documentation-finalization context: this mandatory independent clean-context pass. Read
  `AGENTS.md`, `ARCHITECTURE.md`, the current architecture navigation, documentation rules and
  General/Java-Javadoc/Planning profiles, the planning guide, roadmap, CPU master plan, this
  task, changed production/test sources, relevant CPU package documentation and CPU guide, the
  glossary, and the complete actual diff. The direct implementation and focused tests establish
  the separate partial IR, quotient/remainder ranges, aligned run-owned state, generated partial
  fold plus ascending caller combine, and the fail-closed prepared selector. The generated-code
  oracle result is structural/semantic parity with the specified direct typed algorithm; it does
  not claim JIT assembly identity or a measured speed result.
- The sole allowed evidence root remains
  `/private/tmp/synaptik-cpu-0008p-20260906-L4XZed`. It contains only `protocol.json` and the
  initial `environment.json`; it failed before sealing, forks, timings, or summaries because the
  then-current harness used a Gradle-working-directory-relative source path. It is immutable,
  non-passing, and may not be retried, appended, replaced, or supplemented. The later harness
  correction is for future work only. Accordingly, every production selection is deliberately
  `KEEP_WHOLE_CELL`, including when a caller forges a diagnostic `PartialReductionEvidence`.
- Reused implementation evidence: focused IR, lowering, generated-kernel, execution, preparation,
  prepared-executable, and inventory tests passed after final code changes. An earlier full CPU
  run had three failures: the now-corrected required inventory omission, the existing
  `CpuPartitionDagGeneratedEvidenceTest` fork-2 no-accepted-sample timing failure, and the
  existing `CpuWorkerGroupTest` close-race failure. This task does not claim a final full CPU
  suite pass; that validation is separately pending after this documentation pass.
- Documentation decisions: finalized affected production Javadocs document the CPU-private,
  cold/prepared boundary, typed partial invocation, and non-admitted route. The CPU guide and
  relevant package documentation remain accurate because no optimization is admitted and this
  task may not edit those paths. The glossary needs no change: partial-reduction protocol labels
  and `KEEP_WHOLE_CELL` are task-local, not reusable project terminology.
- Documentation validation: `./gradlew :backends:cpu:javadoc` completed successfully after the
  final Javadoc edits. It retained 94 pre-existing Javadoc warnings in unrelated broad records
  plus incubating-Vector notices; it reported no warning from this task's added documentation.
  `git diff --check`, a trailing-whitespace scan of all three permitted planning files, the
  planning-link/anchor review, and the exact-path inventory check passed; the inventory is exactly
  21 paths (3 planning, 11 production, and 7 tests).

- Planning/documentation review context: this independent clean-context correction. Read the
  governing architecture, planning, and documentation contracts; inspected the current diff and
  `CpuInternalPackageInventoryTest`. Its exact production-file allowlist omits
  `CpuPartialReductionEmitter`, `CpuPartialReductionExecution`, `CpuPartialReductionIr`, and
  `CpuPartialReductionLowering`, so `:backends:cpu:test` requires the explicitly authorized
  seventh test path. The scope therefore changes only from 20 to 21 paths and from six to exactly
  seven focused test paths; Java was not edited in this context.
- Frozen evidence outcome: the sole authorized root
  `/private/tmp/synaptik-cpu-0008p-20260906-L4XZed` contains only `protocol.json` and the initial
  `environment.json`. Its parent failed before the protocol seal, child forks, timings, or
  summaries because the then-current harness resolved its source hash through a
  Gradle-working-directory-relative path. This root is incomplete and non-passing; it must not be
  deleted, appended, retried, replaced, or treated as passing. The harness source-location logic
  has since been corrected for future work, but CPU 0008P authorizes no new evidence run. The
  actual production admission decision remains `KEEP_WHOLE_CELL`.
- Glossary impact: none. This correction changes neither reusable project terminology nor a term
  boundary; `KEEP_WHOLE_CELL` and the evidence-root names remain task-local protocol labels.

- Planning/documentation review context: `01a07813-7d52-7873-b11a-57f95d169ba9` (independent
  clean-context path-budget review).
- Re-read the architecture contract, current architecture navigation, planning guide, roadmap,
  CPU master plan, this task, documentation rules, and General and Planning profiles. Inspected
  the current CPU aggregate IR, aggregate lowering, partition-lowering/DAG/preparation/finalization
  handoff, prepared executable, generated-kernel generator, and schema-specialization topology.
  `CpuAggregateIr` fixes complete output-cell ranges and the aggregate emitter owns that existing
  route; a partial route consequently requires a separate partial IR plus a distinct generated
  partial/combine path and the owners that carry its cold geometry, workspace, and execution
  recipe. The existing ten-production-path ceiling therefore prevented the task's mandatory
  coherent implementation before editing. Eleven is the smallest truthful production-path budget;
  no scope, family, carrier, semantic, performance, shared-contract, or architecture constraint
  was broadened or weakened.
- Glossary impact: none. `partial reduction`, workspace, and generated route are task-local
  planning descriptions here; this correction creates no new reusable project term or changes an
  existing term's meaning.
- Documentation validation in this context passed: `git diff --check`; a local relative-link
  target scan of this task; and `rg -n "[[:blank:]]+$"
  docs/planning/backends/cpu/tasks/0008p-deterministic-partial-reduction-parallelism.md` (no
  output). No Java or Javadoc command applies because this pass changes planning text only.
- Planning/documentation review context: `01a0780a-0b6e-7153-afdb-0c3d6c3037e6` (independent clean-context review).
- Read and checked the architecture contract, current architecture navigation, planning guide,
  CPU master plan/roadmap, Model 0018U1 and 0018V, CPU 0007A1/0008B/0008O, current aggregate IR
  and lowering contracts, documentation rules, and General and Planning profiles.
- Verified that Model 0018V says floating reassociation requires a future conformance tolerance
  and defines none; Model 0018U1 permits same-width modular integral SUM/PROD reassociation.
  Verified `CpuAggregateIr` currently retains complete output-cell ranges and CPU 0007A1 uses
  `iadd`/`imul` and `ladd`/`lmul` modular semantics, so a separate partial IR remains required.
- `git diff --check` passed; a local relative-link scan found no missing task-link target. No Java
  or Javadoc command was run because this review changes planning files only.
- Protocol-finalization review (this separate clean documentation/planning context): read the ADR
  and retained CPU 0008M, 0008N, 0008N1, and 0008O evidence conventions in addition to the
  governing files above. The new protocol freezes a finite matrix, three explicit comparisons,
  warmup/calibration/sample arithmetic, numeric gates, external immutable-evidence schemas,
  seed/hash identity, one-shot rules, and fail-closed selection before timing. Glossary impact is
  none: these are task-local evidence terms, not a changed reusable domain term.
- Protocol-finalization validation passed: `git diff --check`; `rg -n '[[:blank:]]+$'
  docs/planning/backends/cpu/tasks/0008p-deterministic-partial-reduction-parallelism.md` (no
  output); and a `test -f` scan of every local Markdown target on lines 48--51 (all targets
  exist). The arithmetic is explicitly `2*2*3*2=24`, `24*5*9*3=3240`, `3240*4=12960`,
  `24*5*3=360`, and `24*3=72`. No Java or Javadoc command applies to this planning-only edit.

## Implementation notes

Implementation completed within the 21-path budget: three planning paths, eleven production CPU
paths, and seven focused test paths. The generated partial/combine machinery is present and tested,
but the sole authorized profitability root is non-passing; the prepared production selector is
therefore intentionally unreachable and retains the existing whole-cell execution route.

## Completion summary

CPU 0008P is complete as a bounded fail-closed outcome. It delivers the private generated
partial/combine architecture and its direct tests while admitting no optimization: the immutable
non-passing profitability record mandates `KEEP_WHOLE_CELL`. Full CPU validation is not claimed
here; its three earlier failures are recorded above for the separately scheduled final validation.

Status: Complete
