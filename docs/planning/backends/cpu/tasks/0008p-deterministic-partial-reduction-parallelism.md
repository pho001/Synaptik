# CPU Task 0008P: Deterministic Modular Partial-Reduction Parallelism

## Status

Ready

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
| partial eligibility | non-empty domain, at least one output cell, at least two prepared partial ranges, every partial meets the existing `CpuPartitionAnalysisInputs.minimumElementsPerWorker()` fact, and checked workspace geometry fits |

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
- Freeze partial count `P`, range formula, output order, partial order, state layout, and combine algorithm in preparation. `P >= 2` and is at most worker capacity and valid partial-work count. The recipe fixes result construction, but does not fix worker scheduling; modular semantics make the admitted result bit-identical across valid `P` and worker counts.
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

Expected implementation paths (at most 17 CPU source/test paths) include aggregate lowering,
partition lowering/DAG/preparer/finalizer handoff, prepared executable, generated
emitter/generator, workspace binding, a new partial IR/emitter/reference, and focused tests
mirroring their owners. `CpuAggregateIr` continues to mean a complete output-cell range; the
separate partial IR and its generated partial/combine route therefore cannot truthfully be folded
into the existing ordinary-emitter path merely to meet a smaller file count.

Planning paths are exactly:

- this task;
- `docs/planning/backends/cpu/master-plan.md`; and
- `docs/planning/roadmap.md`.

## Maximum scope

At most 20 paths: these three planning paths, eleven production CPU paths, and six focused CPU
test paths. The eleven production paths are the minimum coherent current-topology budget: the
separate `CpuPartialReductionIr` and generated partial/combine route require their own path in
addition to the existing cold lowering, preparation/finalization, and execution handoff owners.
If a shared Prepare/Runtime contract, second workspace type, Model change, another family, or a
21st path is needed, stop and create a separately scoped follow-up.

## Acceptance criteria

1. A separate partial-reduction IR expresses partial body/state/combine/fixed partition/final publication; `CpuAggregateIr` retains complete output-cell range meaning.
2. Lowering selects it only for the exact scope table; all exclusions, overflow, and absent worker proof retain scalar/caller-parallel whole-cell execution.
3. Analysis/finalization declares/validates exact aligned `C * P * S` workspace; partials own disjoint state, combine owns writes, publication follows successful ordered combines.
4. Generated partial/combine and frozen clean-Java oracle are equivalent direct typed algorithms. Generated/decompiled inspection proves no helper/dispatch, allocation, boxing, reflection, carrier lookup, route selection, or worker submission in hot loops.
5. Focused semantics cover both identities, modular overflow, negative operands, empty/zero output fallback, full/single/multi forms, P=2/maximum P, coverage/no overlap, scalar/whole-cell/partial bit identity across valid `P` and worker counts, replay, cancellation/failure/no output publication, interruption, and concurrent RunState isolation.
6. A named finite generated-identity matrix covers both types and kinds, forms, dense offsets where proof permits, P=2/maximum, scalar/whole-cell/partial strategy, and all admitted array signatures. A proportional five-fork generated-versus-frozen-direct performance matrix proves partial and combine, retains raw samples, and sets justified per-case gates before timing.
7. A separate documentation-focused agent finalizes Javadocs, CPU guide/glossary impact, task/master/roadmap, and reasoned no-change conclusions in the same change.

## Tests / validation

Run focused CPU IR/lowering/emitter/executable tests, then:

```bash
./gradlew :backends:cpu:test
./gradlew :backends:cpu:javadoc
git diff --check
```

Add generated Class-File/decompilation and import/manual scans, checked resource/range matrix, and retained five-fork direct-oracle evidence. Repository-wide validation is deferred to CPU 0009/CI because no shared contract changes.

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
architecture/shared contracts. Stop on architecture or Model-permission conflict. Run the specified
CPU validation and hand the final diff plus exact evidence to a separate clean documentation-focused
agent under docs/developer-guide/documentation-rules.md. Do not commit or push. Update evidence and
status only after that pass completes.
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

## Known limitations

- Only fully static dense primitive-array INT32/INT64 ordinary SUM/PROD domains are eligible.
  All other layouts, carriers, types, forms, and reduction families retain the existing route.
- This task deliberately has no floating route: Model 0018V leaves its conformance tolerance
  future, and no exact-state implementation may replace that missing semantic permission.
- Profitability is unproved until the required generated-versus-direct five-fork evidence passes;
  the selector uses the existing prepared minimum-work fact and must retain the whole-cell fallback
  whenever that fact, worker capacity, or measured acceptance gates do not support the route.

## Validation evidence

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

## Implementation notes

No implementation has begun. A first clean implementation attempt stopped before editing because
the former ten-production-path ceiling conflicted with the required separate partial IR and
generated partial/combine route. This clean-context planning/documentation pass corrected the
minimum coherent budget to eleven production paths; it made no executable change. The
implementation agent must record generated artifact/schema impact, tests, structural evidence,
performance gates, and documentation handoff evidence here before requesting final review.

## Completion summary

Pending implementation. This planning review is complete, but CPU 0008P itself remains `Ready`,
not `Complete`.
