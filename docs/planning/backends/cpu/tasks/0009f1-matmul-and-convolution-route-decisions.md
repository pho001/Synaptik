# Task 0009F1: MATMUL and Convolution Route Decisions

## Status

Ready

## Goal

Make four independent, complete CPU route decisions from current source and tests: MATMUL, the
visible Conv1d composition, direct Conv2d, and direct Conv3d. Retain dense specialized generated
compute unless a complete finite direct-Java replacement is demonstrably strictly cheaper across
implementation, verification, and retirement. Finite direct Java is permitted only for a simpler
surrounding composition/orchestration boundary when strictly cheaper. Equality or uncertainty
retains generation. This task creates no performance gate or benchmark campaign.

## Scope

Audit and record one source- and test-backed decision for each separate row below. No row borrows
another row's decision, reference implementation, or verifier.

| Route | Current capability and semantic boundary | Current selected machinery | Complete decision boundary |
| --- | --- | --- | --- |
| MATMUL | Static resolved numeric vector/matrix/right-broadcast-batched MATMUL; non-negative layouts, injective output, exact promotion, rank-one unit M/N normalization, right batch broadcast, and legal empty zero work. Exact recognized suffix: optional rank-one result-typed ADD in either order plus at most ReLU, sigmoid, tanh, exact GELU, tanh-approximate GELU, SiLU, or typed clamp. | `CpuMatmulLowering` makes full-K geometry/access plans; `CpuMatmulCandidateSelector` chooses direct scalar, N-vector, 2x2 scalar, or 2x2 N-vector; `CpuMatmulEmitter` emits typed bodies; `CpuMatmulReferenceKernel` is the clean Java oracle. Output work units own half-open ranges; scalar/vector and caller parallelism bind cold. | Four realizations, types/carriers, dense/general layouts, batches/broadcasts, epilogues, ranges/parallelism, empty/special behavior, aliases/overlap, cold binding, artifact/cache/invocation/inventory. No generic epilogue framework. |
| Conv1d visible composition | Only exact NCW `EXPAND_DIMS(axis 2) -> CONV2D -> SQUEEZE(axis 2)`. Singleton height stays virtual. Conv2d supplies its groups/depthwise, batch, intrinsic bias, geometry, promotion, layouts, carriers, validation, empty behavior, overlap checks, ranges, and zero workspace. It is not a direct Conv1d operation, capability, artifact family, or generated body. | `CpuConv1dCompositionLowering` validates/delegates; no Conv1d IR, emitter, cache key, artifact, or inventory row exists. | Composition/orchestration only. A direct surrounding change must preserve the visible contract and be strictly cheaper; it must not create a direct Conv1d kernel or common abstraction. |
| Conv2d direct body | Static grouped NCHW floating cross-correlation: FLOAT64/FLOAT32/BFLOAT16 inputs, weights, optional intrinsic rank-one bias, ordered promotion, batch/groups/depthwise, positive stride/dilation, non-negative padding, arrays/native-order segments/ordered mixed carriers, checked scalar/vector half-open output ranges, border/tail/empty/special scalar semantics, aliases/overlap, and zero workspace/materialization. External suffix is none, same-type right-broadcast ADD, or ADD-RELU; others fail closed. | `CpuConv2dLowering`, `CpuConv2dIr`, `CpuConv2dEmitter`, `CpuConv2dReferenceKernel`; schema-63 vector/parallel-vector only for eligible dense FLOAT32/FLOAT64 width blocks, scalar otherwise. | Complete scalar/vector generated family plus suffix, geometry/binding, artifact identity/cache, invocation, inventory, validation and retirement; no Conv3d merger or Conv1d fallback. |
| Conv3d direct body | Static grouped NCDHW floating cross-correlation with the same types/promotion/bias/carriers, batch/groups/depthwise, padding/stride/dilation, zero workspace/materialization, checked empty/special/alias/overlap behavior, and half-open output ranges. It has no external Conv3d epilogue; general DAGs fail closed. | `CpuConv3dLowering`, `CpuConv3dIr`, `CpuConv3dEmitter`, `CpuConv3dReferenceKernel`; schema-63 vector/parallel-vector only for eligible dense FLOAT32/FLOAT64 width blocks, scalar otherwise. | Separate complete direct body, including no-epilogue boundary, geometry/binding, artifact/cache, invocation, inventory, validation and retirement. |

All rows keep lowering/selection cold; `CpuPartitionPreparer` declares exact resources,
`CpuPartitionFinalizer` validates assignments then realizes selected artifacts, and
`CpuPreparedExecutable` invokes only prepared work with run-owned range/worker state. Runtime does
not select a route.

## Out of scope

- Common MATMUL/convolution abstraction, universal verifier/oracle/inventory, or direct Conv1d operation.
- Benchmark, five-fork campaign, threshold, speed/JIT claim, packing/materialization/native route.
- Model, Compiler, Training, Prepare, Runtime, public API, capability, workspace, schema,
  dependency, architecture, or Gradle change.
- 0009F2 pooling, 0009F3 attention/batch normalization, and CPU 0009G's broad checkpoint.

## Architecture references and constraints

`ARCHITECTURE.md` is authoritative. CPU analysis/prepare owns admission, lowering, selection,
specialization, and exact resource declaration; finalization follows shared assignment and creates
immutable prepared work; Runtime only invokes it. Keep carrier/layout/geometry/route/worker facts
cold. Hot loops add no allocation, boxing, reflection, string/map dispatch, synchronization, graph
interpretation, or avoidable virtual dispatch; parallelism is outside them. Preserve checked
addresses, aliases/failure-before-write, reuse, and run isolation. Stop on an architecture conflict.

## Package impact

No package is added or moved. Existing `lowering`, `ir`, `codegen.emit`, `reference`, `prepare`,
`executable`, and `cache` owners remain separate. Tests remain in their matching CPU packages. A
selected surrounding Conv1d change stays in its composition owner and cannot create an umbrella
route package/type.

## Affected files and maximum scope

Planning files changed by this task are exactly:

- `docs/planning/backends/cpu/tasks/0009f1-matmul-and-convolution-route-decisions.md`
- `docs/planning/backends/cpu/tasks/0009f-hybrid-route-decisions-for-specialized-compute-families.md`
- `docs/planning/backends/cpu/tasks/0009-portable-generated-coverage-closure-checkpoint.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

If all rows retain current routes and evidence is sound, only those five planning paths change. A
strictly cheaper direct surrounding-composition migration may change at most 13 additional CPU
source/test/evidence/Javadoc paths (18 total) and must retire the complete selected subfamily. A
dense replacement exceeding that limit is not cheaper; retain it rather than staging dual routes.

## Acceptance criteria

1. Records separate complete decisions for MATMUL, Conv1d composition, direct Conv2d, and direct
   Conv3d, including types, layouts, carriers, groups/batches/broadcasts, ranges/parallelism,
   epilogues/composition, workspace, validation, empty/special behavior, aliases/overlap, cold
   binding, artifact/cache/inventory.
2. Retains dense generated compute when direct replacement costs more, is equal, or uncertain;
   selects finite direct Java only for a strictly cheaper complete surrounding replacement. No
   performance result is a gate.
3. For each retained generated hot path, verifies a family-specific clean optimal Java oracle with
   equivalent semantic algorithm, loop/dataflow shape, stores, and avoidable-overhead profile, plus
   proportionate Class-File/decompilation inspection for hidden helpers, allocation, boxing,
   reflection, dispatch, and route leakage. This is not a universal verifier.
4. A migration retires the complete selected subfamily: lowering/selection, binding, invocation,
   artifact/cache/schema/inventory/evidence and call sites. No selected fallback or dual
   maintenance remains; otherwise retain the row.
5. Run focused CPU validation once after executable stabilization. Retained-only decisions reuse
   current focused evidence after independent inspection and do not rerun Java. No broad,
   architecture, conformance, integration, benchmark, or five-fork run absent shared-boundary work.
6. Require a clean implementation context followed by a distinct clean documentation-focused
   context; the latter finalizes affected documentation/Javadoc/glossary impact and planning
   validation without duplicating stable executable tests.

## Tests / validation

For a changed route, run one focused command after stabilization covering the relevant
`CpuMatmulLoweringTest`, `CpuMatmulCandidateSelectorTest`, `CpuMatmulGeneratedKernelTest`,
`CpuMatmulSemanticClosureTest`, `CpuMatmulReferenceTest`, `CpuConv1dCompositionLoweringTest`,
`CpuConv2dLoweringTest`, `CpuConv3dLoweringTest`, `CpuConv2dGeneratedKernelTest`,
`CpuConv3dGeneratedKernelTest`, `CpuConvolutionSemanticClosureTest`, matching evidence tests, and
affected preparation/finalization/executable/cache tests. Include selected-route/retirement,
aliases/canaries/failure-before-write, and inventory evidence as applicable. Retention-only work
inspects current sources, tests, clean Java references, Class-File/decompilation, specialization,
artifact load/generate/persistence, and `generated-coverage-inventory.tsv` without a duplicate
Gradle run. Conv2d/Conv3d include schema-63/0008N1 evidence; MATMUL includes four-realization and
bounded-epilogue evidence; Conv1d proves no independent artifact/inventory route.

The documentation context checks links, headings/fences, terminology/glossary, statuses,
dependencies/frontier, exact scope, final newlines, and new-file whitespace, then runs:

```bash
git diff --check
git status --short -uall
```

## Dependencies and follow-up tasks

- Depends on Ready parent 0009F and completed CPU 0008F, 0008, 0008A, and 0008N1.
- 0009F2 remains Draft and depends on F1; F3 remains Draft and depends on F2; 0009G remains Draft
  and follows F3.
- A strictly cheaper complete migration that exceeds scope becomes one ordered Draft follow-up; it
  cannot conceal an incomplete decision.

## Architecture impact

Expected impact: None. Stop if audit requires a new operation/public API/module dependency/shared
Prepare/Runtime lifecycle/workspace/architecture change.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009F1. Read AGENTS.md,
ARCHITECTURE.md, current architecture plan, Planning Guide, CPU master plan, CPU 0009, parent
0009F, this task, CPU 0008F/0008/0008A/0008N1, and current MATMUL/Conv1d/Conv2d/Conv3d source,
tests, Javadocs, references, Class-File evidence, preparation/finalization/invocation,
artifact/cache/inventory owners. Implement only this specification. Make four independent complete
decisions; retain generated dense compute on equality or uncertainty. Stop on an architecture or
scope conflict. Do not commit, stage, push, use GSD, run Gradle for retention-only work, add a
benchmark gate, or run broad validation absent a shared boundary. Hand the stabilized diff and
evidence to a distinct clean documentation-focused context following documentation-rules.md. Do
not mark Complete until that pass is recorded.
```

## Local decisions

- The task is decision-first and does not presume a migration.
- Conv1d composition is separate because it owns rank-edit orchestration, not independent
  generated compute; it is neither evidence for nor a fallback of direct Conv2d/Conv3d.

## Known limitations

- No benchmark, JIT, or general CPU-backend conclusion is made.
- Conv2d ADD/ADD-RELU and Conv3d no-external-epilogue stay distinct current contracts.

## Validation evidence

Planning creation context read the required architecture, planning, parent/completed tasks,
documentation profiles, and directly relevant current CPU owners/tests/evidence. No Gradle or
executable/Javadoc edit was run. Glossary impact: no change; this planning task introduces no new
reusable project term or altered term boundary. Final planning checks are recorded at finalization.

## Implementation notes

This is a Ready execution contract, not an implementation result. The execution context must add
the four final route decisions, exact inspected source/test/evidence paths, any selected
retirement work, and focused validation evidence here before changing task status.

## Completion summary

Not complete. Completion requires all four decisions, any required complete migration or explicit
retention rationale, the clean documentation-focused pass, and recorded validation evidence.
