# Task 0009F2: Pooling Route Decisions

## Status

Ready

## Goal

Make independent, complete CPU route decisions for exact Pool1d composition, direct Pool2d, and
direct Pool3d. Retain generation whenever a finite direct typed-Java replacement plus verification
and complete generated-route retirement costs more, is equal, or is uncertain. Select direct Java
only when genuinely simpler static composition/window loops are demonstrably strictly cheaper. No
performance gate or benchmark campaign is created.

## Scope

Audit and decide each source-backed row separately. A decision in one row is not evidence for
another; do not create a shared PoolNd abstraction or a universal verifier.

| Route | Current boundary and machinery | Complete decision matrix |
| --- | --- | --- |
| Pool1d composition | Only exact NCW `EXPAND_DIMS(axis 2) -> POOL2D -> SQUEEZE(axis 2)`, with singleton height `(1,1,0,1)`, private single-use virtual rank edits, and current Pool2d max/fixed-count average in the middle. `CpuPool1dCompositionLowering` alone validates topology/descriptors/affine layouts/private memory and delegates to `CpuPool2dLowering`; no Pool1d operation, IR, emitter, reference body, specialization, artifact/cache key, schema projection, inventory row, workspace, or independent invocation exists. | Exact composition, max/average forms, BFLOAT16/FLOAT32/FLOAT64 NCW types and inherited arrays/segments/mixed carriers, rank/layout convention, kernel/stride/dilation/padding/count semantics, empty/special behavior, output-cell ranges/parallelism, workspace/materialization, aliases/overlap, cold binding, and proof that no independent generated route exists to retire. A direct surrounding change cannot manufacture a Pool1d kernel family. |
| Pool2d direct window body | First-class static NCHW `MAX_POOL2D`/`AVERAGE_POOL2D`, BFLOAT16/FLOAT32/FLOAT64 one-input/same-type-output, positive kernel/stride/dilation, non-negative symmetric padding, literal floor/ceil grids, resolved non-negative arbitrary layouts, injective output, and arrays/native-order segments/mixed carriers. `CpuPool2dLowering`, `CpuPool2dIr`, `CpuPool2dEmitter`, and `CpuPool2dReferenceKernel` own a separate generated `DIRECT_SCALAR` output-cell body. | Max excludes padding, gives first logical NaN/winner and negative infinity for all-padding; average has fixed positive kernel-count divisor, conceptual positive-zero padding, FLOAT32 accumulation for BFLOAT16/FLOAT32 and FLOAT64 for FLOAT64, and selected signed-zero/special behavior. Include types/layouts/carriers, all geometry, empty/special behavior, complete half-open output-cell ranges and outer parallelism, zero workspace/materialization, pre-write overlap rejection, cold binding, finalization/invocation, artifact/cache/schema/inventory/evidence. |
| Pool3d direct window body | First-class static NCDHW `MAX_POOL3D`/`AVERAGE_POOL3D`, same floating types/carriers/layout constraints, positive three-axis kernel/stride/dilation, non-negative symmetric padding, literal floor/ceil grids, and injective output. `CpuPool3dLowering`, `CpuPool3dIr`, `CpuPool3dEmitter`, and `CpuPool3dReferenceKernel` independently own rank-five generated `DIRECT_SCALAR` output-cell bodies. | Max has rank-five excluded-padding first-winner/NaN and all-padding negative infinity; average has fixed positive three-dimensional count, conceptual positive-zero padding, selected FLOAT32/FLOAT64 accumulation and signed-zero/special rules. Include all rank-five geometry, types/layouts/carriers, empty/special behavior, ranges/parallelism, zero workspace/materialization, aliases/overlap, cold binding, and complete artifact/cache/schema/inventory/evidence/finalization/invocation paths. |

For retained Pool2d/Pool3d generated hot paths, inspect their rank-specific optimal clean-Java
references and generated bodies. The body must match its oracle's semantic algorithm,
hot-loop/dataflow/store shape, and avoidable-overhead profile; use proportionate representative
Class-File/decompilation checks for helpers, allocation, boxing, reflection, dispatch, cache/route
leakage, and worker management. This is not a universal structural test.

For every migrated row, fully retire the selected generator/lowering/selection, cold binding,
invocation, artifact/cache/schema, inventory, and generated-only evidence paths in the same change.
No selected fallback or dual-maintained route is permitted. If this cannot fit the limit, retain
generation or stop with a bounded follow-up proposal.

## Out of scope

- A Pool1d operation/generated family, PoolNd/shared pooling abstraction, generic window
  interpreter, universal verifier, pooling epilogue fusion, native route, Vector expansion, or
  materialization/workspace policy.
- A benchmark/five-fork campaign, performance/JIT claim, dynamic support, Model/Compiler/Training
  change, public API/capability semantics, shared Prepare/Runtime contract, package/module/
  architecture/Gradle/conformance/integration change, F3, or 0009G.

## Architecture references and constraints

`ARCHITECTURE.md` is authoritative. CPU analysis/prepare owns admission, lowering, cold route
selection and resource declaration; finalization realizes immutable prepared work after assignment;
Runtime invokes it without graph interpretation or route selection. Type, carrier, layout,
geometry, range, and worker facts stay cold. Element/window loops add no allocation, boxing,
reflection, string/map dispatch, synchronization, or avoidable virtual dispatch; parallelism is
outside complete-window loops. Preserve checked addressing, failure-before-write overlap checks,
run isolation, and zero workspace/work contracts. Stop if a complete choice requires architecture,
API, dependency, shared lifecycle, or resource-contract change.

No package/type is added or moved. Existing rank-specific `lowering`, `ir`, `codegen.emit`,
`reference`, `prepare`, `executable`, and `cache` owners remain separate; a direct replacement
stays in its existing family owner.

## Affected files and maximum scope

Planning paths are exactly this task, parent F, CPU 0009, CPU master plan, and roadmap. If all
routes retain sound evidence, only those five paths change. A direct replacement may change at
most 13 additional focused CPU source/test/evidence/Javadoc paths (18 total), including all
retirement work. Exceeding it is not a proven strictly cheaper replacement.

## Acceptance criteria

1. Records independent complete decisions for Pool1d composition, Pool2d, and Pool3d using the
   matrices above; Pool1d never becomes an assumed kernel family.
2. Retains generated Pool2d/Pool3d on greater/equal/uncertain total cost; direct Java is selected
   only with source/test-backed proof of strictly cheaper complete implementation, verification,
   and retirement.
3. Retained generated paths have rank-specific oracle and proportionate Class-File/decompilation
   evidence. A migration removes every selected generated route path with no fallback or dual route.
4. Runs the one focused CPU command once after executable stabilization. Retained-only work
   independently inspects source/tests/evidence and reuses stable Java evidence; it runs no Gradle
   merely to duplicate it. No broad/benchmark/five-fork validation absent a real shared boundary.
5. Uses distinct clean implementation and documentation-focused contexts. The latter finalizes
   Javadocs/docs/glossary impact and planning validation without duplicating stable Java tests.

## Tests / validation

For executable changes, run once after stabilization:

```bash
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool1dCompositionLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool3dLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool2dIrTest --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool3dIrTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuPool2dReferenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuPool3dReferenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPool1dCompositionSemanticClosureTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPool2d3dSemanticClosureTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPool2dGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPool3dGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGeneratorTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageCheckpointTest --tests io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest
```

Record selected-route/retirement, reference/body dataflow, generated bytes/decompilation, no
independent Pool1d artifact, aliases/canaries/failure-before-write where applicable, cold
preparation/finalization/invocation, and cache/schema/inventory truth. The documentation context
uses General and Planning profiles; checks links/headings/anchors/fences/terminology/glossary,
dependencies/frontier, scope, final newlines/new-file whitespace; then runs:

```bash
git diff --check
git status --short -uall
```

## Dependencies and follow-up tasks

Depends on Ready parent 0009F, Complete F1, and CPU 0008G/0008G1. F2 is the sole detailed Ready
frontier. F3 remains a Draft summary depending on F2; do not detail it early. 0009G remains Draft
after F3. Nonessential discovered work is a separate ordered Draft follow-up, never hidden F2 work.

## Architecture impact

Expected impact: None. Stop without implementation edits on an architecture conflict.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009F2. Read AGENTS.md,
ARCHITECTURE.md, current architecture plan, Planning Guide, CPU master plan, CPU 0009, parent
0009F, this task, CPU 0008G/0008G1, and current Pool1d/Pool2d/Pool3d Model, Compiler, and CPU
source, tests, Javadocs, references, Class-File evidence, preparation/finalization/invocation,
artifact/cache/schema/inventory owners. Implement only this specification. Make three independent
complete decisions; retain generation on equality or uncertainty. Stop on architecture/scope
conflict. Do not commit, stage, push, use GSD, add benchmark gates, or run broad validation absent
a shared-boundary change. Hand the stabilized diff and focused evidence to a distinct clean
documentation-focused context following docs/developer-guide/documentation-rules.md. Do not mark
Complete until that pass is recorded.
```

## Local decisions

No migration is presumed. Pool1d composition is not evidence for a Pool2d/Pool3d decision, and
Pool2d/Pool3d remain rank-specific even if their conclusions match.

## Known limitations

No performance, JIT, or whole-backend conclusion is in scope. Current Pool2d/Pool3d direct
window bodies are scalar generated routes; realization expansion is not an F2 objective.

## Validation evidence

Planning-only creation: no Java, Javadoc, Gradle, benchmark, or repository-wide command was run.
This context inspected the required architecture/planning/documentation contracts, CPU 0008G/G1,
the Pool1d composition lowerer, Pool2d/Pool3d lowerers/IRs/emitters/references, lifecycle and
artifact owners, and focused tests. It validates links, headings, fences, statuses, dependencies,
frontier, exact five-path scope, final newline/whitespace, and final Git diff/status checks.

## Implementation notes

Created as the sole detailed F2 frontier after completed F1.

## Completion summary

Planning only: F2 is Ready for a separate clean implementation context. F3 and 0009G remain Draft
summaries. No executable, Javadoc, architecture, build, or test change occurred in this task.

Status: Complete
