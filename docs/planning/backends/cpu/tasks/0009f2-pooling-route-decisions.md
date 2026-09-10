# Task 0009F2: Pooling Route Decisions

## Status

Complete

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

Depends on Ready parent 0009F, Complete F1, and CPU 0008G/0008G1. F3 is the sole next Draft
summary frontier and depends on F2; do not detail it early. 0009G remains Draft after F3.
Nonessential discovered work is a separate ordered Draft follow-up, never hidden F2 work.

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

Implementation draft — the three independent decisions are all **retain current route**:

- **Pool1d composition — retain the exact visible composition.**
  `CpuPool1dCompositionLowering` accepts only the private single-use
  `EXPAND_DIMS(axis 2) -> POOL2D -> SQUEEZE(axis 2)` topology, checks the literal singleton
  height `(1, 1, 0, 1)`, keeps both rank-four values virtual, and delegates a synthetic external
  NCHW boundary to `CpuPool2dLowering`. It thereby inherits the selected Pool2d max/fixed-count
  average rules, BFLOAT16/FLOAT32/FLOAT64 arrays, native-order segments, and mixed carriers;
  checked NCW affine layouts; empty/special-value behavior; complete output-cell ranges and
  caller parallelism; zero workspace/materialization; overlap rejection before writes; cold
  binding; and prepared-artifact invocation. It owns no Pool1d operation, IR, emitter, reference
  body, specialization, schema projection, artifact/cache entry, invocation, or independent
  generated body to retire. A surrounding direct replacement would either duplicate this complete
  Pool2d boundary or manufacture the expressly excluded Pool1d kernel family, so it is not
  demonstrably strictly cheaper. No migration, fallback, or dual route is selected.

- **Pool2d direct max/fixed-count-average — retain the generated family.**
  `CpuPool2dLowering`, `CpuPool2dIr`, `CpuPool2dEmitter`, and
  `CpuPool2dReferenceKernel` independently own static NCHW max and average execution. The cold
  route admits only same-type BFLOAT16/FLOAT32/FLOAT64 resolved non-negative layouts and typed
  array/segment/mixed carriers, recomputes literal floor/ceil geometry and positive fixed
  divisor, preserves empty and special-value rules, rejects input/output overlap before worker
  submission, and declares only direct input/output buffers. The generated body owns a half-open
  output-cell range and its full height-width window; max has excluded-padding first-NaN/first
  winner, positive-zero preference, and all-padding negative infinity, while average has
  conceptual positive-zero padding, FLOAT32 or FLOAT64 accumulation, one divisor, and explicit
  signed-zero handling. Preparation cold-selects scalar or caller-parallel scalar, schema-55
  specialization and carrier pattern; finalization realizes the selected cache artifact after
  assignment; `CpuPreparedExecutable` packs checked geometry and invokes only that prepared
  body. Replacing it requires a finite typed direct body for both kinds, all carriers/layouts and
  range/alias behavior, plus removal of lowering/IR/emitter selection, binding, invocation,
  artifact/cache/schema/inventory, and generated-only evidence. That complete change is greater
  than retention and not strictly cheaper. No migration or selected fallback is chosen.

- **Pool3d direct max/fixed-count-average — retain the generated family.**
  `CpuPool3dLowering`, `CpuPool3dIr`, `CpuPool3dEmitter`, and
  `CpuPool3dReferenceKernel` separately own the NCDHW rank-five route. Their cold facts cover
  three-axis kernel/stride/dilation/padding and literal floor/ceil grids, checked static
  BFLOAT16/FLOAT32/FLOAT64 layouts/carriers, rank-five output-cell ranges/caller parallelism,
  zero workspace/materialization, legal empty work, and failure-before-write overlap validation.
  Generated max traverses depth-height-width in logical order with excluded padding, first
  NaN/winner and all-padding negative infinity; generated average retains the fixed three-axis
  divisor, conceptual positive-zero padding, selected FLOAT32/FLOAT64 accumulation, and
  signed-zero/special-value rules. Schema-56 specialization, artifact/cache identity, cold
  finalization, and prepared typed invocation are distinct from Pool2d. A direct replacement
  would have to duplicate all rank-five bodies and binding/range/alias proof, then completely
  retire this separate lowering/IR/emitter/artifact/cache/schema/inventory/evidence route. It is
  therefore not demonstrably strictly cheaper. No migration or dual route is selected.

## Known limitations

No performance, JIT, or whole-backend conclusion is in scope. Current Pool2d/Pool3d direct
window bodies are scalar generated routes; realization expansion is not an F2 objective.

## Validation evidence

This implementation context read the required architecture/planning/documentation contracts, CPU
0008G/G1, the Model/Compiler pooling contracts, and current Pool1d/Pool2d/Pool3d lowerers, IRs,
emitters, references, lifecycle/artifact owners, Javadocs, focused tests, and coverage inventory.
It found no architecture, API, dependency, build, workspace, materialization, schema, or public
documentation change.

- Pool1d evidence: `CpuPool1dCompositionLoweringTest` proves exact topology, singleton-view
  layouts, private intermediates, schema-55 Pool2d reuse, and byte-identical direct-Pool2d bytes.
  `CpuPool1dCompositionSemanticClosureTest` reconstructs and invokes all 36 current composition
  inventory owners across max/average floor/ceil, the three floating types, and the admitted
  carrier/parallel requests against its independent clean scalar oracle. Source and inventory
  inspection found no Pool1d IR, emitter, reference kernel, specialization, artifact/cache
  family, or standalone generated entry. The repository inventory does contain composition rows,
  but each names `CpuPool2dIr`; they are not a Pool1d generated family.
- Pool2d evidence: `CpuPool2dReferenceKernel` is the rank-specific optimal clean-Java oracle:
  same NCHW ordinal decode, height-width loop order, represented winner/accumulator state, one
  final division/store, and checked direct carrier access. `CpuPool2dReferenceTest`,
  `CpuPool2dLoweringTest`, `CpuPool2dGeneratedKernelTest`, and
  `CpuPool2d3dSemanticClosureTest` cover geometry, types, layouts/carriers, range subdivision,
  special values, empty work, and generated entry execution. The repository-resident
  `CpuPool2dGeneratedKernelTest` parses a representative generated class and checks final,
  field-free, one-entry shape and rejects Synaptik, map, and reflection member references. Source
  review found the rank-specific emitter's direct typed cell/window loop, represented
  winner/accumulator dataflow, and one final store. CPU 0008G's temporary retained
  Class-File/decompilation bundle is historical corroboration only, not a completion dependency.
  Pool2d's class-identity schema is 55.
- Pool3d evidence: `CpuPool3dReferenceKernel` is a separate optimal clean-Java NCDHW oracle with
  the same ordinal decode, depth-height-width order, represented winner/accumulator dataflow,
  one final division/store, and direct typed access. `CpuPool3dReferenceTest`,
  `CpuPool3dLoweringTest`, `CpuPool3dGeneratedKernelTest`, and
  `CpuPool2d3dSemanticClosureTest` cover the independent rank-five semantics and invocation
  matrix. The repository-resident `CpuPool3dGeneratedKernelTest` regenerates all 24 selected
  type/carrier bodies, parses their field-free one-method shape, and checks a representative for
  absent Synaptik, `java.util`, and reflection member references. Source review found the separate
  depth-height-width emitter loop, represented winner/accumulator dataflow, and one final store.
  CPU 0008G1's temporary `javap` reports and scans are historical corroboration only, not a
  completion dependency. Pool3d's class-identity schema is 56, distinct from Pool2d's 55.
- `CpuClassFileKernelGenerator`, `CpuPartitionPreparer`, `CpuPartitionFinalizer`,
  `CpuPreparedExecutable`, and `CpuGeneratedKernelArtifactStore` were traced for both direct
  ranks. Selection, carrier/layout facts, geometry, and worker choices are cold; finalization
  realizes only a compatible selected artifact after resource assignment, and execution binds
  typed carriers and invokes prepared ranges without route selection. Preparation/finalization/
  executable tests cover two selected buffers, zero workspace, artifact truth, scalar/parallel
  complete-cell execution, aliases/canaries, and failure before write.

This documentation-focused context independently inspected the retained source and test owners.
Reuse without rerun is permitted by the Planning Guide and documentation rules because no
executable Java changed after the recorded evidence, and this review found no concrete
discrepancy or cross-check risk. No Gradle, benchmark, five-fork, broad, conformance, or
integration command was run.

No-change conclusions: Javadocs remain accurate because neither public nor implementation
behavior changed. Explanatory and architecture documents need no revision because no boundary or
workflow changed. The glossary needs no entry because this decision records no new or altered
reusable term. Build, architecture, conformance, integration, and executable-test owners need no
change because no source, dependency, capability, or end-to-end behavior changed.

## Implementation notes

All three decisions retain current implementation routes. Consequently this task changes no Java,
test, Javadoc, cache, schema, inventory, artifact, or lifecycle owner; it finalizes only the five
required planning records. The historical temporary evidence locations are not relied on.

## Completion summary

Completed changes: independently finalized the retained-route decisions for Pool1d composition,
Pool2d, and Pool3d, and replaced non-durable Class-File/decompilation claims with bounded
repository-resident source/test evidence. No migration was selected because none is demonstrably
strictly cheaper once implementation, verification, and complete route retirement are included.

Changed paths: this task, parent 0009F, parent CPU 0009, the CPU master plan, and the roadmap.

Validation: inspected current lowerers, IRs, emitters, clean-Java references, preparer/finalizer/
executable/artifact owners, focused tests, and generated-coverage inventory; reviewed Javadocs,
explanatory documentation, glossary, architecture, build, conformance, and integration impact;
reused recorded stable Java evidence without rerunning Gradle; and passed final planning checks
(`git diff --check` and `git status --short -uall`).

Unresolved/follow-up: no unresolved issue. 0009F and CPU 0009 remain Ready and incomplete. F3 is
the sole next Draft summary frontier; 0009G remains later Draft and is not detailed here.

Status: Complete
