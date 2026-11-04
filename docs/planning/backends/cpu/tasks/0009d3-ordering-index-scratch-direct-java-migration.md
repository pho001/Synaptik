# Task 0009D3: Ordering, Index, and Scratch Direct-Java Route Decision

## Reader context

This task records retaining generated SORT, ARGSORT, and TOP_K because replacing the complete typed
family would increase implementation and verification work. `CpuOrderingLowering` and
`CpuOrderingIr` hold structural facts; `CpuClassFileKernelGenerator` selects `CpuOrderingEmitter`.
The 192-row inventory is structural coverage evidence, while the independent scalar reference is the
semantic oracle for ordering results, not a Runtime interpreter.

Here, CPU Prepare selects carrier, layout, and scalar or parallel-scalar entries before Runtime
invokes the bound entry through `CpuPreparedExecutable`. Workers own complete axis slices; each
active range owns disjoint merge scratch. That run scratch is not persistent model state. Catalogue
vector labels do not establish vector execution or authorize production defaults.

Current-topic navigation (not a replacement for the historical record):

- [current stable ordering and paired Top-K results](../../../../reference/tensor/sort-topk.md#top-k-values-and-indices) — Explains value/index pairing, stable logical indices, and sorted versus original-index output.
- [backend ownership versus implementation route](../../../../design/notes/cpu-kernel-strategy.md#strategy) — Separates Planning ownership, CPU route choice, finalization, and Runtime invocation.

## Status

Complete

## Goal

Decide and, only if favorable, migrate the complete [CPU](../../../../glossary.md#term-cpu) ordering family from generated entries to [finite](../../../../glossary.md#term-finite) direct typed Java. The family is exactly stable `SORT`, `ARGSORT`, and two-output `TOP_K`; it shares logical-[axis index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) ordering and two-region [INT64](../../../../glossary.md#data-type--datatype) merge scratch. Direct Java is authorized only when replacing the complete cohesive family lowers total implementation and verification work. No performance benefit or benchmark gate is required.

## [Scope](../../../../glossary.md#term-scope)

First inspect the existing `CpuOrderingLowering`, `CpuOrderingIr`, `CpuOrderingEmitter`, generator selection, preparation/finalization, prepared invocation, [scalar](../../../../glossary.md#term-scalar) reference, semantic closure, generated inventory, evidence ledger, and applicable [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance). Current support is this complete [matrix](../../../../glossary.md#term-matrix-meanings):

| [Dimension](../../../../glossary.md#dimension) | Supported forms |
|---|---|
| Family and output roles | `SORT`: values; `ARGSORT`: [INT64](../../../../glossary.md#data-type--datatype) logical-[axis](../../../../glossary.md#term-axis) indices; `TOP_K`: values then INT64 logical-axis indices. |
| Represented input/value type | [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), `INT64`, [`BOOL`](../../../../glossary.md#data-type--datatype). |
| Direction | Ascending or descending for `SORT`/`ARGSORT`; smallest or largest for `TOP_K`. Floating NaNs remain last in either direction. |
| Selection and order | `SORT`/`ARGSORT` select the complete axis and are sorted. `TOP_K` admits `0 <= k <= axisExtent`, with `sorted` true or false. |
| [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) | Fully static resolved input shape, normalized valid axis, non-negative offset/[strides](../../../../glossary.md#term-stride); output shapes/roles exactly match [Model](../../../../glossary.md#term-model); outputs are [injective](../../../../glossary.md#term-injective). Read input may use general/broadcast layout. |
| Carriers and access | Typed [heap](../../../../glossary.md#term-heap) arrays and [`MemorySegment`](../../../../glossary.md#term-memorysegment) boundaries, including mixed carriers; dense heap-array INT addressing or general long-address odometer routes; scalar or [slice](../../../../glossary.md#slice)-parallel-scalar selected cold. |
| Range and scratch | A worker receives complete independent axis slices only. Every active range owns two disjoint axis-[extent](../../../../glossary.md#term-extent) INT64 merge regions (`2 * axisExtent * Long.BYTES`); empty axis, empty outer domain, or `TOP_K(k=0)` has zero work and zero scratch. |

The checked generated inventory is 192 ordering rows: 48 semantic bases (six represented types × two directions × `SORT`/`ARGSORT`, plus six types × two directions × two `TOP_K.sorted` forms), each across the four currently catalogued [carrier](../../../../glossary.md#term-storage-carrier)/layout request forms. Exact current rejections include wrong input/output [arity](../../../../glossary.md#term-arity) or output roles, wrong output type or shape, invalid/non-[normalized axis](../../../../glossary.md#normalized-axis), `k < 0` or `k > axisExtent`, non-static or unresolved layouts, negative layout offset or stride, non-injective output layout, duplicate logical boundaries, incompatible carriers, and any physical input/output or output/output overlap at cold [binding](../../../../glossary.md#term-binding). Do not extend this matrix.

The route-decision gate compares a complete [finite](../../../../glossary.md#term-finite) direct design—including typed carrier methods, dense/general address forms, stable comparison, direct merge/index scratch, sorted and unsorted `TOP_K`, packed geometry, binding overlap checks, and all replacement tests—with the present generated implementation and evidence. Select direct Java only if it is genuinely less work to implement and verify; otherwise explicitly retain generation without executable changes.

`legacy/pre-rewrite` may be inspected read-only only for capability, observable behavior, algorithm, or test ideas. Do not copy source, package structure, dependencies, [runtime](../../../../glossary.md#term-runtime) coupling, or shortcuts.

## Out of [scope](../../../../glossary.md#term-scope)

[Gather](../../../../glossary.md#gather), gather-[elements](../../../../glossary.md#term-element), [gather-ND](../../../../glossary.md#gather-nd), one-hot, scatter, fold, arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum), aggregate, [scan](../../../../glossary.md#term-cumulative-scan), [normalization](../../../../glossary.md#term-normalization), [loss](../../../../glossary.md#loss), [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), [convolution](../../../../glossary.md#term-channels-first-convolution-layers), a universal structural verifier, public [API](../../../../glossary.md#term-application-programming-interface-api) or [Model](../../../../glossary.md#term-model) semantics, architecture/[module](../../../../glossary.md#term-module)-boundary changes, a permanent direct/generated duplicate route, and a benchmark or performance-ratio gate. D4 remains summary-only: create no D4 specification.

## Architecture references and constraints

[`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) is authoritative. Read it with the [current architecture index](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md), [CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), parent [CPU 0009D](0009d-aggregate-scan-ordering-fold-structural-oracle-closure.md), and completed [D1](0009d1-aggregate-structural-oracle-closure.md), [D1A](0009d1a-aggregate-segment-layout-prologue-hygiene.md), and [D2](0009d2-scan-route-decision-and-possible-direct-java-migration.md).

[CPU](../../../../glossary.md#term-cpu) [Prepare](../../../../glossary.md#prepare), not [Planning](../../../../glossary.md#term-planning) or [Runtime](../../../../glossary.md#term-runtime), selects one concrete typed route. Preparation/finalization retain [layout](../../../../glossary.md#layout), [carrier](../../../../glossary.md#term-storage-carrier), geometry, range, and workspace facts; Runtime invokes prepared work only. Direct Java must preserve [immutable](../../../../glossary.md#term-immutable) prepared recipes, cold overlap [validation](../../../../glossary.md#term-validation) and route selection, worker-owned complete slices, and [run](../../../../glossary.md#term-run)-owned disjoint scratch. Its typed hot loops must be equivalent in algorithm and avoidable-overhead [shape](../../../../glossary.md#shape) to optimal clean Java: no allocation, boxing, reflection, string/map [dispatch](../../../../glossary.md#term-dispatch), [synchronization](../../../../glossary.md#term-synchronization), or avoidable virtual/interface dispatch per [element](../../../../glossary.md#term-element). No [Operation](../../../../glossary.md#operation) or CompiledNode reaches Runtime hot loops. Do not claim performance parity.

## Package impact

The decision-only result changes [planning](../../../../glossary.md#term-planning) records only. A direct result remains in
`io.github.pho001.synaptik.backend.cpu.internal`: [lowering](../../../../glossary.md#lowering)/[IR](../../../../glossary.md#term-intermediate-representation-ir) retain cold structural facts,
[prepare](../../../../glossary.md#prepare)/finalization select and bind the route, executable owns its prepared invocation, and a
new [CPU](../../../../glossary.md#term-cpu)-private direct implementation owns typed ordering loops. No public package, [Model](../../../../glossary.md#term-model)
semantic contract, shared Prepare/[Runtime](../../../../glossary.md#term-runtime) contract, or [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency) changes.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

Decision-only: this task, parent 0009D, parent 0009, [CPU](../../../../glossary.md#term-cpu) master plan, and roadmap. A direct migration may change only CPU-private ordering lowerer/[IR](../../../../glossary.md#term-intermediate-representation-ir) route plumbing, [prepare](../../../../glossary.md#prepare)/finalize/executable invocation [owner](../../../../glossary.md#term-owner), direct typed implementation, generated-route owners, and focused CPU tests/resources. Retire `CpuOrderingEmitter` selection and ordering inventory/evidence/disposition rows only after replacement passes and no selected route references them. Do not pre-emptively remove a shared generator or evidence owner used by another family. Stop and replan if public, shared, or architectural boundaries are crossed.

A direct outcome may modify at most 20 CPU production/test/resource paths in addition to these five
[planning](../../../../glossary.md#term-planning) records. If the complete-family route cannot fit that bound, retain generation or stop and
propose a successor rather than leaving a partial direct/generated [topology](../../../../glossary.md#term-topology).

## Acceptance criteria

- The gate records direct Java or retained generation for the complete ordering family, with a concrete total implementation/evidence comparison.
- A direct route, if authorized, covers every [matrix](../../../../glossary.md#term-matrix-meanings) row and exact rejection above; preserves stable ties, [NaN](../../../../glossary.md#term-nan)-last ordering, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) order and [represented bits](../../../../glossary.md#term-represented-bits), logical-[axis](../../../../glossary.md#term-axis) indices, sorted/unsorted `TOP_K`, empty/zero-`k` behavior, complete-[slice](../../../../glossary.md#slice) [ownership](../../../../glossary.md#term-ownership-meanings), overlap rejection, and disjoint scratch.
- Direct route selection and typed array/segment, dense/general, [scalar](../../../../glossary.md#term-scalar)/parallel-scalar [dispatch](../../../../glossary.md#term-dispatch) are cold; selected hot loops are [finite](../../../../glossary.md#term-finite), typed, and dispatch-free.
- Retire generated routing, inventory, ledger/disposition, and structural evidence only after focused replacement semantics, invocation, retirement, and no-selected-reference checks pass. A retained outcome leaves them intact and records why.
- Review/update affected [CPU](../../../../glossary.md#term-cpu) [Javadoc](../../../../glossary.md#term-javadoc) and documentation under the documentation rules; record no-change conclusions where behavior/contracts do not change. Add no benchmark gate.
- D4 stays summary-only.

## Tests / [validation](../../../../glossary.md#term-validation)

Before deciding, [run](../../../../glossary.md#term-run) or reuse focused ordering [lowering](../../../../glossary.md#lowering)/[IR](../../../../glossary.md#term-intermediate-representation-ir), generator/semantic closure, preparer/finalizer, invocation/overlap/workspace, inventory/evidence, and relevant [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) evidence. At minimum, start from:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOrderingLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuOrderingGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuIndexingOrderingSemanticClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageCheckpointTest
```

A direct outcome adds focused direct route-selection, all-type semantic, [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout),
[slice](../../../../glossary.md#slice)-range/scratch/overlap, exact rejection, and generated-retirement checks, then runs the [CPU](../../../../glossary.md#term-cpu)
[module](../../../../glossary.md#term-module) suite once after executable edits. Its documentation-focused pass runs
`./gradlew :backends:cpu:javadoc` after final [Javadoc](../../../../glossary.md#term-javadoc) changes, checks glossary impact, and reuses
successful Java evidence unless executable behavior changes. Inspect direct source/call sites for
hot-loop hygiene. A retained outcome runs no benchmark; validate the decision record,
links/anchors/fences, status/[dependency](../../../../glossary.md#term-dependency) order, exact changed-path [scope](../../../../glossary.md#term-scope), and `git diff --check`.

## Dependencies and follow-up tasks

D1, D1A, D2, and D3 are Complete. D4 fold is Draft and the next summary-only frontier; parent
0009D remains [Ready](../../../../glossary.md#term-ready) until D4 is resolved, then [CPU](../../../../glossary.md#term-cpu) 0009E follows. No D4 specification may be
created here.

## Architecture impact

Expected impact: None. Stop and report any required public, [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module)-boundary, or
architecture-contract change.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009D3. Read AGENTS.md, ARCHITECTURE.md, current architecture plan, Planning Guide, documentation rules and General/Planning profiles, CPU master plan, CPU 0009/D/D1/D1A/D2, this task, and the complete current ordering lowering/IR/emitter/generator/preparer/finalizer/executable/reference/test/inventory owners. First perform the bounded complete-family route-decision gate. Migrate SORT, ARGSORT, and TOP_K together to finite direct typed Java only if doing so genuinely reduces total implementation and verification work; otherwise retain generation explicitly. Preserve the exact supported matrix, cold selection, complete-slice ownership, stable semantics, overlap validation, and run-owned scratch. Retire generated route/inventory/evidence only after replacement tests and no-selected-reference checks pass. Do not implement indexing outside this family, D4, architecture changes, a benchmark gate, commit, or push. Use legacy/pre-rewrite only as read-only behavioral/test evidence. Hand executable changes and evidence to a separate clean documentation-focused context.
```

## Local decisions

“Ordering/index/scratch [topology](../../../../glossary.md#term-topology)” means only the current `CpuOrderingIr` family, not the separate `CpuIndexingIr` [gather](../../../../glossary.md#gather)/one-hot family or scatter. The two [INT64](../../../../glossary.md#data-type--datatype) merge regions are part of one cohesive ordering route and must be decided with its comparison, output, and invocation topology. The present 192-row generated inventory is evidence to inspect, not a requirement to recreate a universal structural [oracle](../../../../glossary.md#term-test-oracle) or benchmark gate.

Retain the generated ordering route. A complete [finite](../../../../glossary.md#term-finite) direct-Java replacement would need all six
represented types across `SORT`, `ARGSORT`, and both sorted forms of `TOP_K`, with ascending/
descending selection; typed array, [`MemorySegment`](../../../../glossary.md#term-memorysegment), and mixed-[carrier](../../../../glossary.md#term-storage-carrier) [bindings](../../../../glossary.md#term-binding); dense and general
addressing; [scalar](../../../../glossary.md#term-scalar) and parallel-scalar entries; stable floating comparison ([NaN](../../../../glossary.md#term-nan)-last in either
direction and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)/[represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) preservation); logical-[axis](../../../../glossary.md#term-axis) indices; unsorted `TOP_K`;
cold overlap [validation](../../../../glossary.md#term-validation); complete-[slice](../../../../glossary.md#slice) [ownership](../../../../glossary.md#term-ownership-meanings); and two disjoint INT64 merge-scratch regions
per active range. It would also require new route-selection, semantics, invocation, hygiene,
inventory, generated-route replacement, and retirement proof. That is not lower total
implementation or verification work than the established 192-row generated specializations,
which already embody those typed bodies and cold facts. This is a retention decision, not a
performance claim.

## Known limitations

No new performance claim follows from either decision. Existing performance evidence remains historical. Direct Java is not presumed to be selected.

## [Validation](../../../../glossary.md#term-validation) evidence

The focused ordering route command specified above passed on 2026-09-09: 66 tests, zero failures,
zero errors, and zero skips. It covers [lowering](../../../../glossary.md#lowering)/[IR](../../../../glossary.md#term-intermediate-representation-ir) [admission](../../../../glossary.md#term-admission) and exact rejection, generated
all-type ordering semantics against the independent [scalar](../../../../glossary.md#term-scalar) reference, stable ties/[NaN](../../../../glossary.md#term-nan)-last/
[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) behavior, logical indices and both `TOP_K` orders, dense/general and [carrier](../../../../glossary.md#term-storage-carrier) forms,
complete-[slice](../../../../glossary.md#slice) ranges, exact scratch, cold [binding](../../../../glossary.md#term-binding), preparation/finalization, and generated
inventory coverage. Repository inspection confirms that `CpuOrderingLowering`, `CpuOrderingIr`,
`CpuOrderingEmitter`, `CpuClassFileKernelGenerator`, `CpuPartitionPreparer`,
`CpuPartitionFinalizer`, and `CpuPreparedExecutable` still select and invoke that route; no
generated [owner](../../../../glossary.md#term-owner) is retired. Documentation-focused context `/root/cpu_0009d3_docs` independently
inspected those route owners, focused tests/results, 192-row inventory/evidence owners, and the
final [planning](../../../../glossary.md#term-planning) diff; it reused the successful executable evidence because this decision changes
no executable Java.

## Implementation notes

No executable Java changed. The [scalar](../../../../glossary.md#term-scalar) reference remains an independent semantic [oracle](../../../../glossary.md#term-test-oracle), not a
[Runtime](../../../../glossary.md#term-runtime) interpreter. `CpuClassFileKernelGenerator` continues to select `CpuOrderingEmitter`; the
[prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable) continues to invoke the bound typed entry. Current generated rows use scalar
or parallel-scalar selection; “[vector](../../../../glossary.md#term-vector)” request labels in the catalogue do not authorize a vector
ordering hot loop.

## Completion summary

- Completed changes: Recorded the complete ordering route-decision gate and retained the existing
  generated `SORT`/`ARGSORT`/`TOP_K` route; no production, test, generated artifact, inventory,
  or Javadoc path changed.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Files changed or created: This task and synchronized [CPU](../../../../glossary.md#term-cpu) 0009 [planning](../../../../glossary.md#term-planning) records only.
- Tests and validation: Reused the implementation context's seven-class focused CPU command,
  which passed 66 tests with zero failures, errors, or skips. This documentation-focused context
  checked local Markdown links, headings used as anchors, fences, terminology,
  task/dependency/status synchronization, exact changed-path scope, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [dependency glossary](../../../../glossary.md#term-dependency); [synchronization glossary](../../../../glossary.md#term-synchronization); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`. It
  did not rerun Java tests because executable behavior did not change.
- Documentation-agent review: Clean documentation-focused context `/root/cpu_0009d3_docs`
  finalized these planning records using the General and Planning documentation profiles.
- Documentation impact: Planning records are revised; public/API Javadoc, guides, glossary,
  architecture/ADR, Gradle, conformance/integration, and other modules have no change because
  public behavior, Java contracts, architecture, and reusable terminology are unchanged.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>
- Javadoc review: No change; no Java contract or implementation changed, so CPU Javadoc was not
  generated.
- Glossary impact: No change; ordering, logical-axis index, and scratch retain their established
  meanings and boundaries.<span data-glossary-navigation="true"> ([axis index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Unresolved issues: None.
- Follow-up required: D4 fold is the next summary-only frontier; do not create its detailed
  specification until it becomes actionable.

Status: Complete
