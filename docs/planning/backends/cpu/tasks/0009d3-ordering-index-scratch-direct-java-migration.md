# Task 0009D3: Ordering, Index, and Scratch Direct-Java Route Decision

## Status

Complete

## Goal

Decide and, only if favorable, migrate the complete CPU ordering family from generated entries to finite direct typed Java. The family is exactly stable `SORT`, `ARGSORT`, and two-output `TOP_K`; it shares logical-axis index ordering and two-region INT64 merge scratch. Direct Java is authorized only when replacing the complete cohesive family lowers total implementation and verification work. No performance benefit or benchmark gate is required.

## Scope

First inspect the existing `CpuOrderingLowering`, `CpuOrderingIr`, `CpuOrderingEmitter`, generator selection, preparation/finalization, prepared invocation, scalar reference, semantic closure, generated inventory, evidence ledger, and applicable backend conformance. Current support is this complete matrix:

| Dimension | Supported forms |
|---|---|
| Family and output roles | `SORT`: values; `ARGSORT`: INT64 logical-axis indices; `TOP_K`: values then INT64 logical-axis indices. |
| Represented input/value type | `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, `BOOL`. |
| Direction | Ascending or descending for `SORT`/`ARGSORT`; smallest or largest for `TOP_K`. Floating NaNs remain last in either direction. |
| Selection and order | `SORT`/`ARGSORT` select the complete axis and are sorted. `TOP_K` admits `0 <= k <= axisExtent`, with `sorted` true or false. |
| Shape/layout | Fully static resolved input shape, normalized valid axis, non-negative offset/strides; output shapes/roles exactly match Model; outputs are injective. Read input may use general/broadcast layout. |
| Carriers and access | Typed heap arrays and `MemorySegment` boundaries, including mixed carriers; dense heap-array INT addressing or general long-address odometer routes; scalar or slice-parallel-scalar selected cold. |
| Range and scratch | A worker receives complete independent axis slices only. Every active range owns two disjoint axis-extent INT64 merge regions (`2 * axisExtent * Long.BYTES`); empty axis, empty outer domain, or `TOP_K(k=0)` has zero work and zero scratch. |

The checked generated inventory is 192 ordering rows: 48 semantic bases (six represented types × two directions × `SORT`/`ARGSORT`, plus six types × two directions × two `TOP_K.sorted` forms), each across the four currently catalogued carrier/layout request forms. Exact current rejections include wrong input/output arity or output roles, wrong output type or shape, invalid/non-normalized axis, `k < 0` or `k > axisExtent`, non-static or unresolved layouts, negative layout offset or stride, non-injective output layout, duplicate logical boundaries, incompatible carriers, and any physical input/output or output/output overlap at cold binding. Do not extend this matrix.

The route-decision gate compares a complete finite direct design—including typed carrier methods, dense/general address forms, stable comparison, direct merge/index scratch, sorted and unsorted `TOP_K`, packed geometry, binding overlap checks, and all replacement tests—with the present generated implementation and evidence. Select direct Java only if it is genuinely less work to implement and verify; otherwise explicitly retain generation without executable changes.

`legacy/pre-rewrite` may be inspected read-only only for capability, observable behavior, algorithm, or test ideas. Do not copy source, package structure, dependencies, runtime coupling, or shortcuts.

## Out of scope

Gather, gather-elements, gather-ND, one-hot, scatter, fold, arg-extrema, aggregate, scan, normalization, loss, MATMUL, convolution, a universal structural verifier, public API or Model semantics, architecture/module-boundary changes, a permanent direct/generated duplicate route, and a benchmark or performance-ratio gate. D4 remains summary-only: create no D4 specification.

## Architecture references and constraints

[`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) is authoritative. Read it with the [current architecture index](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md), [CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), parent [CPU 0009D](0009d-aggregate-scan-ordering-fold-structural-oracle-closure.md), and completed [D1](0009d1-aggregate-structural-oracle-closure.md), [D1A](0009d1a-aggregate-segment-layout-prologue-hygiene.md), and [D2](0009d2-scan-route-decision-and-possible-direct-java-migration.md).

CPU Prepare, not Planning or Runtime, selects one concrete typed route. Preparation/finalization retain layout, carrier, geometry, range, and workspace facts; Runtime invokes prepared work only. Direct Java must preserve immutable prepared recipes, cold overlap validation and route selection, worker-owned complete slices, and run-owned disjoint scratch. Its typed hot loops must be equivalent in algorithm and avoidable-overhead shape to optimal clean Java: no allocation, boxing, reflection, string/map dispatch, synchronization, or avoidable virtual/interface dispatch per element. No Operation or CompiledNode reaches Runtime hot loops. Do not claim performance parity.

## Package impact

The decision-only result changes planning records only. A direct result remains in
`io.github.pho001.synaptik.backend.cpu.internal`: lowering/IR retain cold structural facts,
prepare/finalization select and bind the route, executable owns its prepared invocation, and a
new CPU-private direct implementation owns typed ordering loops. No public package, Model
semantic contract, shared Prepare/Runtime contract, or module dependency changes.

## Affected files and maximum scope

Decision-only: this task, parent 0009D, parent 0009, CPU master plan, and roadmap. A direct migration may change only CPU-private ordering lowerer/IR route plumbing, prepare/finalize/executable invocation owner, direct typed implementation, generated-route owners, and focused CPU tests/resources. Retire `CpuOrderingEmitter` selection and ordering inventory/evidence/disposition rows only after replacement passes and no selected route references them. Do not pre-emptively remove a shared generator or evidence owner used by another family. Stop and replan if public, shared, or architectural boundaries are crossed.

A direct outcome may modify at most 20 CPU production/test/resource paths in addition to these five
planning records. If the complete-family route cannot fit that bound, retain generation or stop and
propose a successor rather than leaving a partial direct/generated topology.

## Acceptance criteria

- The gate records direct Java or retained generation for the complete ordering family, with a concrete total implementation/evidence comparison.
- A direct route, if authorized, covers every matrix row and exact rejection above; preserves stable ties, NaN-last ordering, signed-zero order and represented bits, logical-axis indices, sorted/unsorted `TOP_K`, empty/zero-`k` behavior, complete-slice ownership, overlap rejection, and disjoint scratch.
- Direct route selection and typed array/segment, dense/general, scalar/parallel-scalar dispatch are cold; selected hot loops are finite, typed, and dispatch-free.
- Retire generated routing, inventory, ledger/disposition, and structural evidence only after focused replacement semantics, invocation, retirement, and no-selected-reference checks pass. A retained outcome leaves them intact and records why.
- Review/update affected CPU Javadoc and documentation under the documentation rules; record no-change conclusions where behavior/contracts do not change. Add no benchmark gate.
- D4 stays summary-only.

## Tests / validation

Before deciding, run or reuse focused ordering lowering/IR, generator/semantic closure, preparer/finalizer, invocation/overlap/workspace, inventory/evidence, and relevant backend conformance evidence. At minimum, start from:

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

A direct outcome adds focused direct route-selection, all-type semantic, carrier/layout,
slice-range/scratch/overlap, exact rejection, and generated-retirement checks, then runs the CPU
module suite once after executable edits. Its documentation-focused pass runs
`./gradlew :backends:cpu:javadoc` after final Javadoc changes, checks glossary impact, and reuses
successful Java evidence unless executable behavior changes. Inspect direct source/call sites for
hot-loop hygiene. A retained outcome runs no benchmark; validate the decision record,
links/anchors/fences, status/dependency order, exact changed-path scope, and `git diff --check`.

## Dependencies and follow-up tasks

D1, D1A, D2, and D3 are Complete. D4 fold is Draft and the next summary-only frontier; parent
0009D remains Ready until D4 is resolved, then CPU 0009E follows. No D4 specification may be
created here.

## Architecture impact

Expected impact: None. Stop and report any required public, dependency, module-boundary, or
architecture-contract change.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009D3. Read AGENTS.md, ARCHITECTURE.md, current architecture plan, Planning Guide, documentation rules and General/Planning profiles, CPU master plan, CPU 0009/D/D1/D1A/D2, this task, and the complete current ordering lowering/IR/emitter/generator/preparer/finalizer/executable/reference/test/inventory owners. First perform the bounded complete-family route-decision gate. Migrate SORT, ARGSORT, and TOP_K together to finite direct typed Java only if doing so genuinely reduces total implementation and verification work; otherwise retain generation explicitly. Preserve the exact supported matrix, cold selection, complete-slice ownership, stable semantics, overlap validation, and run-owned scratch. Retire generated route/inventory/evidence only after replacement tests and no-selected-reference checks pass. Do not implement indexing outside this family, D4, architecture changes, a benchmark gate, commit, or push. Use legacy/pre-rewrite only as read-only behavioral/test evidence. Hand executable changes and evidence to a separate clean documentation-focused context.
```

## Local decisions

“Ordering/index/scratch topology” means only the current `CpuOrderingIr` family, not the separate `CpuIndexingIr` gather/one-hot family or scatter. The two INT64 merge regions are part of one cohesive ordering route and must be decided with its comparison, output, and invocation topology. The present 192-row generated inventory is evidence to inspect, not a requirement to recreate a universal structural oracle or benchmark gate.

Retain the generated ordering route. A complete finite direct-Java replacement would need all six
represented types across `SORT`, `ARGSORT`, and both sorted forms of `TOP_K`, with ascending/
descending selection; typed array, `MemorySegment`, and mixed-carrier bindings; dense and general
addressing; scalar and parallel-scalar entries; stable floating comparison (NaN-last in either
direction and signed-zero/represented-bit preservation); logical-axis indices; unsorted `TOP_K`;
cold overlap validation; complete-slice ownership; and two disjoint INT64 merge-scratch regions
per active range. It would also require new route-selection, semantics, invocation, hygiene,
inventory, generated-route replacement, and retirement proof. That is not lower total
implementation or verification work than the established 192-row generated specializations,
which already embody those typed bodies and cold facts. This is a retention decision, not a
performance claim.

## Known limitations

No new performance claim follows from either decision. Existing performance evidence remains historical. Direct Java is not presumed to be selected.

## Validation evidence

The focused ordering route command specified above passed on 2026-09-09: 66 tests, zero failures,
zero errors, and zero skips. It covers lowering/IR admission and exact rejection, generated
all-type ordering semantics against the independent scalar reference, stable ties/NaN-last/
signed-zero behavior, logical indices and both `TOP_K` orders, dense/general and carrier forms,
complete-slice ranges, exact scratch, cold binding, preparation/finalization, and generated
inventory coverage. Repository inspection confirms that `CpuOrderingLowering`, `CpuOrderingIr`,
`CpuOrderingEmitter`, `CpuClassFileKernelGenerator`, `CpuPartitionPreparer`,
`CpuPartitionFinalizer`, and `CpuPreparedExecutable` still select and invoke that route; no
generated owner is retired. Documentation-focused context `/root/cpu_0009d3_docs` independently
inspected those route owners, focused tests/results, 192-row inventory/evidence owners, and the
final planning diff; it reused the successful executable evidence because this decision changes
no executable Java.

## Implementation notes

No executable Java changed. The scalar reference remains an independent semantic oracle, not a
Runtime interpreter. `CpuClassFileKernelGenerator` continues to select `CpuOrderingEmitter`; the
prepared executable continues to invoke the bound typed entry. Current generated rows use scalar
or parallel-scalar selection; “vector” request labels in the catalogue do not authorize a vector
ordering hot loop.

## Completion summary

- Completed changes: Recorded the complete ordering route-decision gate and retained the existing
  generated `SORT`/`ARGSORT`/`TOP_K` route; no production, test, generated artifact, inventory,
  or Javadoc path changed.
- Files changed or created: This task and synchronized CPU 0009 planning records only.
- Tests and validation: Reused the implementation context's seven-class focused CPU command,
  which passed 66 tests with zero failures, errors, or skips. This documentation-focused context
  checked local Markdown links, headings used as anchors, fences, terminology,
  task/dependency/status synchronization, exact changed-path scope, and `git diff --check`. It
  did not rerun Java tests because executable behavior did not change.
- Documentation-agent review: Clean documentation-focused context `/root/cpu_0009d3_docs`
  finalized these planning records using the General and Planning documentation profiles.
- Documentation impact: Planning records are revised; public/API Javadoc, guides, glossary,
  architecture/ADR, Gradle, conformance/integration, and other modules have no change because
  public behavior, Java contracts, architecture, and reusable terminology are unchanged.
- Javadoc review: No change; no Java contract or implementation changed, so CPU Javadoc was not
  generated.
- Glossary impact: No change; ordering, logical-axis index, and scratch retain their established
  meanings and boundaries.
- Unresolved issues: None.
- Follow-up required: D4 fold is the next summary-only frontier; do not create its detailed
  specification until it becomes actionable.

Status: Complete
