# Task 0009D4: Fold and Window Route Decision and Possible Direct-Java Migration

## Status

Complete

## Goal

Decide whether [finite](../../../../glossary.md#term-finite) direct typed Java lowers the total implementation and verification burden
for the complete currently supported [CPU](../../../../glossary.md#term-cpu) overlap-fold family. Migrate only if it does; otherwise
retain the generated route explicitly. This is a route-complexity decision, not a performance
claim or benchmark gate.

## [Scope](../../../../glossary.md#term-scope)

The cohesive all-or-nothing family is exactly one fully static, resolved-[layout](../../../../glossary.md#layout),
zero-initialized overlap-add [occurrence](../../../../glossary.md#term-occurrence) of `FOLD_AXIS` or `FOLD2D`:

| [Dimension](../../../../glossary.md#dimension) | Exact supported forms |
|---|---|
| Operations and types | `FOLD_AXIS`: [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype). `FOLD2D`: `FLOAT64`, `FLOAT32`, `BFLOAT16`. [`BOOL`](../../../../glossary.md#data-type--datatype) is never admitted. |
| [Shapes](../../../../glossary.md#shape) and mapping | `FOLD_AXIS` removes its final positive window dimension, restores the normalized [target](../../../../glossary.md#term-target) [axis](../../../../glossary.md#term-axis) to `outputSize`, and has exact count/step geometry. `FOLD2D` has [rank](../../../../glossary.md#term-rank)-three columns input, rank-four [canonical](../../../../glossary.md#term-canonical-form) [NCHW](../../../../glossary.md#term-nchw) output, exact `Fold2dAttrs.outputShape`, and checked [kernel](../../../../glossary.md#kernel), [stride](../../../../glossary.md#term-stride), symmetric [padding](../../../../glossary.md#padding), [dilation](../../../../glossary.md#term-dilation), floor/ceil grid mapping. |
| Layout and carriers | Input has non-negative offset/strides and may be dense or general, including zero-stride read axes. Output is distinct, non-overlapping, [injective](../../../../glossary.md#term-injective), and resolved. Each boundary is its typed [heap](../../../../glossary.md#term-heap) array or [`MemorySegment`](../../../../glossary.md#term-memorysegment), including all four combinations and cold-bound offsets. |
| Addition and range | Every selected output ordinal in `[start,end)` starts at represented [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) then receives canonical input-[row-major](../../../../glossary.md#term-row-major) additions. `BFLOAT16` rounds after every addition; `INT32`/`INT64` wrap. Uncovered outputs stay positive zero. Independent output ranges require no workspace. |
| Current generated inventory | 32 checked rows: 20 `FOLD_AXIS` (five types × four [carrier](../../../../glossary.md#term-storage-carrier) combinations) and 12 `FOLD2D` (three types × four carrier combinations), [sealed](../../../../glossary.md#term-sealed) in the wider 160-[owner](../../../../glossary.md#term-owner) [SHA-256](../../../../glossary.md#term-sha-256)-bound ordinary movement/fold projection. |

Inspect the complete fold lowerer, `CpuFoldIr`, emitter, generator selection, preparation,
finalization, [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable) invocation, [scalar](../../../../glossary.md#term-scalar) reference, focused tests, inventory/evidence,
and relevant [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance). Compare generation with one complete [finite](../../../../glossary.md#term-finite) direct design: typed
heap/segment entries, dense [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and general long-address forms, both mappings, cold
route/carrier/layout selection, canonical represented addition, range [ownership](../../../../glossary.md#term-ownership-meanings), and replacement/
retirement evidence. Select direct Java only if that complete design is genuinely less work to
implement and verify. Otherwise retain generation with no executable change.

Reject exactly the current unsupported forms: other operations or multiple-[node](../../../../glossary.md#node) partitions; wrong
input/output [arity](../../../../glossary.md#term-arity) or non-distinct values; type mismatch, `BOOL`, or integral `FOLD2D`; unresolved,
negative, or malformed layouts; non-injective output; invalid ranks, axis, output size, window
size, step, count, shapes, effective kernel, or column grid; unsupported carrier pattern; and
physical input/output overlap at cold [binding](../../../../glossary.md#term-binding). Do not widen support or invent legacy operations.

`legacy/pre-rewrite` is read-only evidence for selected capability, observable behavior, algorithm,
or test ideas only. Do not copy source, package structure, dependencies, [runtime](../../../../glossary.md#term-runtime) coupling, or
shortcuts.

## Out of [scope](../../../../glossary.md#term-scope)

`UNFOLD_AXIS`, `UNFOLD2D`, [pooling](../../../../glossary.md#term-pooling), scatter, [gather](../../../../glossary.md#gather)/indexing, aggregates, scans, ordering,
reductions, [normalization](../../../../glossary.md#term-normalization), [loss](../../../../glossary.md#loss), public [Model](../../../../glossary.md#term-model) semantics, architecture or [module](../../../../glossary.md#term-module) changes, a
permanent generated/direct duplicate route, and a benchmark or performance-ratio gate. Keep 0009E
and each of its children summary-only; create no 0009E child specification.

## Architecture references and constraints

[`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) is authoritative. Read it with the
[current architecture index](../../../../architecture/current-architecture-plan.md),
[Planning Guide](../../../planning-guide.md), [CPU master plan](../master-plan.md), parent
[CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), parent
[CPU 0009D](0009d-aggregate-scan-ordering-fold-structural-oracle-closure.md), and completed
[D1](0009d1-aggregate-structural-oracle-closure.md),
[D1A](0009d1a-aggregate-segment-layout-prologue-hygiene.md),
[D2](0009d2-scan-route-decision-and-possible-direct-java-migration.md), and
[D3](0009d3-ordering-index-scratch-direct-java-migration.md).

[CPU](../../../../glossary.md#term-cpu) [Prepare](../../../../glossary.md#prepare), not [Planning](../../../../glossary.md#term-planning) or [Runtime](../../../../glossary.md#term-runtime), cold-selects the concrete route and [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout) form.
[Lowering](../../../../glossary.md#lowering)/prepare/finalization retain [immutable](../../../../glossary.md#term-immutable) geometry, range, and [binding](../../../../glossary.md#term-binding) facts; Runtime invokes
prepared work only. A direct route preserves cold overlap checking and immutable recipes. Typed hot
loops must match optimal clean Java's semantic algorithm and avoidable-overhead [shape](../../../../glossary.md#shape): no
per-[element](../../../../glossary.md#term-element) allocation, boxing, reflection, string/map [dispatch](../../../../glossary.md#term-dispatch), [synchronization](../../../../glossary.md#term-synchronization), or avoidable
virtual/interface dispatch. No [`Operation`](../../../../glossary.md#operation) or `CompiledNode` reaches Runtime hot loops. The [scalar](../../../../glossary.md#term-scalar)
reference remains a [conformance](../../../../glossary.md#term-conformance)/[fail-closed](../../../../glossary.md#term-fail-closed) [oracle](../../../../glossary.md#term-test-oracle), not a Runtime interpreter.

## Package impact

Decision-only changes [planning](../../../../glossary.md#term-planning) records only. A direct outcome remains in
`io.github.pho001.synaptik.backend.cpu.internal`: [lowering](../../../../glossary.md#lowering)/[IR](../../../../glossary.md#term-intermediate-representation-ir) retain cold structural facts,
[prepare](../../../../glossary.md#prepare)/finalization select and bind the route, executable owns invocation, and a [CPU](../../../../glossary.md#term-cpu)-private
direct [owner](../../../../glossary.md#term-owner) contains typed fold loops. No public package, [Model](../../../../glossary.md#term-model) contract, shared Prepare/[Runtime](../../../../glossary.md#term-runtime)
contract, [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), or architecture boundary changes.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

Decision-only scope is exactly this task, parent 0009D, parent 0009, [CPU](../../../../glossary.md#term-cpu) master plan, and roadmap.
A direct outcome may change only CPU-private fold [lowering](../../../../glossary.md#lowering)/[IR](../../../../glossary.md#term-intermediate-representation-ir) route plumbing, preparation,
finalization/executable invocation, direct typed implementation, generated fold-route [owners](../../../../glossary.md#term-owner), and
focused CPU tests/resources, plus those five [planning](../../../../glossary.md#term-planning) records; at most 20 CPU
production/test/resource paths. Retire `CpuFoldEmitter` selection and fold-only inventory/evidence/
disposition rows only after full replacement semantics, invocation, [conformance](../../../../glossary.md#term-conformance), and no-selected-
reference checks pass. Do not remove shared owners still selected elsewhere. Retain generation or
stop and replan rather than leave a partial [topology](../../../../glossary.md#term-topology) or cross the bound.

## Decision and acceptance criteria

- The gate retains generation for the entire 32-row fold family. A complete direct implementation
  would duplicate five typed axis and three typed 2D bodies across four carrier combinations,
  dense/general forms, cold selection/binding, output ranges, exact mapping/rejections,
  BFLOAT16 per-addition rounding, integral wraparound, and replacement/invocation/retirement
  evidence. That is more implementation and verification work than retaining the coherent route.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [carrier glossary](../../../../glossary.md#term-storage-carrier); [binding glossary](../../../../glossary.md#term-binding); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [rounding glossary](../../../../glossary.md#term-rounding); [wraparound glossary](../../../../glossary.md#term-wraparound))</span>
- The existing route continues to provide positive-zero initialization, canonical input-row-major
  addition, BFLOAT16 per-addition rounding, integral wraparound, padding/ceil/dilation exclusion,
  ranges, injectivity, cold overlap rejection, and exact unsupported-form rejection.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Generated selection, inventory, ledger/disposition, and structural evidence remain intact; no
  direct route, duplicate topology, benchmark gate, or executable change is introduced.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
- A clean documentation-focused context finalizes planning, Javadoc/docs/glossary review, and
  reasoned no-change conclusions. Parent 0009D completes; CPU 0009 remains Ready, and 0009E and
  its children remain summary-only.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc); [CPU glossary](../../../../glossary.md#term-cpu); [Ready glossary](../../../../glossary.md#term-ready))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Before deciding, [run](../../../../glossary.md#term-run) or reuse focused fold [lowering](../../../../glossary.md#lowering)/[IR](../../../../glossary.md#term-intermediate-representation-ir), generated semantic/inventory evidence,
preparer/finalizer, prepared invocation/range/overlap, and relevant [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) evidence.

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuFoldGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuOrdinaryMovementFoldSemanticClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageCheckpointTest
```

A direct outcome adds focused route-selection, all-type/both-family semantics, [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout),
range/overlap/rejection, hot-loop hygiene, retirement, and no-selected-reference checks, then the
[CPU](../../../../glossary.md#term-cpu) [module](../../../../glossary.md#term-module) suite once after executable edits. Run relevant backend conformance when its covered
fold behavior changes. The documentation pass runs `./gradlew :backends:cpu:javadoc` after final
[Javadoc](../../../../glossary.md#term-javadoc) edits, validates glossary impact and local links/anchors/fences/terminology, and reuses
Java evidence unless executable behavior changes afterward. A retained result runs no benchmark;
validate status/order, exact [scope](../../../../glossary.md#term-scope), and `git diff --check`.

## Dependencies and follow-up tasks

D1, D1A, D2, and D3 are Complete. This task completes the all-or-nothing retained-generation
decision before summary-only 0009E. Do not create an 0009E child.

## Architecture impact

Expected impact: None. Stop and report any required public, [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module)-boundary, or
architecture-contract change.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009D4. Read AGENTS.md, ARCHITECTURE.md,
current architecture plan, Planning Guide, documentation rules and General/Planning profiles, CPU
master plan, CPU 0009/D/D1/D1A/D2/D3, this task, and complete current fold
lowering/IR/emitter/generator/preparer/finalizer/executable/reference/test/inventory/evidence and
relevant conformance owners. Perform the bounded complete-family route-decision gate first.
Migrate FOLD_AXIS and FOLD2D together to finite direct typed Java only if it genuinely reduces
total implementation and verification work; otherwise retain generation explicitly. Preserve the
exact matrix, cold selection, canonical addition, range and overlap semantics. Retire generated
fold routing/inventory/evidence only after replacement semantics, invocation, conformance, and
no-selected-reference checks pass. Do not create 0009E children, add a benchmark gate, change
architecture, commit, or push. Use legacy/pre-rewrite only as read-only behavioral/test evidence.
Hand executable changes and exact test evidence to a separate clean documentation-focused context.
```

## Local decisions

“Fold/window family” means only `CpuFoldIr`'s `FOLD_AXIS` and `FOLD2D` overlap-add mappings.
Unfold and [pooling](../../../../glossary.md#term-pooling) windows are not a cohesive subfamily. The family is all-or-nothing because its
current emitter, cold geometry, typed carriers, output-range [ownership](../../../../glossary.md#term-ownership-meanings), and [canonical](../../../../glossary.md#term-canonical-form) represented
addition share proof obligations. The 32 rows and wider 160-[owner](../../../../glossary.md#term-owner) SHA-bound projection are
evidence to inspect, not a mandate for a universal structural [oracle](../../../../glossary.md#term-test-oracle) or benchmark.

## Known limitations

Existing performance facts remain historical. The retained route makes no new performance claim;
it is selected solely because a direct replacement would cost more to implement and verify.

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context ran the specified eight-class command on 2026-09-09: 131 tests,
zero failures, zero errors, and zero skips (3 [lowering](../../../../glossary.md#lowering), 2 [IR](../../../../glossary.md#term-intermediate-representation-ir), 12 generated-fold, 2 semantic
closure, 29 preparer, 16 finalizer, 58 executable, and 9 [checkpoint](../../../../glossary.md#term-checkpoint)). It established the exact
five-type [axis](../../../../glossary.md#term-axis)/three-type [NCHW](../../../../glossary.md#term-nchw) [matrix](../../../../glossary.md#term-matrix-meanings), four [carrier](../../../../glossary.md#term-storage-carrier) combinations, output-owned range semantics,
zero workspace, positive-zero/[canonical](../../../../glossary.md#term-canonical-form)-addition policy, and checked 20+12 inventory projection.

Documentation-focused context `/root/cpu_0009d4_docs` independently inspected the complete fold
lowering/IR/emitter/generator/preparer/finalizer/prepared-executable/[scalar](../../../../glossary.md#term-scalar)-reference/test and
inventory/evidence [ownership](../../../../glossary.md#term-ownership-meanings). It reused that stabilized executable evidence because no Java,
test, resource, or behavioral change followed it. It validated the final five-record [scope](../../../../glossary.md#term-scope), local
links, heading anchors, fences, terminology, status and [dependency](../../../../glossary.md#term-dependency) order, and `git diff --check`.
No [Javadoc](../../../../glossary.md#term-javadoc) generation or [conformance](../../../../glossary.md#term-conformance)/integration command ran: neither Java APIs nor covered
behavior changed.

## Implementation notes

Retain the generated route for the complete `FOLD_AXIS`/`FOLD2D` family. `CpuFoldLowering`,
`CpuFoldIr`, `CpuFoldEmitter`, `CpuClassFileKernelGenerator`, preparation/finalization, [prepared
execution](../../../../glossary.md#prepared-execution--preparedexecution), [scalar](../../../../glossary.md#term-scalar) reference, focused tests, and the 32-row inventory remain selected and intact.
No executable Java, test, resource, architecture, public [API](../../../../glossary.md#term-application-programming-interface-api), [Javadoc](../../../../glossary.md#term-javadoc), glossary, [Gradle](../../../../glossary.md#term-gradle),
[conformance](../../../../glossary.md#term-conformance)/integration, or other-[module](../../../../glossary.md#term-module) change is needed.

## Completion summary

- Completed changes: Recorded the complete retained-generated-route decision for all 32 fold
  rows and synchronized the five CPU planning records.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: This task, parent 0009D, parent 0009, [CPU](../../../../glossary.md#term-cpu) master plan, and roadmap.
- Tests and validation: Reused the implementation context's specified eight-class CPU command:
  131 tests, zero failures, errors, or skips. This documentation pass checked links, anchors,
  fences, terminology, status/dependency order, exact changed-path scope, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Documentation-agent review: Clean context `/root/cpu_0009d4_docs` applied the General and
  Planning documentation profiles and independently reviewed route ownership and evidence.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Documentation impact: Planning records only. Architecture/ADR, Gradle, backend conformance,
  integration, and other modules remain unchanged because behavior and ownership do not change.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: No change; Java behavior and contracts are unchanged, so Javadoc was not [run](../../../../glossary.md#term-run).
- Glossary impact: No change; fold, carrier, range ownership, and canonical addition retain their
  established meanings.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Unresolved issues: None.
- Follow-up required: CPU 0009E remains the summary-only next frontier; do not create child specs
  until it becomes actionable.

Status: Complete
