# Task 0009D4: Fold and Window Route Decision and Possible Direct-Java Migration

## Status

Complete

## Goal

Decide whether finite direct typed Java lowers the total implementation and verification burden
for the complete currently supported CPU overlap-fold family. Migrate only if it does; otherwise
retain the generated route explicitly. This is a route-complexity decision, not a performance
claim or benchmark gate.

## Scope

The cohesive all-or-nothing family is exactly one fully static, resolved-layout,
zero-initialized overlap-add occurrence of `FOLD_AXIS` or `FOLD2D`:

| Dimension | Exact supported forms |
|---|---|
| Operations and types | `FOLD_AXIS`: `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`. `FOLD2D`: `FLOAT64`, `FLOAT32`, `BFLOAT16`. `BOOL` is never admitted. |
| Shapes and mapping | `FOLD_AXIS` removes its final positive window dimension, restores the normalized target axis to `outputSize`, and has exact count/step geometry. `FOLD2D` has rank-three columns input, rank-four canonical NCHW output, exact `Fold2dAttrs.outputShape`, and checked kernel, stride, symmetric padding, dilation, floor/ceil grid mapping. |
| Layout and carriers | Input has non-negative offset/strides and may be dense or general, including zero-stride read axes. Output is distinct, non-overlapping, injective, and resolved. Each boundary is its typed heap array or `MemorySegment`, including all four combinations and cold-bound offsets. |
| Addition and range | Every selected output ordinal in `[start,end)` starts at represented positive zero then receives canonical input-row-major additions. `BFLOAT16` rounds after every addition; `INT32`/`INT64` wrap. Uncovered outputs stay positive zero. Independent output ranges require no workspace. |
| Current generated inventory | 32 checked rows: 20 `FOLD_AXIS` (five types × four carrier combinations) and 12 `FOLD2D` (three types × four carrier combinations), sealed in the wider 160-owner SHA-256-bound ordinary movement/fold projection. |

Inspect the complete fold lowerer, `CpuFoldIr`, emitter, generator selection, preparation,
finalization, prepared executable invocation, scalar reference, focused tests, inventory/evidence,
and relevant backend conformance. Compare generation with one complete finite direct design: typed
heap/segment entries, dense rank-one and general long-address forms, both mappings, cold
route/carrier/layout selection, canonical represented addition, range ownership, and replacement/
retirement evidence. Select direct Java only if that complete design is genuinely less work to
implement and verify. Otherwise retain generation with no executable change.

Reject exactly the current unsupported forms: other operations or multiple-node partitions; wrong
input/output arity or non-distinct values; type mismatch, `BOOL`, or integral `FOLD2D`; unresolved,
negative, or malformed layouts; non-injective output; invalid ranks, axis, output size, window
size, step, count, shapes, effective kernel, or column grid; unsupported carrier pattern; and
physical input/output overlap at cold binding. Do not widen support or invent legacy operations.

`legacy/pre-rewrite` is read-only evidence for selected capability, observable behavior, algorithm,
or test ideas only. Do not copy source, package structure, dependencies, runtime coupling, or
shortcuts.

## Out of scope

`UNFOLD_AXIS`, `UNFOLD2D`, pooling, scatter, gather/indexing, aggregates, scans, ordering,
reductions, normalization, loss, public Model semantics, architecture or module changes, a
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

CPU Prepare, not Planning or Runtime, cold-selects the concrete route and carrier/layout form.
Lowering/prepare/finalization retain immutable geometry, range, and binding facts; Runtime invokes
prepared work only. A direct route preserves cold overlap checking and immutable recipes. Typed hot
loops must match optimal clean Java's semantic algorithm and avoidable-overhead shape: no
per-element allocation, boxing, reflection, string/map dispatch, synchronization, or avoidable
virtual/interface dispatch. No `Operation` or `CompiledNode` reaches Runtime hot loops. The scalar
reference remains a conformance/fail-closed oracle, not a Runtime interpreter.

## Package impact

Decision-only changes planning records only. A direct outcome remains in
`io.github.pho001.synaptik.backend.cpu.internal`: lowering/IR retain cold structural facts,
prepare/finalization select and bind the route, executable owns invocation, and a CPU-private
direct owner contains typed fold loops. No public package, Model contract, shared Prepare/Runtime
contract, module dependency, or architecture boundary changes.

## Affected files and maximum scope

Decision-only scope is exactly this task, parent 0009D, parent 0009, CPU master plan, and roadmap.
A direct outcome may change only CPU-private fold lowering/IR route plumbing, preparation,
finalization/executable invocation, direct typed implementation, generated fold-route owners, and
focused CPU tests/resources, plus those five planning records; at most 20 CPU
production/test/resource paths. Retire `CpuFoldEmitter` selection and fold-only inventory/evidence/
disposition rows only after full replacement semantics, invocation, conformance, and no-selected-
reference checks pass. Do not remove shared owners still selected elsewhere. Retain generation or
stop and replan rather than leave a partial topology or cross the bound.

## Decision and acceptance criteria

- The gate retains generation for the entire 32-row fold family. A complete direct implementation
  would duplicate five typed axis and three typed 2D bodies across four carrier combinations,
  dense/general forms, cold selection/binding, output ranges, exact mapping/rejections,
  BFLOAT16 per-addition rounding, integral wraparound, and replacement/invocation/retirement
  evidence. That is more implementation and verification work than retaining the coherent route.
- The existing route continues to provide positive-zero initialization, canonical input-row-major
  addition, BFLOAT16 per-addition rounding, integral wraparound, padding/ceil/dilation exclusion,
  ranges, injectivity, cold overlap rejection, and exact unsupported-form rejection.
- Generated selection, inventory, ledger/disposition, and structural evidence remain intact; no
  direct route, duplicate topology, benchmark gate, or executable change is introduced.
- A clean documentation-focused context finalizes planning, Javadoc/docs/glossary review, and
  reasoned no-change conclusions. Parent 0009D completes; CPU 0009 remains Ready, and 0009E and
  its children remain summary-only.

## Tests / validation

Before deciding, run or reuse focused fold lowering/IR, generated semantic/inventory evidence,
preparer/finalizer, prepared invocation/range/overlap, and relevant backend-conformance evidence.

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

A direct outcome adds focused route-selection, all-type/both-family semantics, carrier/layout,
range/overlap/rejection, hot-loop hygiene, retirement, and no-selected-reference checks, then the
CPU module suite once after executable edits. Run relevant backend conformance when its covered
fold behavior changes. The documentation pass runs `./gradlew :backends:cpu:javadoc` after final
Javadoc edits, validates glossary impact and local links/anchors/fences/terminology, and reuses
Java evidence unless executable behavior changes afterward. A retained result runs no benchmark;
validate status/order, exact scope, and `git diff --check`.

## Dependencies and follow-up tasks

D1, D1A, D2, and D3 are Complete. This task completes the all-or-nothing retained-generation
decision before summary-only 0009E. Do not create an 0009E child.

## Architecture impact

Expected impact: None. Stop and report any required public, dependency, module-boundary, or
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
Unfold and pooling windows are not a cohesive subfamily. The family is all-or-nothing because its
current emitter, cold geometry, typed carriers, output-range ownership, and canonical represented
addition share proof obligations. The 32 rows and wider 160-owner SHA-bound projection are
evidence to inspect, not a mandate for a universal structural oracle or benchmark.

## Known limitations

Existing performance facts remain historical. The retained route makes no new performance claim;
it is selected solely because a direct replacement would cost more to implement and verify.

## Validation evidence

The implementation context ran the specified eight-class command on 2026-09-09: 131 tests,
zero failures, zero errors, and zero skips (3 lowering, 2 IR, 12 generated-fold, 2 semantic
closure, 29 preparer, 16 finalizer, 58 executable, and 9 checkpoint). It established the exact
five-type axis/three-type NCHW matrix, four carrier combinations, output-owned range semantics,
zero workspace, positive-zero/canonical-addition policy, and checked 20+12 inventory projection.

Documentation-focused context `/root/cpu_0009d4_docs` independently inspected the complete fold
lowering/IR/emitter/generator/preparer/finalizer/prepared-executable/scalar-reference/test and
inventory/evidence ownership. It reused that stabilized executable evidence because no Java,
test, resource, or behavioral change followed it. It validated the final five-record scope, local
links, heading anchors, fences, terminology, status and dependency order, and `git diff --check`.
No Javadoc generation or conformance/integration command ran: neither Java APIs nor covered
behavior changed.

## Implementation notes

Retain the generated route for the complete `FOLD_AXIS`/`FOLD2D` family. `CpuFoldLowering`,
`CpuFoldIr`, `CpuFoldEmitter`, `CpuClassFileKernelGenerator`, preparation/finalization, prepared
execution, scalar reference, focused tests, and the 32-row inventory remain selected and intact.
No executable Java, test, resource, architecture, public API, Javadoc, glossary, Gradle,
conformance/integration, or other-module change is needed.

## Completion summary

- Completed changes: Recorded the complete retained-generated-route decision for all 32 fold
  rows and synchronized the five CPU planning records.
- Files changed or created: This task, parent 0009D, parent 0009, CPU master plan, and roadmap.
- Tests and validation: Reused the implementation context's specified eight-class CPU command:
  131 tests, zero failures, errors, or skips. This documentation pass checked links, anchors,
  fences, terminology, status/dependency order, exact changed-path scope, and `git diff --check`.
- Documentation-agent review: Clean context `/root/cpu_0009d4_docs` applied the General and
  Planning documentation profiles and independently reviewed route ownership and evidence.
- Documentation impact: Planning records only. Architecture/ADR, Gradle, backend conformance,
  integration, and other modules remain unchanged because behavior and ownership do not change.
- Javadoc review: No change; Java behavior and contracts are unchanged, so Javadoc was not run.
- Glossary impact: No change; fold, carrier, range ownership, and canonical addition retain their
  established meanings.
- Unresolved issues: None.
- Follow-up required: CPU 0009E remains the summary-only next frontier; do not create child specs
  until it becomes actionable.

Status: Complete
