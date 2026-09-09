# Task 0009D4: Fold and Window Route Decision and Possible Direct-Java Migration

## Status

Ready

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

## Acceptance criteria

- The gate selects direct Java or retained generation for the whole 32-row fold family, recording
  a concrete comparison of total implementation and evidence work.
- A selected direct route covers every supported form and exact rejection: positive-zero
  initialization, canonical input-row-major addition, BFLOAT16 per-addition rounding, integral
  wraparound, padding/ceil/dilation exclusion, ranges, injectivity, and cold overlap rejection.
- Carrier/layout/range selection is cold; selected entries are typed and hot-loop dispatch-free,
  preserve dense rank-one/general-address behavior, and introduce no workspace.
- Retire generated selection, inventory, ledger/disposition, and structural evidence only after
  focused replacement semantics, prepared invocation, relevant conformance, no-selected-reference,
  and retirement checks. A retained outcome leaves them intact and records why.
- The implementation reviews affected Javadoc/docs; a clean documentation-focused context
  finalizes Javadoc/docs/glossary impact and reasoned no-change conclusions. Add no benchmark gate.
- Parent 0009D stays Ready until D4 execution completes. 0009E and its children stay summary-only.

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

D1, D1A, D2, and D3 are Complete. This is the sole detailed Ready frontier and must complete its
cohesive all-or-nothing route decision before summary-only 0009E. Do not create an 0009E child.

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

No direct route is presumed selected. Existing performance facts remain historical; neither
outcome makes a performance claim.

## Validation evidence

Planning creation evidence on 2026-09-09: source/test inspection established the exact five-type
axis/three-type NCHW matrix, four carrier combinations, output-owned range semantics, zero
workspace, positive-zero/canonical-addition policy, and checked 20+12 inventory projection. This
planning-only context runs no Java or conformance command; the implementation context must record
the focused command and any replacement evidence. This documentation-focused planning context
validates final five-record scope, local links/anchors/fences, terminology, status/order, and
`git diff --check` after synchronization.

## Implementation notes

None. This Ready specification changes no executable Java, tests, architecture, Javadoc, glossary,
or generated inventory.

## Completion summary

Pending execution. Status remains Ready until route decision, any authorized implementation,
validation, and the separate documentation-focused pass are recorded.

Status: Ready
