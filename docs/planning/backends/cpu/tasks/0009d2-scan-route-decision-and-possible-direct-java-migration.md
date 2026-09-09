# Task 0009D2: Scan Route Decision and Possible Direct-Java Migration

## Status

Complete

## Goal

Decide the least costly maintainable implementation route for the ordinary cumulative-scan family.
Use finite direct typed Java only if focused inspection establishes that it is genuinely simpler to
implement and verify than retaining the current generated route; otherwise retain generation with
an explicit recorded decision. No performance benefit or new benchmark gate is required.

## Scope

Inspect the current scan lowering, preparation/invocation boundary, `CpuScanEmitter`, semantic and
inventory evidence, and the clean/reference implementation used by current tests. The
`legacy/pre-rewrite` branch is read-only behavioral/test evidence only; do not copy its source,
packages, dependencies, runtime coupling, or shortcuts. Cover current
CUM_SUM and CUM_PROD forms across admitted types, axes, inclusive/exclusive and forward/reverse
modes, carriers, layouts, and caller-owned complete-slice ranges.

The first implementation step is a decision gate. It must compare the full supported matrix,
typed carrier/address routes, BFLOAT16 conversion-after-each-value semantics, integral wrap
semantics, range ownership, and existing generated inventory/evidence. Direct migration is
authorized only when that inspection proves a cohesive finite direct design can replace the full
selected generated route with less implementation and verification work. Otherwise record
retention, preserve the generated route and all evidence, and complete the task without code
changes.

## Out of scope

Aggregate work and D1 evidence; ordering, fold, general reductions, normalization, loss, pooling,
attention, batch normalization, MATMUL, convolution, a universal structural verifier, a required
performance study or benchmark gate, architecture/module-boundary changes, and a permanent
direct/generated duplicate route.

## Architecture references and constraints

[`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) is authoritative. Read it with the [current
architecture index](../../../../architecture/current-architecture-plan.md), [Planning
Guide](../../../planning-guide.md), [CPU master plan](../master-plan.md), parent
[CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), and parent
[CPU 0009D](0009d-aggregate-scan-ordering-fold-structural-oracle-closure.md). Prepare selects a
concrete typed implementation; Runtime invokes prepared work; ranges own complete scan slices;
dispatch, carrier selection, geometry validation, and partitioning stay cold. Element loops must
not allocate, box, reflect, synchronize, use string/map dispatch, or make avoidable
virtual/interface calls. A retained generated route remains supported only with its current
proportionate hygiene, semantic, and inventory evidence.

## Package impact

The decision-only outcome changes planning records only. If the gate authorizes direct migration,
the task must first name the exact existing CPU-private lowering, prepare, executable, generator,
and test packages it changes; no public package, dependency, or architecture change is allowed.

## Affected files and maximum scope

Decision-only outcome: this task, parent 0009D, parent 0009, CPU master plan, and roadmap. A
direct-migration outcome may change only the CPU-private owners and focused tests/resources proved
necessary by the decision, plus those planning records; stop and replan if it crosses a public or
shared boundary or needs more than 18 CPU production/test/resource paths.

## Acceptance criteria

- The decision gate documents the inspected scan matrix and concludes either a full direct route
  with less implementation/verification work or an explicit retained generated route.
- A direct route, if authorized, preserves selected typed carrier/address entries, complete-slice
  range ownership, all current scan semantics and exact rejections, and has no selected generated
  scan route after replacement validation.
- A retained route preserves existing scan semantics, inventory, and hygiene evidence without
  unsupported or performance claims.
- No partial direct/generated split is retained unless the task records a concrete lower-complexity
  reason and exact route/evidence ownership.
- D3 remains summary-only until D2 is complete.

## Tests / validation

Before deciding, run or reuse current focused scan lowering, preparation/invocation, semantic, and
generated-route hygiene evidence, recording exact commands and results. A direct-migration outcome
adds focused route-selection, semantic/canary, invocation/range, and retirement checks, then runs
the CPU module suite once after final executable edits. A retained outcome runs no new benchmark;
it validates the decision record, selected-route references, Markdown links/anchors/fences, status
and order, exact changed-path scope, and `git diff --check`.

## Dependencies and follow-up tasks

CPU 0009D1 is Complete as an aggregate retained-route decision. D2 is complete; D3 ordering is
now the next summary-only frontier. D4 fold follows D3, and CPU 0009E follows D4.

## Architecture impact

Expected impact: None. Stop and report any required public, dependency, module-boundary, or
architecture-contract change.

## Implementation prompt

```text
Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, Planning Guide, documentation
rules with General and Planning profiles, CPU master plan, CPU 0009/D/D1, and this task. Implement
CPU 0009D2 exactly. First perform and record the scan route decision gate. Migrate to finite direct
typed Java only if the full current scan matrix is genuinely cheaper to implement and verify than
retained generation; otherwise retain the generated route explicitly. Preserve cold selection,
complete-slice range ownership, semantics, and existing evidence. Do not create D3 details, add a
benchmark gate, change architecture, commit, or push. Hand any executable diff and evidence to a
separate clean documentation-focused context.
```

## Local decisions

Retain the generated scan route. The gate inspected the complete admitted matrix: two kinds
(`CUM_SUM` and `CUM_PROD`), five represented numeric types (FLOAT64, FLOAT32, BFLOAT16, INT32,
and INT64), inclusive/exclusive and forward/reverse modes, every normalized axis of a non-scalar
static resolved layout, and the selected heap, segment, mixed-carrier, dense, and general-layout
forms. BOOL and scalar inputs remain exact rejections. The checked ordinary matrix makes the
finite tested projection concrete: 40 semantic bases (2 x 5 x 2 x 2) times four carrier/layout
requests equals 160 generated scan entries.

Direct Java is not lower-complexity. It would need a new finite dispatch-free implementation
surface for all typed array/`MemorySegment` ordered input/output combinations, the dense rank-one
array form, the general coordinate/address form, and the existing guarded INT64 segment cursor
form; it would also need replacement invocation, range, inventory, hygiene, and retirement
evidence. The existing generated route already selects its typed body during preparation, accepts
only complete caller-owned slice ranges, retains sequential per-slice accumulation, rounds
BFLOAT16 after every value, preserves INT32/INT64 Java-width wraparound, and keeps layout/carrier
facts cold. Replacing it would duplicate those tested specializations and add a retired-generated
proof without reducing verification work. This is a retention decision, not a performance claim
or a benchmark result.

## Known limitations

No new performance claim follows from either route decision. Existing performance facts remain
historical evidence.

## Validation evidence

The focused route evidence passed on 2026-09-09:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuScanGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateScanSemanticClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageCheckpointTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest
```

The command passed 65 tests with zero failures, errors, or skips. The scan semantic closure
defined and invoked all 160 scan artifacts in its 460 aggregate-and-scan owner set; the
checkpoint sealed the exact generated inventory projection. `CpuScanGeneratedKernelTest` also
exercises all kinds/modes/types against the independent scalar reference, BFLOAT16
round-to-nearest-ties-to-even after every operation, integral wraparound, and a caller-owned
subrange for the guarded reverse INT64 `MemorySegment` product route. Lowering and IR tests cover
the five-type admission matrix, exact BOOL/scalar rejection, normalized-axis identity, and
injective general layouts. The preparer/finalizer tests cover the existing cold route-to-executable
handoff. Read-only `legacy/pre-rewrite` inspection found no corresponding current scan emitter;
no legacy source was copied or used as a design dependency.

Documentation-focused context `/root/cpu_0009d2_docs` independently inspected the final planning
diff, current CPU scan route, and cited tests. It confirmed the retained-route decision and the
no-change conclusions below; its planning validation is recorded in this task's completion
summary.

## Implementation notes

No executable Java changed. `CpuScanLowering` continues to produce a two-boundary,
zero-workspace `CpuScanIr`; `CpuPartitionPreparer` and `CpuPartitionFinalizer` continue to retain
the scan geometry through preparation; `CpuClassFileKernelGenerator` continues to select
`CpuScanEmitter`; and the prepared executable invokes the resulting typed entry. Existing
inventory SHA-256 remains
`527f36de41b64c228dd215c6f3182138b4c70db24c915117c4d7f82743ac41cc`.

## Completion summary

- Completed changes: Recorded the full scan route-decision gate and retained the existing
  generated route; no production, test, generated artifact, inventory, or Javadoc path changed.
- Files changed or created: This task, parent 0009D, parent 0009, CPU master plan, and roadmap.
- Tests and validation: The seven-class focused CPU command passed 65 tests with zero failures,
  errors, or skips. Documentation-focused context `/root/cpu_0009d2_docs` independently inspected
  the listed scan source/tests and final diff; it checked local Markdown links, headings used as
  anchors, fences, terminology, task/order/status synchronization, the exact five-path scope, and
  `git diff --check`. It did not rerun Java tests because no executable Java changed.
- Documentation-agent review: Clean documentation-focused context `/root/cpu_0009d2_docs`
  finalized these planning records using the General and Planning documentation profiles.
- Documentation impact: Planning records are substantively revised; public/API Javadoc, guides,
  glossary, architecture/ADR, Gradle, conformance/integration, and other modules have no current
  change because executable behavior, public API, architecture, and reusable terminology are
  unchanged.
- Javadoc review: No change; no Java contract or implementation changed.
- Glossary impact: No change; the retained route uses existing terms without changing their
  meanings or boundaries.
- Unresolved issues: None in implementation. D3 remains summary-only and must not receive a
  detailed specification until it becomes the frontier.

Status: Complete
Follow-up required: D3 ordering is the next summary-only frontier; create no detailed D3
specification until it becomes actionable.
