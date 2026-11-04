# Task 0009D2: [Scan](../../../../glossary.md#term-cumulative-scan) Route Decision and Possible Direct-Java Migration

## Reader context

This task records retaining generated CUM_SUM and CUM_PROD across admitted numeric types, scan
modes, and carrier/layout forms. `CpuScanLowering` produces `CpuScanIr` with two boundaries and zero
workspace; `CpuPartitionPreparer` and `CpuPartitionFinalizer` preserve scan geometry, and
`CpuClassFileKernelGenerator` selects `CpuScanEmitter`. A direct route would have to replace that
complete typed surface and its evidence, not merely supply an arithmetic loop.

Here, the CPU executable assigns workers complete scan slices, keeping sequential accumulation
within each slice. Preparation resolves carrier and layout choices before execution. Semantic tests
compare results with an independent scalar reference, including per-value `BFLOAT16` rounding and
integral wraparound; the inventory checkpoint identifies generated artifacts. `BOOL` and scalar
inputs remain recorded rejections, and retention supplies no new performance claim.

Current-topic navigation (not a replacement for the historical record):

- [current cumulative scan modes](../../../../reference/tensor/fixed-scans.md#cumulative-scan-expressions) — Explains axis slices, inclusion, direction, identities, and the metadata-versus-evaluation boundary.
- [backend ownership versus implementation route](../../../../design/notes/cpu-kernel-strategy.md#strategy) — Separates Planning ownership, CPU route choice, finalization, and Runtime invocation.

## Status

Complete

## Goal

Decide the least costly maintainable implementation route for the ordinary cumulative-[scan](../../../../glossary.md#term-cumulative-scan) family.
Use [finite](../../../../glossary.md#term-finite) direct typed Java only if focused inspection establishes that it is genuinely simpler to
implement and verify than retaining the current generated route; otherwise retain generation with
an explicit recorded decision. No performance benefit or new benchmark gate is required.

## [Scope](../../../../glossary.md#term-scope)

Inspect the current [scan](../../../../glossary.md#term-cumulative-scan) [lowering](../../../../glossary.md#lowering), preparation/invocation boundary, `CpuScanEmitter`, semantic and
inventory evidence, and the clean/reference implementation used by current tests. The
`legacy/pre-rewrite` branch is read-only behavioral/test evidence only; do not copy its source,
packages, dependencies, [runtime](../../../../glossary.md#term-runtime) coupling, or shortcuts. Cover current
CUM_SUM and CUM_PROD forms across admitted types, [axes](../../../../glossary.md#term-axis), inclusive/exclusive and forward/reverse
modes, carriers, [layouts](../../../../glossary.md#layout), and caller-owned complete-[slice](../../../../glossary.md#slice) ranges.

The first implementation step is a decision gate. It must compare the full supported [matrix](../../../../glossary.md#term-matrix-meanings),
typed [carrier](../../../../glossary.md#term-storage-carrier)/address routes, [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion-after-each-value semantics, integral wrap
semantics, range [ownership](../../../../glossary.md#term-ownership-meanings), and existing generated inventory/evidence. Direct migration is
authorized only when that inspection proves a cohesive [finite](../../../../glossary.md#term-finite) direct design can replace the full
selected generated route with less implementation and verification work. Otherwise record
retention, preserve the generated route and all evidence, and complete the task without code
changes.

## Out of [scope](../../../../glossary.md#term-scope)

Aggregate work and D1 evidence; ordering, fold, general reductions, [normalization](../../../../glossary.md#term-normalization), [loss](../../../../glossary.md#loss), [pooling](../../../../glossary.md#term-pooling),
attention, [batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training), [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), [convolution](../../../../glossary.md#term-channels-first-convolution-layers), a universal structural verifier, a required
performance study or benchmark gate, architecture/[module](../../../../glossary.md#term-module)-boundary changes, and a permanent
direct/generated duplicate route.

## Architecture references and constraints

[`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) is authoritative. Read it with the [current
architecture index](../../../../architecture/current-architecture-plan.md), [Planning
Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>, parent
[CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), and parent
[CPU 0009D](0009d-aggregate-scan-ordering-fold-structural-oracle-closure.md). [Prepare](../../../../glossary.md#prepare) selects a
concrete typed implementation; [Runtime](../../../../glossary.md#term-runtime) invokes prepared work; ranges own complete [scan](../../../../glossary.md#term-cumulative-scan) slices;
[dispatch](../../../../glossary.md#term-dispatch), [carrier](../../../../glossary.md#term-storage-carrier) selection, geometry [validation](../../../../glossary.md#term-validation), and partitioning stay cold. [Element](../../../../glossary.md#term-element) loops must
not allocate, box, reflect, synchronize, use string/map dispatch, or make avoidable
virtual/interface calls. A retained generated route remains supported only with its current
proportionate hygiene, semantic, and inventory evidence.

## Package impact

The decision-only outcome changes [planning](../../../../glossary.md#term-planning) records only. If the gate authorizes direct migration,
the task must first name the exact existing [CPU](../../../../glossary.md#term-cpu)-private [lowering](../../../../glossary.md#lowering), [prepare](../../../../glossary.md#prepare), executable, generator,
and test packages it changes; no public package, [dependency](../../../../glossary.md#term-dependency), or architecture change is allowed.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

Decision-only outcome: this task, parent 0009D, parent 0009, [CPU](../../../../glossary.md#term-cpu) master plan, and roadmap. A
direct-migration outcome may change only the CPU-private [owners](../../../../glossary.md#term-owner) and focused tests/resources proved
necessary by the decision, plus those [planning](../../../../glossary.md#term-planning) records; stop and replan if it crosses a public or
shared boundary or needs more than 18 CPU production/test/resource paths.

## Acceptance criteria

- The decision gate documents the inspected scan matrix and concludes either a full direct route
  with less implementation/verification work or an explicit retained generated route.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- A direct route, if authorized, preserves selected typed carrier/address entries, complete-slice
  range ownership, all current scan semantics and exact rejections, and has no selected generated
  scan route after replacement validation.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [slice glossary](../../../../glossary.md#slice); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- A retained route preserves existing scan semantics, inventory, and hygiene evidence without
  unsupported or performance claims.
- No partial direct/generated split is retained unless the task records a concrete lower-complexity
  reason and exact route/evidence ownership.
- D3 remains summary-only until D2 is complete.

## Tests / [validation](../../../../glossary.md#term-validation)

Before deciding, [run](../../../../glossary.md#term-run) or reuse current focused [scan](../../../../glossary.md#term-cumulative-scan) [lowering](../../../../glossary.md#lowering), preparation/invocation, semantic, and
generated-route hygiene evidence, recording exact commands and results. A direct-migration outcome
adds focused route-selection, semantic/[canary](../../../../glossary.md#term-canary), invocation/range, and retirement checks, then runs
the [CPU](../../../../glossary.md#term-cpu) [module](../../../../glossary.md#term-module) suite once after final executable edits. A retained outcome runs no new benchmark;
it validates the decision record, selected-route references, Markdown links/anchors/fences, status
and order, exact changed-path [scope](../../../../glossary.md#term-scope), and `git diff --check`.

## Dependencies and follow-up tasks

[CPU](../../../../glossary.md#term-cpu) 0009D1 is Complete as an aggregate retained-route decision. D2 is complete; D3 ordering is
now the next summary-only frontier. D4 fold follows D3, and CPU 0009E follows D4.

## Architecture impact

Expected impact: None. Stop and report any required public, [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module)-boundary, or
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

Retain the generated [scan](../../../../glossary.md#term-cumulative-scan) route. The gate inspected the complete admitted [matrix](../../../../glossary.md#term-matrix-meanings): two kinds
(`CUM_SUM` and `CUM_PROD`), five represented numeric types ([FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype),
and [INT64](../../../../glossary.md#data-type--datatype)), inclusive/exclusive and forward/reverse modes, every [normalized axis](../../../../glossary.md#normalized-axis) of a non-[scalar](../../../../glossary.md#term-scalar)
static resolved [layout](../../../../glossary.md#layout), and the selected [heap](../../../../glossary.md#term-heap), segment, mixed-[carrier](../../../../glossary.md#term-storage-carrier), dense, and general-layout
forms. [BOOL](../../../../glossary.md#data-type--datatype) and scalar inputs remain exact rejections. The checked ordinary matrix makes the
[finite](../../../../glossary.md#term-finite) tested projection concrete: 40 semantic bases (2 x 5 x 2 x 2) times four carrier/layout
requests equals 160 generated scan entries.

Direct Java is not lower-complexity. It would need a new finite [dispatch](../../../../glossary.md#term-dispatch)-free implementation
surface for all typed array/[`MemorySegment`](../../../../glossary.md#term-memorysegment) ordered input/output combinations, the dense [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
array form, the general [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/address form, and the existing guarded INT64 segment cursor
form; it would also need replacement invocation, range, inventory, hygiene, and retirement
evidence. The existing generated route already selects its typed body during preparation, accepts
only complete caller-owned [slice](../../../../glossary.md#slice) ranges, retains sequential per-slice accumulation, rounds
BFLOAT16 after every value, preserves INT32/INT64 Java-width [wraparound](../../../../glossary.md#term-wraparound), and keeps layout/carrier
facts cold. Replacing it would duplicate those tested specializations and add a retired-generated
proof without reducing verification work. This is a retention decision, not a performance claim
or a benchmark result.

## Known limitations

No new performance claim follows from either route decision. Existing performance facts remain
historical evidence.

## [Validation](../../../../glossary.md#term-validation) evidence

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

The command passed 65 tests with zero failures, errors, or skips. The [scan](../../../../glossary.md#term-cumulative-scan) semantic closure
defined and invoked all 160 scan [artifacts](../../../../glossary.md#term-artifact) in its 460 aggregate-and-scan [owner](../../../../glossary.md#term-owner) set; the
[checkpoint](../../../../glossary.md#term-checkpoint) [sealed](../../../../glossary.md#term-sealed) the exact generated inventory projection. `CpuScanGeneratedKernelTest` also
exercises all kinds/modes/types against the independent [scalar](../../../../glossary.md#term-scalar) reference, [BFLOAT16](../../../../glossary.md#term-bfloat16)
[round-to-nearest](../../../../glossary.md#floating-point-precision-and-rounding)-[ties-to-even](../../../../glossary.md#term-ties-to-even) after every [operation](../../../../glossary.md#operation), integral [wraparound](../../../../glossary.md#term-wraparound), and a caller-owned
subrange for the guarded reverse [INT64](../../../../glossary.md#data-type--datatype) [`MemorySegment`](../../../../glossary.md#term-memorysegment) product route. [Lowering](../../../../glossary.md#lowering) and [IR](../../../../glossary.md#term-intermediate-representation-ir) tests cover
the five-type [admission](../../../../glossary.md#term-admission) [matrix](../../../../glossary.md#term-matrix-meanings), exact [BOOL](../../../../glossary.md#data-type--datatype)/scalar rejection, normalized-[axis](../../../../glossary.md#term-axis) [identity](../../../../glossary.md#term-identity-meanings), and
[injective](../../../../glossary.md#term-injective) general [layouts](../../../../glossary.md#layout). The preparer/finalizer tests cover the existing cold route-to-executable
[handoff](../../../../glossary.md#term-handoff). Read-only `legacy/pre-rewrite` inspection found no corresponding current scan emitter;
no legacy source was copied or used as a design [dependency](../../../../glossary.md#term-dependency).

Documentation-focused context `/root/cpu_0009d2_docs` independently inspected the final [planning](../../../../glossary.md#term-planning)
diff, current [CPU](../../../../glossary.md#term-cpu) scan route, and cited tests. It confirmed the retained-route decision and the
no-change conclusions below; its planning validation is recorded in this task's completion
summary.

## Implementation notes

No executable Java changed. `CpuScanLowering` continues to produce a two-boundary,
zero-workspace `CpuScanIr`; `CpuPartitionPreparer` and `CpuPartitionFinalizer` continue to retain
the [scan](../../../../glossary.md#term-cumulative-scan) geometry through preparation; `CpuClassFileKernelGenerator` continues to select
`CpuScanEmitter`; and the [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable) invokes the resulting typed entry. Existing
inventory [SHA-256](../../../../glossary.md#term-sha-256) remains
`527f36de41b64c228dd215c6f3182138b4c70db24c915117c4d7f82743ac41cc`.

## Completion summary

- Completed changes: Recorded the full scan route-decision gate and retained the existing
  generated route; no production, test, generated artifact, inventory, or Javadoc path changed.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Files changed or created: This task, parent 0009D, parent 0009, [CPU](../../../../glossary.md#term-cpu) master plan, and roadmap.
- Tests and validation: The seven-class focused CPU command passed 65 tests with zero failures,
  errors, or skips. Documentation-focused context <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`/root/cpu_0009d2_docs` independently inspected
  the listed scan source/tests and final diff; it checked local Markdown links, headings used as
  anchors, fences, terminology, task/order/status synchronization, the exact five-path scope, and<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check`. It did not rerun Java tests because no executable Java changed.
- Documentation-agent review: Clean documentation-focused context `/root/cpu_0009d2_docs`
  finalized these planning records using the General and Planning documentation profiles.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Planning records are substantively revised; public/API Javadoc, guides,
  glossary, architecture/ADR, Gradle, conformance/integration, and other modules have no current
  change because executable behavior, public API, architecture, and reusable terminology are
  unchanged.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: No change; no Java contract or implementation changed.
- Glossary impact: No change; the retained route uses existing terms without changing their
  meanings or boundaries.
- Unresolved issues: None in implementation. D3 remains summary-only and must not receive a
  detailed specification until it becomes the frontier.

Status: Complete
Follow-up required: D3 ordering is the next summary-only frontier; create no detailed D3
specification until it becomes actionable.
