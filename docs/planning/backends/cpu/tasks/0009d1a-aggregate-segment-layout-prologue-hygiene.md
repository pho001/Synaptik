# Task 0009D1A: Aggregate Segment-Layout Prologue Hygiene

## Status

Complete

## Goal

Remove the avoidable per-entry `ByteOrder.nativeOrder()` and `ValueLayout.withOrder(...)` work
from generated aggregate scalar and parallel-scalar entries with `MemorySegment` access. The
predefined unaligned `ValueLayout.JAVA_*_UNALIGNED` layouts are already native-endian; generated
entries must retain those typed layouts directly, as optimal clean Java does. This narrow
production prerequisite discovered by CPU 0009D1 review does not complete or promote D1.

## Scope

- Correct only aggregate scalar/parallel-scalar segment and mixed entry setup in
  `CpuCarrierEmitter.prepareSegmentLayouts` and, if required to retain that family boundary, its
  `CpuClassFileKernelGenerator` call contract. Segment-free aggregate entries retain no layout
  prologue.
- Preserve the typed unaligned native-endian FFM semantics for all six types (`FLOAT64`,
  `FLOAT32`, `BFLOAT16`, `INT64`, `INT32`, `BOOL`), arrays, dense/general layouts, exact aggregate
  algorithms, accumulators/conversions, ordered carrier ABI, range ownership, lowering,
  preparation, selection, and deterministic artifact identity. Preserve unrelated vector,
  parallel-vector, and family behavior.
- Inspect schema/cache/specialization/binary-name compatibility before changing bytes. Make the
  smallest documented schema bump only if byte compatibility requires it. If bytes change,
  regenerate affected inventory class digests and only exact dependent tests/resources.

## Out of scope

- D1's test-only per-owner clean method/projection binding and operation-specific structural
  accumulator/conversion proof; scan, ordering, fold, and every other family.
- API, support/admission/selection, architecture, algorithms, ABI, layouts, generic shared-layout
  redesign, performance measurement/promotion, broad CPU 0009 checkpoint, Gradle, or refactoring.

## Architecture references and constraints

- [Architecture contract](../../../../../ARCHITECTURE.md), [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md), [General profile](../../../../developer-guide/documentation/general-style.md), [Planning profile](../../../../developer-guide/documentation/planning-style.md), [CPU master plan](../master-plan.md), [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), [CPU 0009D](0009d-aggregate-scan-ordering-fold-structural-oracle-closure.md), and [CPU 0009D1](0009d1-aggregate-structural-oracle-closure.md).
- Prepare retains lowering and selected-realization ownership. Generated entries execute prepared
  typed inputs and introduce no generic dispatch, allocation, reflection, map/string dispatch,
  boxing, or avoidable virtual/interface dispatch.
- Generated Class-Files retain the optimal clean-Java algorithm, hot-loop/dataflow shape, and
  avoidable-overhead profile. This is one-time entry setup reasoning, not a performance-parity
  claim. Stop for an FFM, schema/cache, selection, shared-contract, or architecture decision.

## Package impact

Only existing CPU-private `...internal.codegen.emit` paths may own emission and evidence.
`...internal.cache` is touched only for a proved compatibility bump. No public package, dependency,
or architecture boundary changes are authorized.

## Affected files and maximum scope

Expected production paths are `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`,
`backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
(only if needed for aggregate-only behavior), and `.../internal/cache/CpuGeneratorSchema.java`
(only if a compatibility bump is proved). Evidence is the existing aggregate structural oracle and
only exact generator/schema/inventory digest owners. Planning paths are this task, D1, parent D,
master plan, and roadmap. At most three production paths, five CPU test/evidence-resource paths,
and five planning paths may change. No new generic test utility, guide, glossary, architecture/ADR,
conformance/integration, Gradle, or unrelated module path is authorized.

## Acceptance criteria

- Generated aggregate scalar/parallel-scalar segment-bearing entries read the exact typed predefined
  `JAVA_*_UNALIGNED` layout and contain neither `ByteOrder.nativeOrder()` nor
  `ValueLayout.withOrder(...)`. Typed `MemorySegment.get`/`set` calls preserve matching layout
  classes and native-endian behavior; segment-free aggregates acquire no layout prologue.
- Arrays, all six types, dense/general/mixed forms, algorithms, accumulator/conversion semantics,
  stores, ranges, parallel partitioning, ABI/descriptor, selection, and unrelated vector/family
  bytes/behavior remain unchanged.
- Automated generated-versus-optimal-clean-Java Class-File/decompilation inspection proves removal
  is limited to redundant recreation; a mutation that reintroduces either call fails. It rejects
  generic bridges, allocation, boxing, reflection, map/string dispatch, and unapproved
  virtual/interface dispatch. Record proportionate one-time-entry overhead reasoning only; do not
  claim performance parity.
- Targeted semantic/canary evidence still exercises scalar and parallel-scalar segment/mixed forms.
  If bytes change, regenerate only affected inventory digests and exact dependent tests/resources;
  verify deterministic generation and the schema/cache decision, including stale-envelope rejection
  and regeneration if a bump is required.
- Review/update `CpuCarrierEmitter` Javadoc for the direct native-endian predefined-layout
  invariant. A separate documentation-focused context finalizes it and records reasoned no-change
  conclusions for public/API Javadoc, guides, glossary, architecture/ADR, Gradle,
  conformance/integration, other modules, and D1's still-unmodified proof work.
- D1 remains Draft and unpromoted. After D1A it resumes its two blockers: non-tautological
  per-owner clean-method/projection binding and operation-specific accumulator/conversion proof.

## Tests / validation

After code stabilizes, run only exact affected CPU tests, with this baseline:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateStructuralOracleTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateScanSemanticClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGeneratorTest
```

If bytes change, add only current inventory/checkpoint/evidence and schema/cache owners identified
by digest/compatibility inspection. Record Class-File/decompilation and mutation results,
unchanged/changed digest list, schema decision, changed-path count, Markdown links/anchors/fences/
terminology, and `git diff --check`. Documentation reuses successful Java evidence unless it changes
executable Java, then generates module Javadoc after final Javadoc edits. Broad validation and
performance remain CPU 0009G/CI.

## Dependencies and follow-up tasks

- CPU 0009C/documentation pass: Complete; parent D: Draft; D1: Draft with the reviewed hygiene gap.
- Current emitter/generator, aggregate structural-oracle evidence, inventory digest owners, Java 26
  Class-File API, and FFM layouts.

D1A must complete before D1 can complete; D2 remains after D1. D1A does not substitute for either
of D1's two remaining test proofs.

## Architecture impact

Expected impact: None. This is CPU-private generated-entry overhead correction, not changed FFM
semantics, backend selection, or schema policy unless byte compatibility requires it.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009D1A. Read AGENTS.md, ARCHITECTURE.md,
the current architecture plan, Planning Guide, documentation rules and General/Planning profiles,
CPU master plan, CPU 0009/D/D1, this task, the emitter/generator, and aggregate oracle evidence.
Correct only redundant native-order/layout recreation in generated aggregate scalar/parallel-scalar
segment setup. Preserve typed native-endian FFM semantics and every boundary stated here. Inspect
schema/identity before bytes change; regenerate only affected digests/tests when needed. Do not
implement D1's remaining proofs, claim performance parity, commit, or push. Hand stable executable
evidence to a separate clean documentation-focused context before marking Complete.
```

## Local decisions

`JAVA_*_UNALIGNED` is the typed native-endian source of truth. Applying
`withOrder(nativeOrder())` creates a distinct layout without changing required access semantics,
and is therefore avoidable entry work. Schema 66 makes the changed generated bytes incompatible
with prior persistent envelopes while retaining generated-class identity. Production correction
precedes D1's test-only closure.

## Known limitations

No vector/family redesign, algorithm change, broad performance evidence, or D1 structural
promotion is included. D1 remains incomplete after D1A until its two named proof blockers pass.

## Validation evidence

Implementation context recorded the focused CPU validation as passing: 45 tests, including the
aggregate structural oracle, aggregate semantic closure, class-file generator, cache/specialization,
and generated-coverage catalog owners. Its Class-File projection showed all 138 changed aggregate
rows are non-BOOL entries with at least one `MemorySegment` carrier; only evidence-key and
class-SHA columns changed. The 162 byte-identical aggregate rows are 150 array-only rows and 12
BOOL `MemorySegment` rows, not 162 array artifacts. All 17,163 non-aggregate rows are byte
identical. The current inventory SHA-256 is
`527f36de41b64c228dd215c6f3182138b4c70db24c915117c4d7f82743ac41cc`; the explicit pre-D1A
inventory SHA-256 was
`e821d08cad4e5903510d4b6f700f94708c01b913f9e3a5e8b4e5d6f3bd718c1d`.

Schema 66 is current. Earlier persistent envelopes are incompatible misses, and generated-class
identity is unchanged. The aggregate structural projection verifies direct typed
`JAVA_*_UNALIGNED` layouts and rejects reintroduced `ByteOrder.nativeOrder()` or
`ValueLayout.withOrder(...)`; it records proportionate entry-prologue hygiene, not performance
parity. The independent reviewer accepted six-owner evidence (43 tests) with no blocker.

Documentation-focused context `/root` reviewed the final emitter Javadoc,
the aggregate test/evidence contract, task D1A, D1/parent synchronization, CPU master plan, and
roadmap. It reused implementation test evidence because its edits are documentation/Javadoc only,
generated CPU Javadoc after final edits, checked Markdown links/anchors/fences/terminology and
whitespace, and ran `git diff --check`.

## Implementation notes

The generator passes the aggregate-family predicate to the shared layout setup. Only aggregate
scalar and parallel-scalar segment-bearing entries retain typed predefined unaligned layouts
directly; all other shared-layout callers preserve their prior byte shape. The schema/cache change
is required because 138 aggregate generated classes changed bytes, while their generated binary
names and specialization identity remain stable.

## Completion summary

- Completed changes: Aggregate scalar and parallel-scalar segment setup now retains predefined
  typed unaligned native-endian layouts directly; schema 66 invalidates stale persistent envelopes.
- Files changed or created: `CpuCarrierEmitter`, `CpuClassFileKernelGenerator`,
  `CpuGeneratorSchema`, exact aggregate/inventory evidence owners, this task, D1/parent planning,
  CPU master plan, and roadmap.
- Tests and validation: Reused implementation context's focused 45-test CPU result and independent
  reviewer's 43-test six-owner result; Class-File/decompilation and mutation evidence passed.
  Documentation finalization generated CPU Javadoc and passed Markdown/static checks and
  `git diff --check`.
- Documentation-agent review: `/root` finalized the emitter Javadoc and planning
  evidence/status synchronization in this same change.
- Documentation impact: No public/API guide change: this is a CPU-private generated-entry setup
  correction with unchanged public behavior and workflow.
- Javadoc review: Updated `CpuCarrierEmitter.prepareSegmentLayouts` to identify aggregate scalar
  and parallel-scalar direct-layout use and the boolean parameter's boundary.
- Glossary impact: No change; no reusable project term or meaning changed.
- No-change conclusions: Architecture documentation/ADR and architecture tests remain unchanged
  because ownership, dependencies, and FFM semantics did not change. Gradle, conformance and
  integration tests, other modules, and performance documentation remain unchanged because the
  correction is CPU-private, has focused CPU coverage, and supplies no performance measurement.
- Unresolved issues: D1 remains Draft and unpromoted.
- Follow-up required: D1 must now complete non-tautological per-owner clean-method/projection
  binding and operation-specific accumulator/conversion proof before D2 scan work.

Status: Complete
