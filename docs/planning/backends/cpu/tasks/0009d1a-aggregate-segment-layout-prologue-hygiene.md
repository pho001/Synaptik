# Task 0009D1A: Aggregate Segment-[Layout](../../../../glossary.md#layout) Prologue Hygiene

## Reader context

This task records a narrow aggregate-entry correction: `CpuCarrierEmitter.prepareSegmentLayouts`
retains predefined typed native-endian unaligned FFM layouts instead of recreating equivalent
layouts. `MemorySegment` is the accessed storage carrier; `ValueLayout` describes how typed reads
and writes access it. `CpuClassFileKernelGenerator` supplies the aggregate-family boundary so
unrelated shared-layout callers retain their prior byte shape.

Here, structural inspection and mutation checks establish removal of redundant prologue calls, while
semantic canaries check preserved execution behavior. Neither proves performance parity or completes
D1's then-outstanding proofs. `CpuGeneratorSchema` version 66 rejects incompatible persistent
generated-code envelopes despite unchanged generated-class identity. Those cache envelopes are
distinct from invocation scratch or model state; Prepare's selected-realization ownership and
Runtime invocation remain unchanged.

Current-topic navigation (not a replacement for the historical record):

- [current generated-class boundary](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit) — Explains emitters, typed entries, hidden classes, and the limits of Class-File inspection.
- [current CPU preparation and invocation handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) — Names the current analysis, shared assignment, finalization, cold binding, and Runtime participants.

## Status

Complete

## Goal

Remove the avoidable per-entry `ByteOrder.nativeOrder()` and `ValueLayout.withOrder(...)` work
from generated aggregate [scalar](../../../../glossary.md#term-scalar) and parallel-scalar entries with [`MemorySegment`](../../../../glossary.md#term-memorysegment) access. The
predefined unaligned `ValueLayout.JAVA_*_UNALIGNED` [layouts](../../../../glossary.md#layout) are already native-endian; generated
entries must retain those typed layouts directly, as optimal clean Java does. This narrow
production prerequisite discovered by [CPU](../../../../glossary.md#term-cpu) 0009D1 review does not complete or promote D1.

## [Scope](../../../../glossary.md#term-scope)

- Correct only aggregate [scalar](../../../../glossary.md#term-scalar)/parallel-scalar segment and mixed entry setup in
  `CpuCarrierEmitter.prepareSegmentLayouts` and, if required to retain that family boundary, its
  `CpuClassFileKernelGenerator` call contract. Segment-free aggregate entries retain no layout
  prologue.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Preserve the typed unaligned native-endian [FFM](../../../../glossary.md#term-ffm) semantics for all six types ([`FLOAT64`](../../../../glossary.md#term-float64),
  [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT64`](../../../../glossary.md#data-type--datatype), [`INT32`](../../../../glossary.md#data-type--datatype), [`BOOL`](../../../../glossary.md#data-type--datatype)), arrays, dense/general layouts, exact aggregate
  algorithms, accumulators/conversions, ordered carrier ABI, range ownership, lowering,
  preparation, selection, and deterministic artifact identity. Preserve unrelated vector,
  parallel-vector, and family behavior.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering); [artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [vector glossary](../../../../glossary.md#term-vector))</span>
- Inspect schema/cache/specialization/binary-name compatibility before changing bytes. Make the
  smallest documented schema bump only if byte compatibility requires it. If bytes change,
  regenerate affected inventory class digests and only exact dependent tests/resources.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [specialization glossary](../../../../glossary.md#term-specialization); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- D1's test-only per-owner clean method/projection binding and operation-specific structural
  accumulator/conversion proof; scan, ordering, fold, and every other family.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [binding glossary](../../../../glossary.md#term-binding); [operation glossary](../../../../glossary.md#operation); [accumulator glossary](../../../../glossary.md#accumulator-type); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- API, support/admission/selection, architecture, algorithms, ABI, layouts, generic shared-layout
  redesign, performance measurement/promotion, broad CPU 0009 checkpoint, Gradle, or refactoring.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [admission glossary](../../../../glossary.md#term-admission); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [layouts glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Architecture references and constraints

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [documentation rules](../../../../developer-guide/documentation-rules.md), [General profile](../../../../developer-guide/documentation/general-style.md), [Planning profile](../../../../developer-guide/documentation/planning-style.md), [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>, [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), [CPU 0009D](0009d-aggregate-scan-ordering-fold-structural-oracle-closure.md), and [CPU 0009D1](0009d1-aggregate-structural-oracle-closure.md).
- Prepare retains lowering and selected-realization ownership. Generated entries execute prepared
  typed inputs and introduce no generic dispatch, allocation, reflection, map/string dispatch,
  boxing, or avoidable virtual/interface dispatch.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Generated Class-Files retain the optimal clean-Java algorithm, hot-loop/dataflow shape, and
  avoidable-overhead profile. This is one-time entry setup reasoning, not a performance-parity
  claim. Stop for an FFM, schema/cache, selection, shared-contract, or architecture decision.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape); [FFM glossary](../../../../glossary.md#term-ffm); [schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache))</span>

## Package impact

Only existing [CPU](../../../../glossary.md#term-cpu)-private `...internal.codegen.emit` paths may own emission and evidence.
`...internal.cache` is touched only for a proved compatibility bump. No public package, [dependency](../../../../glossary.md#term-dependency),
or architecture boundary changes are authorized.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

Expected production paths are `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`,
`backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
(only if needed for aggregate-only behavior), and `.../internal/cache/CpuGeneratorSchema.java`
(only if a compatibility bump is proved). Evidence is the existing aggregate structural [oracle](../../../../glossary.md#term-test-oracle) and
only exact generator/[schema](../../../../glossary.md#term-schema)/inventory [digest](../../../../glossary.md#term-digest) [owners](../../../../glossary.md#term-owner). [Planning](../../../../glossary.md#term-planning) paths are this task, D1, parent D,
master plan, and roadmap. At most three production paths, five [CPU](../../../../glossary.md#term-cpu) test/evidence-resource paths,
and five planning paths may change. No new generic test utility, guide, glossary, architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr),
[conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), or unrelated [module path](../../../../glossary.md#term-module-path) is authorized.

## Acceptance criteria

- Generated aggregate [scalar](../../../../glossary.md#term-scalar)/parallel-scalar segment-bearing entries read the exact typed predefined
  `JAVA_*_UNALIGNED` [layout](../../../../glossary.md#layout) and contain neither `ByteOrder.nativeOrder()` nor
  `ValueLayout.withOrder(...)`. Typed `MemorySegment.get`/`set` calls preserve matching layout
  classes and native-endian behavior; segment-free aggregates acquire no layout prologue.
- Arrays, all six types, dense/general/mixed forms, algorithms, accumulator/conversion semantics,
  stores, ranges, parallel partitioning, ABI/descriptor, selection, and unrelated vector/family
  bytes/behavior remain unchanged.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [descriptor glossary](../../../../glossary.md#term-descriptor); [vector glossary](../../../../glossary.md#term-vector))</span>
- Automated generated-versus-optimal-clean-Java Class-File/decompilation inspection proves removal
  is limited to redundant recreation; a mutation that reintroduces either call fails. It rejects
  generic bridges, allocation, boxing, reflection, map/string dispatch, and unapproved
  virtual/interface dispatch. Record proportionate one-time-entry overhead reasoning only; do not
  claim performance parity.<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Targeted semantic/canary evidence still exercises scalar and parallel-scalar segment/mixed forms.
  If bytes change, regenerate only affected inventory digests and exact dependent tests/resources;
  verify deterministic generation and the schema/cache decision, including stale-envelope rejection
  and regeneration if a bump is required.<span data-glossary-navigation="true"> ([canary glossary](../../../../glossary.md#term-canary); [schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache))</span>
- Review/update `CpuCarrierEmitter` Javadoc for the direct native-endian predefined-layout
  invariant. A separate documentation-focused context finalizes it and records reasoned no-change
  conclusions for public/API Javadoc, guides, glossary, architecture/ADR, Gradle,
  conformance/integration, other modules, and D1's still-unmodified proof work.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [invariant glossary](../../../../glossary.md#term-invariant); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>
- D1 remains Draft and unpromoted. After D1A it resumes its two blockers: non-tautological
  per-owner clean-method/projection binding and operation-specific accumulator/conversion proof.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [binding glossary](../../../../glossary.md#term-binding); [operation glossary](../../../../glossary.md#operation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

After code stabilizes, [run](../../../../glossary.md#term-run) only exact affected [CPU](../../../../glossary.md#term-cpu) tests, with this baseline:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateStructuralOracleTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateScanSemanticClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGeneratorTest
```

If bytes change, add only current inventory/[checkpoint](../../../../glossary.md#term-checkpoint)/evidence and [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) [owners](../../../../glossary.md#term-owner) identified
by [digest](../../../../glossary.md#term-digest)/compatibility inspection. Record Class-File/[decompilation](../../../../glossary.md#term-decompilation) and mutation results,
unchanged/changed digest list, schema decision, changed-path count, Markdown links/anchors/fences/
terminology, and `git diff --check`. Documentation reuses successful Java evidence unless it changes
executable Java, then generates [module](../../../../glossary.md#term-module) [Javadoc](../../../../glossary.md#term-javadoc) after final Javadoc edits. Broad validation and
performance remain CPU 0009G/[CI](../../../../glossary.md#term-continuous-integration-ci).

## Dependencies and follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0009C/documentation pass: Complete; parent D: Draft; D1: Draft with the reviewed hygiene gap.
- Current emitter/generator, aggregate structural-oracle evidence, inventory digest owners, Java 26
  Class-File API, and FFM layouts.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [digest glossary](../../../../glossary.md#term-digest); [Class-File API glossary](../../../../glossary.md#term-class-file-api); [FFM glossary](../../../../glossary.md#term-ffm); [layouts glossary](../../../../glossary.md#layout))</span>

D1A must complete before D1 can complete; D2 remains after D1. D1A does not substitute for either
of D1's two remaining test proofs.

## Architecture impact

Expected impact: None. This is [CPU](../../../../glossary.md#term-cpu)-private generated-entry overhead correction, not changed [FFM](../../../../glossary.md#term-ffm)
semantics, [backend](../../../../glossary.md#backend) selection, or [schema](../../../../glossary.md#term-schema) policy unless byte compatibility requires it.

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
`withOrder(nativeOrder())` creates a distinct [layout](../../../../glossary.md#layout) without changing required access semantics,
and is therefore avoidable entry work. [Schema](../../../../glossary.md#term-schema) 66 makes the changed generated bytes incompatible
with prior persistent envelopes while retaining generated-class [identity](../../../../glossary.md#term-identity-meanings). Production correction
precedes D1's test-only closure.

## Known limitations

No [vector](../../../../glossary.md#term-vector)/family redesign, algorithm change, broad performance evidence, or D1 structural
promotion is included. D1 remains incomplete after D1A until its two named proof blockers pass.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context recorded the focused [CPU](../../../../glossary.md#term-cpu) validation as passing: 45 tests, including the
aggregate structural [oracle](../../../../glossary.md#term-test-oracle), aggregate semantic closure, class-file generator, [cache](../../../../glossary.md#term-cache)/[specialization](../../../../glossary.md#term-specialization),
and generated-coverage catalog [owners](../../../../glossary.md#term-owner). Its Class-File projection showed all 138 changed aggregate
rows are non-[BOOL](../../../../glossary.md#data-type--datatype) entries with at least one [`MemorySegment`](../../../../glossary.md#term-memorysegment) [carrier](../../../../glossary.md#term-storage-carrier); only evidence-key and
class-SHA columns changed. The 162 byte-identical aggregate rows are 150 array-only rows and 12
BOOL `MemorySegment` rows, not 162 array [artifacts](../../../../glossary.md#term-artifact). All 17,163 non-aggregate rows are byte
identical. The current inventory [SHA-256](../../../../glossary.md#term-sha-256) is
`527f36de41b64c228dd215c6f3182138b4c70db24c915117c4d7f82743ac41cc`; the explicit pre-D1A
inventory SHA-256 was
`e821d08cad4e5903510d4b6f700f94708c01b913f9e3a5e8b4e5d6f3bd718c1d`.

[Schema](../../../../glossary.md#term-schema) 66 is current. Earlier persistent envelopes are incompatible misses, and generated-class
[identity](../../../../glossary.md#term-identity-meanings) is unchanged. The aggregate structural projection verifies direct typed
`JAVA_*_UNALIGNED` [layouts](../../../../glossary.md#layout) and rejects reintroduced `ByteOrder.nativeOrder()` or
`ValueLayout.withOrder(...)`; it records proportionate entry-prologue hygiene, not performance
parity. The independent reviewer accepted six-[owner](../../../../glossary.md#term-owner) evidence (43 tests) with no blocker.

Documentation-focused context `/root` reviewed the final emitter [Javadoc](../../../../glossary.md#term-javadoc),
the aggregate test/evidence contract, task D1A, D1/parent [synchronization](../../../../glossary.md#term-synchronization), CPU master plan, and
roadmap. It reused implementation test evidence because its edits are documentation/Javadoc only,
generated CPU Javadoc after final edits, checked Markdown links/anchors/fences/terminology and
whitespace, and ran `git diff --check`.

## Implementation notes

The generator passes the aggregate-family predicate to the shared [layout](../../../../glossary.md#layout) setup. Only aggregate
[scalar](../../../../glossary.md#term-scalar) and parallel-scalar segment-bearing entries retain typed predefined unaligned layouts
directly; all other shared-layout callers preserve their prior byte [shape](../../../../glossary.md#shape). The [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) change
is required because 138 aggregate generated classes changed bytes, while their generated [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)
names and [specialization](../../../../glossary.md#term-specialization) [identity](../../../../glossary.md#term-identity-meanings) remain stable.

## Completion summary

- Completed changes: Aggregate scalar and parallel-scalar segment setup now retains predefined
  typed unaligned native-endian layouts directly; schema 66 invalidates stale persistent envelopes.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [layouts glossary](../../../../glossary.md#layout); [schema glossary](../../../../glossary.md#term-schema))</span>
- Files changed or created: `CpuCarrierEmitter`, `CpuClassFileKernelGenerator`,
  `CpuGeneratorSchema`, exact aggregate/inventory evidence owners, this task, D1/parent planning,
  CPU master plan, and roadmap.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Tests and validation: Reused implementation context's focused 45-test CPU result and independent
  reviewer's 43-test six-owner result; Class-File/decompilation and mutation evidence passed.
  Documentation finalization generated CPU Javadoc and passed Markdown/static checks and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [owner glossary](../../../../glossary.md#term-owner); [decompilation glossary](../../../../glossary.md#term-decompilation))</span>
  `git diff --check`.
- Documentation-agent review: `/root` finalized the emitter Javadoc and planning
  evidence/status synchronization in this same change.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Documentation impact: No public/API guide change: this is a CPU-private generated-entry setup
  correction with unchanged public behavior and workflow.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: Updated `CpuCarrierEmitter.prepareSegmentLayouts` to identify aggregate scalar
  and parallel-scalar direct-layout use and the boolean parameter's boundary.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Glossary impact: No change; no reusable project term or meaning changed.
- No-change conclusions: Architecture documentation/ADR and architecture tests remain unchanged
  because ownership, dependencies, and FFM semantics did not change. Gradle, conformance and
  integration tests, other modules, and performance documentation remain unchanged because the
  correction is CPU-private, has focused CPU coverage, and supplies no performance measurement.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [FFM glossary](../../../../glossary.md#term-ffm); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>
- Unresolved issues: D1 remains Draft and unpromoted.
- Follow-up required: D1 must now complete non-tautological per-owner clean-method/projection
  binding and operation-specific accumulator/conversion proof before D2 scan work.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [operation glossary](../../../../glossary.md#operation); [accumulator glossary](../../../../glossary.md#accumulator-type); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>

Status: Complete
