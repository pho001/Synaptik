# Task 0009C1: CONCAT and STACK Generated-Entry Hot-Path Hygiene

## Reader context

This task records the narrow CONCAT/STACK production correction that unblocked 0009C's 48
composition rows. `CpuDataMovementEmitter` replaces an impossible terminal exception with the final
statically prepared occurrence arm. Prepare's exhaustive occurrence partition justifies that arm; it
is not a runtime catch-all. Independent clean-Java execution checks represented-bit semantics, while
opcode scans and paired topology evidence address structural hygiene.

`CpuGeneratorSchema` advances compatibility from 64 to 65, and `CpuGeneratedKernelArtifactStoreTest`
checks stale-envelope rejection and subsequent persisted hits. Inventory `DENSE_LINEAR` does not
imply `DENSE_HEAP_ARRAY_INT`: segment-contiguous entries can use `GENERAL_LONG`. Normalized
structural equality was not proved, so the bounded generated-versus-direct fallback protocol ran.
Its eight representative cases do not establish broad performance closure or authorize production
strategy defaults.

Current-topic navigation (not a replacement for the historical record):

- [current generated-class boundary](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit) — Explains emitters, typed entries, hidden classes, and the limits of Class-File inspection.
- [current CPU preparation and invocation handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) — Names the current analysis, shared assignment, finalization, cold binding, and Runtime participants.

## Status

Complete

## Goal

Correct the narrow production prerequisite discovered while reviewing [CPU](../../../../glossary.md#term-cpu) 0009C: every selected
generated `CONCAT` and `STACK` entry must preserve the existing [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) composition
semantics and clean-Java-equivalent loop/[dataflow](../../../../glossary.md#term-dataflow) while containing no avoidable terminal
`IllegalStateException` allocation, constructor invocation, or `ATHROW` path. This correction
unblocks CPU 0009C's 48-row CONCAT/STACK hygiene closure; it does not broaden movement support.

## [Scope](../../../../glossary.md#term-scope)

- Inspect the selected generated entries for all 24 `CONCAT` and 24 `STACK` rows. Preserve and
  assert two separate source-derived cross-tabs, without aggregate masking: the checked inventory
  access regimes are 12 `DENSE_LINEAR` plus 12 `GENERAL_ODOMETER` rows for each form, while the
  actual `CpuKernelSpecialization.loopAddressing` emitter paths are 6
  `DENSE_HEAP_ARRAY_INT` plus 18 `GENERAL_LONG` rows for each form. The six dense-but-segment-
  contiguous rows select <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>`GENERAL_LONG`; they must not be misreported as dense heap-array paths.
  Replace only the impossible-after-preparation terminal path with
  a verifier-valid exhaustive selection chain. For each statically prepared occurrence map of
  length <span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>`n`, emit comparisons only for occurrences `0 .. n - 2`; use [occurrence](../../../../glossary.md#term-occurrence) `n - 1` as the
  unconditional terminal assignment/path. For CONCAT, each earlier arm tests its prepared prefix
  boundary and the final arm receives the remaining coordinate; for STACK, each earlier arm tests
  its prepared occurrence coordinate and the final arm is unconditional. Apply this topology to<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [topology glossary](../../../../glossary.md#term-topology))</span>
  `emitDenseConcat`, `emitDenseStack`, `emitConcat`, and `emitStack`, and to the independently
  `javac`-compiled clean oracle. Prepare proves that the prefixes/occurrence coordinates partition
  every admitted invocation exhaustively, so this is neither runtime validation nor silent default
  semantics. The sole permitted unconditional arm is the statically prepared final occurrence;
  it must assign the typed source/value before the common store. Do not use a dummy/default value,
  a <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [Prepare glossary](../../../../glossary.md#prepare); [partition glossary](../../../../glossary.md#partition); [runtime glossary](../../../../glossary.md#term-runtime); [validation glossary](../../../../glossary.md#term-validation))</span>`switch` default branch, an exception path, or any catch-all semantics as that final
  occurrence, and do not read an uninitialized local.
- Preserve occurrence order (including repeated inputs), ordered typed carriers, axis/coordinate
  mapping, represented-bit loads/stores, range and sentinel behavior, dense-int and general-long
  cursor/loop shape, and all existing cold guards/fallback selection.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [shape glossary](../../../../glossary.md#shape))</span>
- Add or tighten automated semantic, Class-File/decompilation, and mutation evidence so both the
  generated selected entry and independently compiled clean-oracle entry reject <span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation))</span>`NEW`,
  `ANEWARRAY`, `NEWARRAY`, `MULTIANEWARRAY`, `INVOKEDYNAMIC`, `ATHROW`, and
  `IllegalStateException.<init>`, rather than merely filtering invocations. The same automated
  member scan rejects generated/reference/oracle/helper leakage, reflection, map/string or generic
  carrier dispatch, boxing, and unapproved virtual/interface dispatch; only the exact typed array
  opcodes or explicitly justified <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) calls may remain. Verify the clean Java has the
  same all-but-final-test/unconditional-final topology.
- Inspect `CpuGeneratorSchema`, `CpuKernelSpecialization` identity/compatibility bytes, generated
  binary naming, persistent-cache behavior, and exact fixture digest resources before changing
  bytes. The generated-byte change advances <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [cache glossary](../../../../glossary.md#term-cache); [fixture glossary](../../../../glossary.md#term-fixture); [digest glossary](../../../../glossary.md#term-digest))</span>`CURRENT_VERSION` from 64 to 65. Retain the existing
  class-identity projection, structural key, and generated binary name only if automated evidence
  proves that the existing family projection remains semantically correct; otherwise update their
  owned contract. Update literal schema assertions, implementation Javadocs, cache expectations,
  and affected resources/digest-ledger evidence. For real CONCAT and STACK fixtures, prove that a
  persisted same-key schema-64 envelope is rejected and regenerated, then that schema-65 persists
  and is subsequently a persisted hit.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Any new operation, provider capability, preparer admission/selection rule, IR/lowering design,
  materialization policy, public API, architecture contract, native route, or broad CPU 0009C
  structural-evidence redesign.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [admission glossary](../../../../glossary.md#term-admission); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [architecture contract glossary](../../../../glossary.md#architecture-contract); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Altering `CONCAT`/`STACK` semantics, validation boundaries, legal input geometry, selected
  strategy, fallback coverage, range ownership, or existing tests merely to make the hygiene check
  pass. Do not weaken a test, omit a selected realization, or allowlist the constructor/allocation
  terminal path.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- CPU 0009C's affine, other movement, indexing, scatter, random, or full 2,252-row closure;
  CPU 0009D--0009G; literal byte-for-byte, `javac`, CFG, or JIT identity; and a new broad
  performance campaign.<span data-glossary-navigation="true"> ([CFG glossary](../../../../glossary.md#term-control-flow-graph-cfg); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): backend prepare owns lowering and kernel
  selection; generated code must preserve the optimal clean-Java algorithm, hot-loop/dataflow
  shape, and avoidable-overhead profile.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape))</span>
- [Current architecture navigation](../../../../architecture/current-architecture-plan.md),
  [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [documentation rules](../../../../developer-guide/documentation-rules.md),
  [General profile](../../../../developer-guide/documentation/general-style.md),
  [Planning profile](../../../../developer-guide/documentation/planning-style.md), [CPU master
  plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>, parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md),
  completed [CPU 0009B](0009b-ordinary-pointwise-and-cast-structural-oracle-closure.md), and
  blocked [CPU 0009C](0009c-affine-movement-indexing-scatter-random-structural-oracle-closure.md).

## Architecture constraints

- Prepare remains the owner of supported, validated lowering and selected realization. Generated
  entries execute prepared inputs only; they must not acquire Model interpretation, generic
  validation, reflection, maps, boxing, allocation, helper bridges, or avoidable virtual/interface
  dispatch.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [owner glossary](../../../../glossary.md#term-owner); [lowering glossary](../../../../glossary.md#lowering); [Model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- A well-written specialized clean-Java oracle is the performance and structural review oracle.
  Preserve its composition selection and primitive loop/dataflow shape; a dead exceptional path is
  not permitted as a substitute for a prepared-input invariant.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- Schema/cache identity is a correctness boundary. This generated-byte change advances the
  current-only compatibility schema from 64 to 65. Inspect class identity, structural key,
  generated binary naming, cache load/miss behavior, and SHA-bound evidence; retain the existing
  family projection only with automated proof that it remains semantically correct. Prove that
  stale same-key schema-64 CONCAT/STACK envelopes cannot be reused and that schema-65 envelopes
  regenerate, persist, and subsequently hit.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>

## Package impact

Existing production packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` owns direct movement bytecode
  emission and generated-entry evidence.<span data-glossary-navigation="true"> ([bytecode emission glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` owns generator compatibility and
  generated-artifact identity.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

Packages added or changed:

- None. No public package, [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), or architecture boundary changes.

Type placement:

- `...emit.CpuDataMovementEmitter` — owns only the selected CONCAT/STACK [bytecode](../../../../glossary.md#term-bytecode) correction.
- `...cache.CpuGeneratorSchema` — advances `CURRENT_VERSION` from 64 to 65 and updates its
  implementation Javadoc for the byte-shaping CONCAT/STACK topology correction.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [topology glossary](../../../../glossary.md#term-topology))</span>
- `...cache.CpuKernelSpecialization` — retains or changes its existing class-identity projection,
  structural key, and generated binary name only as automated evidence proves correct.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- `...emit.CpuAffineMovementIndexingScatterRandomStructuralOracleTest` and existing movement
  semantic/evidence fixtures — own exact selected-entry semantic, hygiene, mutation, and
  digest synchronization evidence; add no generic production bridge.<span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- `...emit.CpuAffineMovementIndexingScatterRandomCleanJavaOracle` — owns only the independently
  compiled counterpart source and must use the same verifier-valid all-but-final-test and
  statically prepared unconditional-final-occurrence topology, never a <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>`switch` default,
  initialized fallback value, exception path, or catch-all semantics.

## Affected files

Actual production, build, test, resource, and evidence [fan-out](../../../../glossary.md#term-fan-out):

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`, only if inspected compatibility invalidation requires it
- `backends/cpu/build.gradle.kts` — forwards the explicit opt-in performance properties to the
  isolated harness; it does not change ordinary test or production behavior.
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineMovementIndexingScatterRandomStructuralOracleTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineMovementIndexingScatterRandomCleanJavaOracle.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryMovementFoldSemanticClosureTest.java`, only if its direct selected-entry composition [fixture](../../../../glossary.md#term-fixture) needs exact [digest](../../../../glossary.md#term-digest) [synchronization](../../../../glossary.md#term-synchronization)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`, `CpuGeneratedCoverageCheckpointTest.java`, and `CpuGeneratedCoverageEvidenceTest.java`, only for the concrete generated-name/definition, [registry](../../../../glossary.md#term-registry), or digest evidence their current assertions own
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java` and `CpuGeneratedKernelArtifactStoreTest.java`, only for the exact composition [identity](../../../../glossary.md#term-identity-meanings)/[schema](../../../../glossary.md#term-schema) and persisted hit/miss/stale-envelope assertions their current packages own
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConcatStackPerformanceTest.java`
  — the required fallback-performance harness and source-derived representative mapping.
- All existing [CPU](../../../../glossary.md#term-cpu) test files with a literal `assertEquals(64,
  CpuGeneratorSchema.CURRENT_VERSION)` (or equivalent fully qualified assertion) may receive the
  mechanical current-schema update to `65` when that is their only task-owned change. The same
  allowance covers only stale-schema wording or Javadoc associated with that literal expectation;
  it does not authorize behavioral, fixture, family, or unrelated cleanup in those files.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/cpu-0009c1-post-terminal-frozen-layout-transition.tsv`
  — reversible 24-row transition evidence.
- The canonical inventory, disposition registry, evidence ledger, and their exact inventory-SHA
  consumers — regenerated only for schema-65 identity facts.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `backends/cpu/evidence/cpu-0009c1-concat-stack-performance/` — the sealed, 63-file retained
  fallback-performance bundle and its <span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed))</span>`SHA256SUMS` [manifest](../../../../glossary.md#term-manifest).

[Planning](../../../../glossary.md#term-planning) paths: this task, CPU 0009C, parent CPU 0009, CPU master plan, and roadmap.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may modify at most two production Java paths, one build path, twenty-two [CPU](../../../../glossary.md#term-cpu) Java-test
paths, five checked CPU resource paths, the [sealed](../../../../glossary.md#term-sealed) 63-file performance-evidence tree, and five
[planning](../../../../glossary.md#term-planning) paths. The expanded [fan-out](../../../../glossary.md#term-fan-out) is a discovered required consequence of the mandatory
fallback performance protocol, the frozen-[layout](../../../../glossary.md#layout) correction, [schema](../../../../glossary.md#term-schema) invalidation, and checked-in
inventory [integrity](../../../../glossary.md#term-integrity); it authorizes no unrelated refactor. The Java-test bound includes the twelve
literal-schema assertions, composition/[cache](../../../../glossary.md#term-cache)/structural [owners](../../../../glossary.md#term-owner), and the performance harness. The
resource bound includes the [canonical](../../../../glossary.md#term-canonical-form) inventory and exact SHA consumers, ledger/[registry](../../../../glossary.md#term-registry) joins,
and reversible transition resource. The prior eighteen-path estimate did not include those
required evidence owners. Across the literal-assertion files, the only permitted task-owned
change is the mechanical `CURRENT_VERSION`/schema expectation update from `64` to `65` and
associated stale-schema wording or [Javadoc](../../../../glossary.md#term-javadoc). This task may not modify [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/
[Runtime](../../../../glossary.md#term-runtime)/[Engine](../../../../glossary.md#term-engine) APIs, architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr), [conformance](../../../../glossary.md#term-conformance)/integration tests, glossary, guides, or
unrelated [modules](../../../../glossary.md#term-module). If the correction needs a new production abstraction, changed [admission](../../../../glossary.md#term-admission)/
selection semantics, or another family, stop and propose a separate Draft task.

## Acceptance criteria

- All 48 exact checked CONCAT/STACK rows deterministically generate, define/load, and invoke their
  actual typed selected entry across applicable array/segment/mixed carrier roles, inventory access,
  actual emitter path, full/empty/interior/tail ranges, and occurrence maps. They retain bit-exact
  results and untouched sentinels against independently authored specialized clean Java. Automated
  accounting preserves two exact form × source-derived cross-tabs, with owner IDs in every cell:
  inventory access reports 12 <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [occurrence glossary](../../../../glossary.md#term-occurrence); [owner glossary](../../../../glossary.md#term-owner))</span>`DENSE_LINEAR` plus 12 `GENERAL_ODOMETER` rows for CONCAT and the
  same 12/12 split for STACK; actual `loopAddressing` emitter paths report 6
  `DENSE_HEAP_ARRAY_INT` plus 18 `GENERAL_LONG` rows for CONCAT and the same 6/18 split for STACK.
  The six segment-contiguous dense rows per form belong to inventory <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>`DENSE_LINEAR` but emitter
  `GENERAL_LONG`. An aggregate count may not mask a missing applicable cell; a later absent cell
  requires its source-derived inapplicability reason.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Class-File inspection and decompilation of every generated selected entry and every compiled
  clean-oracle entry prove the all-but-final-test/unconditional-final selection topology and no<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation); [oracle glossary](../../../../glossary.md#term-test-oracle); [topology glossary](../../../../glossary.md#term-topology))</span>
  `NEW`, `ANEWARRAY`, `NEWARRAY`, `MULTIANEWARRAY`, `INVOKEDYNAMIC`, `ATHROW`,
  `IllegalStateException.<init>`, generic/helper bridge, reflection, map/string dispatch, boxing,
  or avoidable virtual/interface dispatch. Any narrowly permitted primitive/FFM carrier call is
  explicitly tied to its typed selected carrier. The test fails if a forbidden terminal path,
  dummy/initialized fallback values, <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [FFM glossary](../../../../glossary.md#term-ffm))</span>`switch`-default or exception paths, or unapproved member
  survives; the required unconditional final statically prepared occurrence arm remains
  permitted. Opcode evidence, not invoke-only filtering, is required.<span data-glossary-navigation="true"> ([Opcode glossary](../../../../glossary.md#term-opcode))</span>
- The comparison proves composition topology separately for CONCAT boundary selection and STACK
  occurrence selection, including repeated inputs and axis insertion/removal. It preserves loop
  direction, range guard, coordinate/address progression, typed load/store, and dense-int versus
  general-long dataflow rather than comparing names, hashes, catalog rows, or shared helpers.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [guard glossary](../../../../glossary.md#term-guard); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>
- Mutation controls fail for reintroducing `NEW`, constructor invocation, or `ATHROW`; changed
  occurrence order/boundary, axis coordinate map, carrier ABI/order, range/tail direction,
  represented-bit conversion/store, and an allowed invocation target. No mutation can pass through
  self-comparison, a stale digest, or test weakening.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [target glossary](../../../../glossary.md#term-target); [digest glossary](../../../../glossary.md#term-digest))</span>
- Generated bytes advance `CpuGeneratorSchema.CURRENT_VERSION` from 64 to 65. Automated evidence
  proves whether the existing class-identity projection, structural key, and generated binary name
  remain semantically correct; it refreshes literal schema assertions, implementation Javadocs,
  cache expectations, fixture SHA-256s, and ledgers as applicable. All twelve existing CPU tests
  with literal current-schema assertions update their <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [fixture glossary](../../../../glossary.md#term-fixture); [CPU glossary](../../../../glossary.md#term-cpu))</span>`64` expectation to `65` when that mechanical
  expectation (and any associated stale-schema wording/Javadoc) is their only task-owned change.
  Behavioral edits remain limited to the named composition/cache/structural tests. Real CONCAT and
  STACK fixtures prove same-key schema-64 persisted-envelope rejection and regeneration, followed
  by schema-65 persistence and a persisted hit.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Make the performance decision before timing with a normalized generated-versus-clean-oracle
  projection for every form × actual emitter path × carrier shape. Compare either before/after
  projections or a frozen expected projection. It retains ordered selection comparisons, the
  unconditional final statically prepared occurrence arm, coordinate and source-address
  expressions, typed carrier loads, output store, range guard/back-edge, invocations, and opcodes;
  it ignores only labels, local numbers, constant-pool indices, and the removed unreachable
  exception sequence. Active mutations fail if any retained live fact changes or if the removed
  terminal exception sequence returns or reappears. If that proof fails, run the repository's matching generated-versus-
  optimal-direct protocol from
  CPU 0007A1C for every affected CONCAT/STACK form × actual emitter path × carrier shape case:
  separate identical
  preallocated carriers/geometry/ranges, exact semantic/canary checks, matching method handles,
  five isolated Java 26 forks with <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [constant glossary](../../../../glossary.md#term-constant); [run glossary](../../../../glossary.md#term-run); [canary glossary](../../../../glossary.md#term-canary))</span>`-Xms1g -Xmx1g`, `-XX:-TieredCompilation`, and `-Xbatch`, at
  least five randomized warmups, seven randomized measured rounds, adaptive batches of at least
  25 ms on each side, balanced generated/direct order, per-fork medians, and the median
  of fork medians. Every per-case fork ratio and aggregate generated/direct ratio must be<span data-glossary-navigation="true"> ([batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [fork glossary](../../../../glossary.md#term-fork-meanings); [median glossary](../../../../glossary.md#term-median))</span>
  `<= 1.15x`; a missing, interrupted, noisy, or over-threshold result is fail-closed and leaves
  the task incomplete. If the normalized projection proves only removal of the unreachable terminal,
  do not measure: add automated Class-File evidence showing that exact fact for every form × actual
  emitter path × carrier shape and record why no new measurement is warranted. This task does not claim or
  launch CPU 0009G's broad campaign.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- A separate documentation-focused clean context finalizes changed implementation Javadocs and
  planning documentation, and records reasoned no-change conclusions for public/API Javadocs,
  guides, glossary, architecture/ADRs, architecture/conformance/integration tests, Gradle, and
  unrelated modules.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation context runs the focused production/evidence suite after executable code stabilizes:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuOrdinaryMovementFoldSemanticClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAffineMovementIndexingScatterRandomStructuralOracleTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGeneratorTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageCheckpointTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageEvidenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest
```

The extended structural-[oracle](../../../../glossary.md#term-test-oracle) test must [run](../../../../glossary.md#term-run) the automated generated-and-clean [opcode](../../../../glossary.md#term-opcode)/member scan,
selection-[topology](../../../../glossary.md#term-topology) check, mutations, and both exact source-derived cross-tabs: inventory access
(`DENSE_LINEAR`/`GENERAL_ODOMETER`) and actual `loopAddressing` emitter path
(`DENSE_HEAP_ARRAY_INT`/`GENERAL_LONG`). The extended
`CpuClassFileKernelGeneratorTest`/`CpuKernelSpecializationTest` must prove the applicable class
[identity](../../../../glossary.md#term-identity-meanings) projection, deterministic generated [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) name, [descriptor](../../../../glossary.md#term-descriptor), and definition behavior.
The extended `CpuGeneratedKernelArtifactStoreTest` must use real CONCAT and STACK fixtures to prove
[schema](../../../../glossary.md#term-schema)-64 same-key stale-envelope rejection/re-generation, then schema-65 persisted generated and
hit behavior. `CpuGeneratedCoverageCheckpointTest`
and `CpuGeneratedCoverageEvidenceTest` must prove checked inventory/disposition/evidence-ledger
SHA/[digest](../../../../glossary.md#term-digest) joins remain synchronized. Use `javap` only as recorded supplementary inspection, not as
a replacement for automated evidence. The normalized-projection gate decides objectively whether
the bounded five-[fork](../../../../glossary.md#term-fork-meanings) command above is required.

Documentation context reuses successful Java evidence unless it changes executable Java. It runs
the affected [module](../../../../glossary.md#term-module) [Javadoc](../../../../glossary.md#term-javadoc) task after final Javadoc edits, checks Markdown links/anchors/fences,
status/[dependency](../../../../glossary.md#term-dependency) [synchronization](../../../../glossary.md#term-synchronization), terminology, whitespace, path [scope](../../../../glossary.md#term-scope), and:

```bash
git diff --check
```

Repository-wide/capability-[checkpoint](../../../../glossary.md#term-checkpoint) validation and broad representative performance closure defer
to [CPU](../../../../glossary.md#term-cpu) 0009G/[CI](../../../../glossary.md#term-continuous-integration-ci).

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0009B and its documentation pass.
- Current checked CPU 0009C inventory, semantic fixture, structural-oracle test, evidence registry,
  emitter, cache identity, and generated schema contracts.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [oracle glossary](../../../../glossary.md#term-test-oracle); [registry glossary](../../../../glossary.md#term-registry); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [schema glossary](../../../../glossary.md#term-schema))</span>
- Java 26 Class-File, [Foreign Function and Memory API](../../../../glossary.md#application-binary-interface-and-native-bridge), and CPU generated-code test environment.

## Follow-up tasks

- Required: CPU 0009C resumes only after this task is Complete and its documentation pass records
  the 48 rows as hygienic; it then completes its existing 2,252-row structural closure without
  weakening or allowlisting composition hygiene.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- CPU 0009D remains after CPU 0009C. CPU 0009G alone owns broad representative performance and
  final generated-coverage checkpoint claims.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Architecture impact

Expected impact: None. Stop for an architecture, [dependency](../../../../glossary.md#term-dependency), provider/[admission](../../../../glossary.md#term-admission)/selection,
semantics, or [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings) decision that cannot be justified within existing contracts.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU task 0009C1. Read AGENTS.md,
ARCHITECTURE.md, the Planning Guide, CPU master plan, CPU 0009, CPU 0009C, and this task.
Implement exactly this CONCAT/STACK generated-entry hygiene correction; do not weaken tests or
allowlist the terminal constructor. Stop for an architecture or scope conflict. After executable
validation, hand the stable diff and exact evidence to a separate clean documentation-focused
context under documentation-rules.md. Do not commit or push.
```

## Local decisions

- The 48 rows are a single cohesive production prerequisite because one emitter-owned invariant
  affects both CONCAT and STACK selected entries and blocks the same CPU 0009C hygiene gate.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- The prerequisite is numbered `0009C1` as an explicit corrective child discovered at CPU 0009C's
  active frontier. It depends on completed CPU 0009B, and CPU 0009C depends on it; this is an
  ordered prerequisite, not a circular dependency.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- The final occurrence is a terminal assignment only because Prepare has already proved the
  ordered prefix/occurrence partition exhaustive for admitted inputs. It is the sole allowed
  unconditional statically prepared occurrence arm, never a dummy/default value, <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [Prepare glossary](../../../../glossary.md#prepare); [partition glossary](../../../../glossary.md#partition))</span>`switch`
  default, exception path, or catch-all for invalid runtime state; it does not change the existing
  cold validation boundary.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [validation glossary](../../../../glossary.md#term-validation))</span>
- Normalized structural equality was not proved for the changed generated and clean-Java entries.
  The mandatory fallback performance protocol therefore ran; this is not a general
  structural-equivalence finding.<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- The fallback found that general-long [`MemorySegment`](../../../../glossary.md#term-memorysegment) CONCAT/STACK access built native-order
  layouts in the hot loop. Generation-fixed typed/frozen layouts remove that overhead only for
  composition loads and shared stores, preserving the optimal clean-Java equivalence objective.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [objective glossary](../../../../glossary.md#term-objective))</span>

## Known limitations

This task proves neither arbitrary-[bytecode](../../../../glossary.md#term-bytecode) equivalence nor broad generated/direct performance
parity. Its 48 source rows map to eight distinct form/path/[carrier](../../../../glossary.md#term-storage-carrier) representatives; source [shapes](../../../../glossary.md#shape)
were scaled only to expose loop cost while preserving selected path/carrier mapping. It changes no
public capability, does not close [PAD](../../../../glossary.md#padding)/TILE, and does not close the remaining [CPU](../../../../glossary.md#term-cpu) 0009C forms.

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context ran the nine requested [CPU](../../../../glossary.md#term-cpu) suites sequentially with `--rerun-tasks`:
`CpuOrdinaryMovementFoldSemanticClosureTest`,
`CpuAffineMovementIndexingScatterRandomStructuralOracleTest`, `CpuClassFileKernelGeneratorTest`,
`CpuGeneratedCoverageCheckpointTest`, `CpuGeneratedCoverageEvidenceTest`,
`CpuKernelSpecializationTest`, `CpuGeneratedKernelArtifactStoreTest`,
`CpuGeneratedCoverageUnsupportedBoundaryTest`, and `CpuConcatStackPerformanceTest`; all passed.
The final regenerated [canonical](../../../../glossary.md#term-canonical-form) inventory was byte-equal. [Schema](../../../../glossary.md#term-schema) 65 is current; the canonical
inventory [SHA-256](../../../../glossary.md#term-sha-256) is `e821d08cad4e5903510d4b6f700f94708c01b913f9e3a5e8b4e5d6f3bd718c1d`, with
17,636 rows (17,463 generated and 173 rejected). The transition [checkpoint](../../../../glossary.md#term-checkpoint) reran 9 tests with 0
failures and reconstructed `2e9f07e30b47429387c8c1043207ee6e8feae327f09ac24a7a8f8c51821918fb`
from 24 complete old rows. Those rows are exactly 12 CONCAT and 12 STACK mixed/segment rows
across six types; only evidence key and class SHA-256 differ. This narrow post-terminal/
pre-frozen transition must not be projected onto the earlier full task delta. The [sealed](../../../../glossary.md#term-sealed)
performance directory contains 63 files and its `SHA256SUMS` [manifest](../../../../glossary.md#term-manifest) validates.

The accepted fallback protocol is five isolated [forks](../../../../glossary.md#term-fork-meanings), five warmups, seven measured [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), at
least 25 ms per side, balanced order, `-Xms1g -Xmx1g`, `-XX:-TieredCompilation`, `-Xbatch`, and
`<= 1.15` for every sample, [fork](../../../../glossary.md#term-fork-meanings), case, and aggregate. Aggregate generated/direct ratios:

```text
CONCAT_DENSE_HEAP      0.8555022697497232
CONCAT_BOUNDED_HEAP    0.999867393590381
CONCAT_BOUNDED_SEGMENT 1.000256310992101
CONCAT_BOUNDED_MIXED   1.0008255283079883
STACK_DENSE_HEAP       0.8180704345166037
STACK_GENERAL_HEAP     1.000866428804377
STACK_GENERAL_SEGMENT  1.000566687342971
STACK_GENERAL_MIXED    0.9992810357364201
aggregate median       0.999867393590381
```

This documentation context did not rerun successful Java semantic or performance suites. It runs
CPU [Javadoc](../../../../glossary.md#term-javadoc), Markdown links/anchors/fences/whitespace and changed-path checks, and `git diff --check`
after final documentation edits.

## Implementation notes

The emitter now uses an unconditional final statically prepared CONCAT/STACK [occurrence](../../../../glossary.md#term-occurrence) arm.
[Schema](../../../../glossary.md#term-schema) 65 invalidates envelopes while the proved [specialization](../../../../glossary.md#term-specialization) [identity](../../../../glossary.md#term-identity-meanings) projection remains
unchanged. Frozen typed [layouts](../../../../glossary.md#layout) are generation-fixed only for composition loads/shared stores.

## Completion summary

- Completed changes: finalized schema-65 composition-hygiene, fallback-performance, reversible
  transition, and exact-inventory documentation; synchronized CPU 0009C1/0009C/parent/master-plan/
  roadmap status.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Files changed or created: task-owned production, build, Java-test, resource, and evidence paths
  listed above, plus this task, parent CPU 0009, CPU master plan, and roadmap.
- Tests and [validation](../../../../glossary.md#term-validation): reused nine sequential `--rerun-tasks` suite passes, byte-equal regenerated
  inventory, valid sealed hashes, and transition checkpoint 9 tests/0 failures. This context ran
  CPU Javadoc, targeted Markdown link/anchor/fence/whitespace and changed-path checks, and<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
  `git diff --check`.
- Documentation-agent review: clean context `01a081b7-d9ac-7f32-884b-ea611ad1e6cd`; Planning
  plus API/Javadoc profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: public/[API](../../../../glossary.md#term-application-programming-interface-api) Javadocs, guides, glossary, `ARCHITECTURE.md`, current
  architecture plan, and ADRs require no change: this is CPU-private hot-path/evidence work with
  no public contract, architecture, terminology, or decision change. Architecture, backend
  conformance, integration tests, other modules, and public behavior likewise require no change.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: `CpuDataMovementEmitter` and `CpuGeneratorSchema` accurately document the
  selected terminal arm/schema-65 boundary; the performance harness, clean oracle, structural,
  checkpoint, and cache test Javadocs were reviewed and remained accurate after their updates.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [cache glossary](../../../../glossary.md#term-cache))</span>
- Unresolved issues: None for 0009C1. CPU 0009 remains incomplete; PAD/TILE and other 0009C work
  remain outside this task.<span data-glossary-navigation="true"> ([PAD glossary](../../../../glossary.md#padding))</span>
- Follow-up required: [CPU](../../../../glossary.md#term-cpu) 0009C is Complete; CPU 0009D is the next Draft task.

Status: Complete
