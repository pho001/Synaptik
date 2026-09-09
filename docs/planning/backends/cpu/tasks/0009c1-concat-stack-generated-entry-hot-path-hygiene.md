# Task 0009C1: CONCAT and STACK Generated-Entry Hot-Path Hygiene

## Status

Complete

## Goal

Correct the narrow production prerequisite discovered while reviewing CPU 0009C: every selected
generated `CONCAT` and `STACK` entry must preserve the existing represented-bit composition
semantics and clean-Java-equivalent loop/dataflow while containing no avoidable terminal
`IllegalStateException` allocation, constructor invocation, or `ATHROW` path. This correction
unblocks CPU 0009C's 48-row CONCAT/STACK hygiene closure; it does not broaden movement support.

## Scope

- Inspect the selected generated entries for all 24 `CONCAT` and 24 `STACK` rows. Preserve and
  assert two separate source-derived cross-tabs, without aggregate masking: the checked inventory
  access regimes are 12 `DENSE_LINEAR` plus 12 `GENERAL_ODOMETER` rows for each form, while the
  actual `CpuKernelSpecialization.loopAddressing` emitter paths are 6
  `DENSE_HEAP_ARRAY_INT` plus 18 `GENERAL_LONG` rows for each form. The six dense-but-segment-
  contiguous rows select `GENERAL_LONG`; they must not be misreported as dense heap-array paths.
  Replace only the impossible-after-preparation terminal path with
  a verifier-valid exhaustive selection chain. For each statically prepared occurrence map of
  length `n`, emit comparisons only for occurrences `0 .. n - 2`; use occurrence `n - 1` as the
  unconditional terminal assignment/path. For CONCAT, each earlier arm tests its prepared prefix
  boundary and the final arm receives the remaining coordinate; for STACK, each earlier arm tests
  its prepared occurrence coordinate and the final arm is unconditional. Apply this topology to
  `emitDenseConcat`, `emitDenseStack`, `emitConcat`, and `emitStack`, and to the independently
  `javac`-compiled clean oracle. Prepare proves that the prefixes/occurrence coordinates partition
  every admitted invocation exhaustively, so this is neither runtime validation nor silent default
  semantics. The sole permitted unconditional arm is the statically prepared final occurrence;
  it must assign the typed source/value before the common store. Do not use a dummy/default value,
  a `switch` default branch, an exception path, or any catch-all semantics as that final
  occurrence, and do not read an uninitialized local.
- Preserve occurrence order (including repeated inputs), ordered typed carriers, axis/coordinate
  mapping, represented-bit loads/stores, range and sentinel behavior, dense-int and general-long
  cursor/loop shape, and all existing cold guards/fallback selection.
- Add or tighten automated semantic, Class-File/decompilation, and mutation evidence so both the
  generated selected entry and independently compiled clean-oracle entry reject `NEW`,
  `ANEWARRAY`, `NEWARRAY`, `MULTIANEWARRAY`, `INVOKEDYNAMIC`, `ATHROW`, and
  `IllegalStateException.<init>`, rather than merely filtering invocations. The same automated
  member scan rejects generated/reference/oracle/helper leakage, reflection, map/string or generic
  carrier dispatch, boxing, and unapproved virtual/interface dispatch; only the exact typed array
  opcodes or explicitly justified `MemorySegment` calls may remain. Verify the clean Java has the
  same all-but-final-test/unconditional-final topology.
- Inspect `CpuGeneratorSchema`, `CpuKernelSpecialization` identity/compatibility bytes, generated
  binary naming, persistent-cache behavior, and exact fixture digest resources before changing
  bytes. The generated-byte change advances `CURRENT_VERSION` from 64 to 65. Retain the existing
  class-identity projection, structural key, and generated binary name only if automated evidence
  proves that the existing family projection remains semantically correct; otherwise update their
  owned contract. Update literal schema assertions, implementation Javadocs, cache expectations,
  and affected resources/digest-ledger evidence. For real CONCAT and STACK fixtures, prove that a
  persisted same-key schema-64 envelope is rejected and regenerated, then that schema-65 persists
  and is subsequently a persisted hit.

## Out of scope

- Any new operation, provider capability, preparer admission/selection rule, IR/lowering design,
  materialization policy, public API, architecture contract, native route, or broad CPU 0009C
  structural-evidence redesign.
- Altering `CONCAT`/`STACK` semantics, validation boundaries, legal input geometry, selected
  strategy, fallback coverage, range ownership, or existing tests merely to make the hygiene check
  pass. Do not weaken a test, omit a selected realization, or allowlist the constructor/allocation
  terminal path.
- CPU 0009C's affine, other movement, indexing, scatter, random, or full 2,252-row closure;
  CPU 0009D--0009G; literal byte-for-byte, `javac`, CFG, or JIT identity; and a new broad
  performance campaign.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): backend prepare owns lowering and kernel
  selection; generated code must preserve the optimal clean-Java algorithm, hot-loop/dataflow
  shape, and avoidable-overhead profile.
- [Current architecture navigation](../../../../architecture/current-architecture-plan.md),
  [Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md),
  [General profile](../../../../developer-guide/documentation/general-style.md),
  [Planning profile](../../../../developer-guide/documentation/planning-style.md), [CPU master
  plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md),
  completed [CPU 0009B](0009b-ordinary-pointwise-and-cast-structural-oracle-closure.md), and
  blocked [CPU 0009C](0009c-affine-movement-indexing-scatter-random-structural-oracle-closure.md).

## Architecture constraints

- Prepare remains the owner of supported, validated lowering and selected realization. Generated
  entries execute prepared inputs only; they must not acquire Model interpretation, generic
  validation, reflection, maps, boxing, allocation, helper bridges, or avoidable virtual/interface
  dispatch.
- A well-written specialized clean-Java oracle is the performance and structural review oracle.
  Preserve its composition selection and primitive loop/dataflow shape; a dead exceptional path is
  not permitted as a substitute for a prepared-input invariant.
- Schema/cache identity is a correctness boundary. This generated-byte change advances the
  current-only compatibility schema from 64 to 65. Inspect class identity, structural key,
  generated binary naming, cache load/miss behavior, and SHA-bound evidence; retain the existing
  family projection only with automated proof that it remains semantically correct. Prove that
  stale same-key schema-64 CONCAT/STACK envelopes cannot be reused and that schema-65 envelopes
  regenerate, persist, and subsequently hit.

## Package impact

Existing production packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` owns direct movement bytecode
  emission and generated-entry evidence.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` owns generator compatibility and
  generated-artifact identity.

Packages added or changed:

- None. No public package, module dependency, or architecture boundary changes.

Type placement:

- `...emit.CpuDataMovementEmitter` — owns only the selected CONCAT/STACK bytecode correction.
- `...cache.CpuGeneratorSchema` — advances `CURRENT_VERSION` from 64 to 65 and updates its
  implementation Javadoc for the byte-shaping CONCAT/STACK topology correction.
- `...cache.CpuKernelSpecialization` — retains or changes its existing class-identity projection,
  structural key, and generated binary name only as automated evidence proves correct.
- `...emit.CpuAffineMovementIndexingScatterRandomStructuralOracleTest` and existing movement
  semantic/evidence fixtures — own exact selected-entry semantic, hygiene, mutation, and
  digest synchronization evidence; add no generic production bridge.
- `...emit.CpuAffineMovementIndexingScatterRandomCleanJavaOracle` — owns only the independently
  compiled counterpart source and must use the same verifier-valid all-but-final-test and
  statically prepared unconditional-final-occurrence topology, never a `switch` default,
  initialized fallback value, exception path, or catch-all semantics.

## Affected files

Actual production, build, test, resource, and evidence fan-out:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`, only if inspected compatibility invalidation requires it
- `backends/cpu/build.gradle.kts` — forwards the explicit opt-in performance properties to the
  isolated harness; it does not change ordinary test or production behavior.
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineMovementIndexingScatterRandomStructuralOracleTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineMovementIndexingScatterRandomCleanJavaOracle.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryMovementFoldSemanticClosureTest.java`, only if its direct selected-entry composition fixture needs exact digest synchronization
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`, `CpuGeneratedCoverageCheckpointTest.java`, and `CpuGeneratedCoverageEvidenceTest.java`, only for the concrete generated-name/definition, registry, or digest evidence their current assertions own
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java` and `CpuGeneratedKernelArtifactStoreTest.java`, only for the exact composition identity/schema and persisted hit/miss/stale-envelope assertions their current packages own
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConcatStackPerformanceTest.java`
  — the required fallback-performance harness and source-derived representative mapping.
- All existing CPU test files with a literal `assertEquals(64,
  CpuGeneratorSchema.CURRENT_VERSION)` (or equivalent fully qualified assertion) may receive the
  mechanical current-schema update to `65` when that is their only task-owned change. The same
  allowance covers only stale-schema wording or Javadoc associated with that literal expectation;
  it does not authorize behavioral, fixture, family, or unrelated cleanup in those files.
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/cpu-0009c1-post-terminal-frozen-layout-transition.tsv`
  — reversible 24-row transition evidence.
- The canonical inventory, disposition registry, evidence ledger, and their exact inventory-SHA
  consumers — regenerated only for schema-65 identity facts.
- `backends/cpu/evidence/cpu-0009c1-concat-stack-performance/` — the sealed, 63-file retained
  fallback-performance bundle and its `SHA256SUMS` manifest.

Planning paths: this task, CPU 0009C, parent CPU 0009, CPU master plan, and roadmap.

## Maximum scope

This task may modify at most two production Java paths, one build path, twenty-two CPU Java-test
paths, five checked CPU resource paths, the sealed 63-file performance-evidence tree, and five
planning paths. The expanded fan-out is a discovered required consequence of the mandatory
fallback performance protocol, the frozen-layout correction, schema invalidation, and checked-in
inventory integrity; it authorizes no unrelated refactor. The Java-test bound includes the twelve
literal-schema assertions, composition/cache/structural owners, and the performance harness. The
resource bound includes the canonical inventory and exact SHA consumers, ledger/registry joins,
and reversible transition resource. The prior eighteen-path estimate did not include those
required evidence owners. Across the literal-assertion files, the only permitted task-owned
change is the mechanical `CURRENT_VERSION`/schema expectation update from `64` to `65` and
associated stale-schema wording or Javadoc. This task may not modify Model/Compiler/Prepare/
Runtime/Engine APIs, architecture/ADR, conformance/integration tests, glossary, guides, or
unrelated modules. If the correction needs a new production abstraction, changed admission/
selection semantics, or another family, stop and propose a separate Draft task.

## Acceptance criteria

- All 48 exact checked CONCAT/STACK rows deterministically generate, define/load, and invoke their
  actual typed selected entry across applicable array/segment/mixed carrier roles, inventory access,
  actual emitter path, full/empty/interior/tail ranges, and occurrence maps. They retain bit-exact
  results and untouched sentinels against independently authored specialized clean Java. Automated
  accounting preserves two exact form × source-derived cross-tabs, with owner IDs in every cell:
  inventory access reports 12 `DENSE_LINEAR` plus 12 `GENERAL_ODOMETER` rows for CONCAT and the
  same 12/12 split for STACK; actual `loopAddressing` emitter paths report 6
  `DENSE_HEAP_ARRAY_INT` plus 18 `GENERAL_LONG` rows for CONCAT and the same 6/18 split for STACK.
  The six segment-contiguous dense rows per form belong to inventory `DENSE_LINEAR` but emitter
  `GENERAL_LONG`. An aggregate count may not mask a missing applicable cell; a later absent cell
  requires its source-derived inapplicability reason.
- Class-File inspection and decompilation of every generated selected entry and every compiled
  clean-oracle entry prove the all-but-final-test/unconditional-final selection topology and no
  `NEW`, `ANEWARRAY`, `NEWARRAY`, `MULTIANEWARRAY`, `INVOKEDYNAMIC`, `ATHROW`,
  `IllegalStateException.<init>`, generic/helper bridge, reflection, map/string dispatch, boxing,
  or avoidable virtual/interface dispatch. Any narrowly permitted primitive/FFM carrier call is
  explicitly tied to its typed selected carrier. The test fails if a forbidden terminal path,
  dummy/initialized fallback values, `switch`-default or exception paths, or unapproved member
  survives; the required unconditional final statically prepared occurrence arm remains
  permitted. Opcode evidence, not invoke-only filtering, is required.
- The comparison proves composition topology separately for CONCAT boundary selection and STACK
  occurrence selection, including repeated inputs and axis insertion/removal. It preserves loop
  direction, range guard, coordinate/address progression, typed load/store, and dense-int versus
  general-long dataflow rather than comparing names, hashes, catalog rows, or shared helpers.
- Mutation controls fail for reintroducing `NEW`, constructor invocation, or `ATHROW`; changed
  occurrence order/boundary, axis coordinate map, carrier ABI/order, range/tail direction,
  represented-bit conversion/store, and an allowed invocation target. No mutation can pass through
  self-comparison, a stale digest, or test weakening.
- Generated bytes advance `CpuGeneratorSchema.CURRENT_VERSION` from 64 to 65. Automated evidence
  proves whether the existing class-identity projection, structural key, and generated binary name
  remain semantically correct; it refreshes literal schema assertions, implementation Javadocs,
  cache expectations, fixture SHA-256s, and ledgers as applicable. All twelve existing CPU tests
  with literal current-schema assertions update their `64` expectation to `65` when that mechanical
  expectation (and any associated stale-schema wording/Javadoc) is their only task-owned change.
  Behavioral edits remain limited to the named composition/cache/structural tests. Real CONCAT and
  STACK fixtures prove same-key schema-64 persisted-envelope rejection and regeneration, followed
  by schema-65 persistence and a persisted hit.
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
  five isolated Java 26 forks with `-Xms1g -Xmx1g`, `-XX:-TieredCompilation`, and `-Xbatch`, at
  least five randomized warmups, seven randomized measured rounds, adaptive batches of at least
  25 ms on each side, balanced generated/direct order, per-fork medians, and the median
  of fork medians. Every per-case fork ratio and aggregate generated/direct ratio must be
  `<= 1.15x`; a missing, interrupted, noisy, or over-threshold result is fail-closed and leaves
  the task incomplete. If the normalized projection proves only removal of the unreachable terminal,
  do not measure: add automated Class-File evidence showing that exact fact for every form × actual
  emitter path × carrier shape and record why no new measurement is warranted. This task does not claim or
  launch CPU 0009G's broad campaign.
- A separate documentation-focused clean context finalizes changed implementation Javadocs and
  planning documentation, and records reasoned no-change conclusions for public/API Javadocs,
  guides, glossary, architecture/ADRs, architecture/conformance/integration tests, Gradle, and
  unrelated modules.

## Tests / validation

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

The extended structural-oracle test must run the automated generated-and-clean opcode/member scan,
selection-topology check, mutations, and both exact source-derived cross-tabs: inventory access
(`DENSE_LINEAR`/`GENERAL_ODOMETER`) and actual `loopAddressing` emitter path
(`DENSE_HEAP_ARRAY_INT`/`GENERAL_LONG`). The extended
`CpuClassFileKernelGeneratorTest`/`CpuKernelSpecializationTest` must prove the applicable class
identity projection, deterministic generated binary name, descriptor, and definition behavior.
The extended `CpuGeneratedKernelArtifactStoreTest` must use real CONCAT and STACK fixtures to prove
schema-64 same-key stale-envelope rejection/re-generation, then schema-65 persisted generated and
hit behavior. `CpuGeneratedCoverageCheckpointTest`
and `CpuGeneratedCoverageEvidenceTest` must prove checked inventory/disposition/evidence-ledger
SHA/digest joins remain synchronized. Use `javap` only as recorded supplementary inspection, not as
a replacement for automated evidence. The normalized-projection gate decides objectively whether
the bounded five-fork command above is required.

Documentation context reuses successful Java evidence unless it changes executable Java. It runs
the affected module Javadoc task after final Javadoc edits, checks Markdown links/anchors/fences,
status/dependency synchronization, terminology, whitespace, path scope, and:

```bash
git diff --check
```

Repository-wide/capability-checkpoint validation and broad representative performance closure defer
to CPU 0009G/CI.

## Dependencies

- Complete CPU 0009B and its documentation pass.
- Current checked CPU 0009C inventory, semantic fixture, structural-oracle test, evidence registry,
  emitter, cache identity, and generated schema contracts.
- Java 26 Class-File, Foreign Function and Memory API, and CPU generated-code test environment.

## Follow-up tasks

- Required: CPU 0009C resumes only after this task is Complete and its documentation pass records
  the 48 rows as hygienic; it then completes its existing 2,252-row structural closure without
  weakening or allowlisting composition hygiene.
- CPU 0009D remains after CPU 0009C. CPU 0009G alone owns broad representative performance and
  final generated-coverage checkpoint claims.

## Architecture impact

Expected impact: None. Stop for an architecture, dependency, provider/admission/selection,
semantics, or schema/identity decision that cannot be justified within existing contracts.

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
  affects both CONCAT and STACK selected entries and blocks the same CPU 0009C hygiene gate.
- The prerequisite is numbered `0009C1` as an explicit corrective child discovered at CPU 0009C's
  active frontier. It depends on completed CPU 0009B, and CPU 0009C depends on it; this is an
  ordered prerequisite, not a circular dependency.
- The final occurrence is a terminal assignment only because Prepare has already proved the
  ordered prefix/occurrence partition exhaustive for admitted inputs. It is the sole allowed
  unconditional statically prepared occurrence arm, never a dummy/default value, `switch`
  default, exception path, or catch-all for invalid runtime state; it does not change the existing
  cold validation boundary.
- Normalized structural equality was not proved for the changed generated and clean-Java entries.
  The mandatory fallback performance protocol therefore ran; this is not a general
  structural-equivalence finding.
- The fallback found that general-long `MemorySegment` CONCAT/STACK access built native-order
  layouts in the hot loop. Generation-fixed typed/frozen layouts remove that overhead only for
  composition loads and shared stores, preserving the optimal clean-Java equivalence objective.

## Known limitations

This task proves neither arbitrary-bytecode equivalence nor broad generated/direct performance
parity. Its 48 source rows map to eight distinct form/path/carrier representatives; source shapes
were scaled only to expose loop cost while preserving selected path/carrier mapping. It changes no
public capability, does not close PAD/TILE, and does not close the remaining CPU 0009C forms.

## Validation evidence

The implementation context ran the nine requested CPU suites sequentially with `--rerun-tasks`:
`CpuOrdinaryMovementFoldSemanticClosureTest`,
`CpuAffineMovementIndexingScatterRandomStructuralOracleTest`, `CpuClassFileKernelGeneratorTest`,
`CpuGeneratedCoverageCheckpointTest`, `CpuGeneratedCoverageEvidenceTest`,
`CpuKernelSpecializationTest`, `CpuGeneratedKernelArtifactStoreTest`,
`CpuGeneratedCoverageUnsupportedBoundaryTest`, and `CpuConcatStackPerformanceTest`; all passed.
The final regenerated canonical inventory was byte-equal. Schema 65 is current; the canonical
inventory SHA-256 is `e821d08cad4e5903510d4b6f700f94708c01b913f9e3a5e8b4e5d6f3bd718c1d`, with
17,636 rows (17,463 generated and 173 rejected). The transition checkpoint reran 9 tests with 0
failures and reconstructed `2e9f07e30b47429387c8c1043207ee6e8feae327f09ac24a7a8f8c51821918fb`
from 24 complete old rows. Those rows are exactly 12 CONCAT and 12 STACK mixed/segment rows
across six types; only evidence key and class SHA-256 differ. This narrow post-terminal/
pre-frozen transition must not be projected onto the earlier full task delta. The sealed
performance directory contains 63 files and its `SHA256SUMS` manifest validates.

The accepted fallback protocol is five isolated forks, five warmups, seven measured samples, at
least 25 ms per side, balanced order, `-Xms1g -Xmx1g`, `-XX:-TieredCompilation`, `-Xbatch`, and
`<= 1.15` for every sample, fork, case, and aggregate. Aggregate generated/direct ratios:

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
CPU Javadoc, Markdown links/anchors/fences/whitespace and changed-path checks, and `git diff --check`
after final documentation edits.

## Implementation notes

The emitter now uses an unconditional final statically prepared CONCAT/STACK occurrence arm.
Schema 65 invalidates envelopes while the proved specialization identity projection remains
unchanged. Frozen typed layouts are generation-fixed only for composition loads/shared stores.

## Completion summary

- Completed changes: finalized schema-65 composition-hygiene, fallback-performance, reversible
  transition, and exact-inventory documentation; synchronized CPU 0009C1/0009C/parent/master-plan/
  roadmap status.
- Files changed or created: task-owned production, build, Java-test, resource, and evidence paths
  listed above, plus this task, parent CPU 0009, CPU master plan, and roadmap.
- Tests and validation: reused nine sequential `--rerun-tasks` suite passes, byte-equal regenerated
  inventory, valid sealed hashes, and transition checkpoint 9 tests/0 failures. This context ran
  CPU Javadoc, targeted Markdown link/anchor/fence/whitespace and changed-path checks, and
  `git diff --check`.
- Documentation-agent review: clean context `01a081b7-d9ac-7f32-884b-ea611ad1e6cd`; Planning
  plus API/Javadoc profiles.
- Documentation impact: public/API Javadocs, guides, glossary, `ARCHITECTURE.md`, current
  architecture plan, and ADRs require no change: this is CPU-private hot-path/evidence work with
  no public contract, architecture, terminology, or decision change. Architecture, backend
  conformance, integration tests, other modules, and public behavior likewise require no change.
- Javadoc review: `CpuDataMovementEmitter` and `CpuGeneratorSchema` accurately document the
  selected terminal arm/schema-65 boundary; the performance harness, clean oracle, structural,
  checkpoint, and cache test Javadocs were reviewed and remained accurate after their updates.
- Unresolved issues: None for 0009C1. CPU 0009 remains incomplete; PAD/TILE and other 0009C work
  remain outside this task.
- Follow-up required: CPU 0009C is Complete; CPU 0009D is the next Draft task.

Status: Complete
