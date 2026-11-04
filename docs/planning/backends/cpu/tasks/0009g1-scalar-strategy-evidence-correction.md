# Task 0009G1: [Scalar](../../../../glossary.md#term-scalar)-Strategy Evidence Correction

## Reader context

This task records an evidence-only repair adding selected-`SCALAR` witnesses for seven
scalar-immediate forms while retaining `SCALAR_ADD`’s four-strategy orchestration basis.
`CpuScalarImmediateClampMatrixOracle` constructs finite fixtures; `CpuPartitionPreparer` supplies
observed selection; `CpuGeneratedCoverageCheckpointTest` requires a generated scalar owner wherever
live fixtures establish that scalar is meaningful. Scalar-immediate describes an operation’s
argument form, whereas `SCALAR` names its selected execution strategy.

Here, inventory hashes bind six regenerated resources and their direct Java consumers into one
fail-closed accounting chain. Semantic closure concerns generated-entry results; structural-oracle
evidence concerns implementation shape. Adding selection witnesses does not upgrade either
disposition. `CpuScalarEmitter`, schemas, preparation policy, and Runtime behavior remain unchanged
in this record; Runtime executes prepared work rather than selecting strategies. The correction does
not authorize new production defaults.

Current-topic navigation (not a replacement for the historical record):

- [current exact typed scalar metadata](../../../../reference/tensor/data-types.md#exact-typed-scalar-values) — Distinguishes immutable immediate bits from a rank-zero Tensor or live storage.
- [current ordinary-production route policy](../../../../design/notes/cpu-kernel-strategy.md#current-ordinary-production-decision) — Prevents historical vector or parallel evidence from being read as current production-default authorization.

## Status

Complete

## Goal

Correct the sealed [CPU](../../../../glossary.md#term-cpu) generated-coverage evidence immediately after completed CPU 0009G. Add one
[operation](../../../../glossary.md#operation)-specific, selected-[`SCALAR`](../../../../glossary.md#term-scalar) witness for each currently admitted scalar-immediate form
that lacks one: `SCALAR_SUB`, `SCALAR_MUL`, `SCALAR_DIV`, `SCALAR_MIN`, `SCALAR_MAX`,
`SCALAR_POW`, and `SCALAR_CLAMP`. Preserve the existing `SCALAR_ADD` scalar/[vector](../../../../glossary.md#term-vector)/
parallel-scalar/parallel-vector orchestration witnesses. Make the [canonical](../../../../glossary.md#term-canonical-form) inventory [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation)
when an admitted live operation form with a meaningful scalar execution [model](../../../../glossary.md#term-model) lacks selected
scalar evidence.

This is an evidence correction, not a capability or strategy-policy change. The corrected
resources must describe the current `CpuScalarEmitter` and live preparer selection truthfully.

## [Scope](../../../../glossary.md#term-scope)

- Derive one finite, direct scalar-selection witness for each of the seven listed operation forms
  from the existing scalar-immediate matrix fixture/oracle machinery. Each witness must be
  provider-supported, lowerable, preparer-admitted, generated, and selected as <span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [operation glossary](../../../../glossary.md#operation); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [fixture glossary](../../../../glossary.md#term-fixture); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>[`SCALAR`](../../../../glossary.md#term-scalar) by the
  live `CpuPartitionPreparer`.
- Retain the existing ADD orchestration basis unchanged: it remains the only required witness set
  for scalar/vector and caller-parallel orchestration sharing. Do not make the seven missing
  forms prove vector or parallel selection.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>
- Extend the matrix and coverage-checkpoint assertions so the set of live admitted operation
  forms having a meaningful scalar execution model is derived from live fixtures and is a subset
  of forms with a generated selected-scalar inventory owner. A form whose scalar request is
  rejected or has no scalar execution model must remain explicitly outside that requirement.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner))</span>
- Regenerate and seal the complete finite resource closure whose bytes or current-total claims
  change: the scalar-immediate form ledger, generated coverage inventory,
  `generated-pointwise-semantic-closure.tsv`, `specialized-semantic-closure-gaps.tsv`,
  `generated-coverage-evidence-ledger.tsv`, and `generated-coverage-gap-matrix.json`. Reconcile
  their exact generated/rejected accounting, hashes, schemas, ownership, selected-strategy
  columns, and current-total projections with live reconstruction.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([schemas glossary](../../../../glossary.md#term-schema); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Synchronize every direct Java consumer of the inventory's SHA-256, every direct
  scalar-immediate count assertion, and the direct semantic/structural closure tests that seal
  those facts. Retain every SHA-256 assertion and negative stale-digest check; update its expected
  canonical value only after byte-exact regeneration.<span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [digest glossary](../../../../glossary.md#term-digest); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Keep evidence limited to scalar-selection presence. Preserve all existing semantic invocation,
  clean-Java, structural, route, materialization, rejection, and performance dispositions unless
  their checked row identity necessarily changes because of these seven added witnesses.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Production Java; CPU capability, lowering, emitter, generator schema, cache identity,
  preparation eligibility, selection policy, route, materialization policy, or Runtime behavior.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [materialization glossary](../../../../glossary.md#materialization); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- New semantic forms, types, carriers, layouts, arbitrary immediate/bound enumeration, vector or
  parallel witnesses for the seven forms, and any universal all-strategy requirement.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [vector glossary](../../../../glossary.md#term-vector))</span>
- Benchmarks, five-fork campaigns, performance claims, performance-disposition changes, or fresh
  profiling.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- Public API/Javadoc, CPU guide, glossary, architecture, ADR, Gradle, architecture-test,
  backend-conformance, or integration-test changes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references and constraints

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) remains authoritative: CPU backend analysis
  owns lowering and strategy selection; planning selects ownership rather than a route; Runtime
  executes prepared work and does not select a strategy.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [CPU 0008Q1](0008q1-finite-scalar-immediate-clamp-matrix.md) remains the finite source-derived
  scalar-immediate/clamp fixture authority. <span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [scalar glossary](../../../../glossary.md#term-scalar); [fixture glossary](../../../../glossary.md#term-fixture))</span>[CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md)
  and completed [CPU 0009G](0009g-final-support-correctness-hygiene-and-inventory-checkpoint.md)
  remain the inventory/checkpoint context.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- The correction is test/evidence-only. It must observe current selection; it must not cause or
  justify a production selection change. Generated hot-path semantics and all existing exact
  rejection boundaries remain unchanged.
- The checkpoint must distinguish “a selected scalar witness is meaningful and missing” from
  “scalar is rejected or not an execution model for this live form.” It must not infer universal
  scalar support from an unrelated form or from ADD orchestration evidence.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

## Package impact

Existing test-only package:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` retains scalar-immediate fixture
  construction, live preparation selection, resource sealing, and coverage checkpoint accounting.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [fixture glossary](../../../../glossary.md#term-fixture); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

No package, production type, public [API](../../../../glossary.md#term-application-programming-interface-api), or [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency) changes.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

Expected implementation paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampMatrixOracle.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampMatrixTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedCoverageCheckpointTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedDirectEvidenceClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampCleanJavaStructuralEquivalenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseSemanticClosureManifestTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedStructuralOracleCatalogTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineGeneratedCoverageFixtureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineMovementIndexingScatterRandomStructuralOracleTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuMaskedAdvancedReductionSemanticClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryMovementFoldSemanticClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPool1dCompositionSemanticClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPool2d3dSemanticClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuRandomOneHotSemanticClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedCoverageEvidenceTest.java`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/scalar-immediate-clamp-matrix-forms.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-inventory.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-pointwise-semantic-closure.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/specialized-semantic-closure-gaps.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-evidence-ledger.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-gap-matrix.json`
- This task.
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Maximum implementation scope: the 21 [CPU](../../../../glossary.md#term-cpu) test/evidence paths above and this [planning](../../../../glossary.md#term-planning) task
(22 paths). Final documentation and status [synchronization](../../../../glossary.md#term-synchronization) additionally permits the CPU master
plan and repository roadmap, for an exact final scope of 24 paths. This exceptional cohesive
mechanical closure is safer than weakening a SHA-bound, [fail-closed](../../../../glossary.md#term-fail-closed) evidence chain: every CPU
path is a direct byte-[digest](../../../../glossary.md#term-digest), count/total, or direct semantic/structural consumer of the seven new
inventory [owners](../../../../glossary.md#term-owner), so leaving any one stale would either fail legitimately or conceal an
inconsistent seal. Zero production, public documentation, build, architecture, [conformance](../../../../glossary.md#term-conformance),
integration, benchmark, generator-[schema](../../../../glossary.md#term-schema), or other paths.
If a [fixture](../../../../glossary.md#term-fixture) requires a generator/schema/capability/selection change, an additional [canonical](../../../../glossary.md#term-canonical-form)
resource outside this [finite](../../../../glossary.md#term-finite) list, or a [scalar](../../../../glossary.md#term-scalar) witness cannot be selected by current live
preparation, stop and create a narrow Draft follow-up; do not alter the production policy,
weaken or remove digest [binding](../../../../glossary.md#term-binding), or broaden this correction.

## Acceptance criteria

1. The live [scalar](../../../../glossary.md#term-scalar)-immediate [matrix](../../../../glossary.md#term-matrix-meanings) has exactly one [operation](../../../../glossary.md#operation)-specific selected-`SCALAR` witness
   for each of `SCALAR_SUB`, `SCALAR_MUL`, `SCALAR_DIV`, `SCALAR_MIN`, `SCALAR_MAX`,
   `SCALAR_POW`, and `SCALAR_CLAMP`. Each is direct, generated, provider-supported, lowerable,
   preparer-admitted, and selected by live preparation rather than a hard-coded resource claim.
2. The existing `SCALAR_ADD` scalar/vector/parallel-scalar/parallel-vector orchestration basis is
   preserved without using it as proxy evidence for any other operation form.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>
3. The coverage checkpoint derives the live admitted forms for which scalar is meaningful and
   requires an exact selected-scalar generated owner for every such form. It fails for a missing,
   duplicate, stale, relabelled, unsupported, or non-generated witness, while explicit
   scalar-not-meaningful/rejected forms remain outside the predicate.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [owner glossary](../../../../glossary.md#term-owner))</span>
4. The scalar-immediate form ledger, generated inventory, pointwise semantic closure,
   specialized semantic-gap closure, evidence ledger, and gap matrix are regenerated from live
   facts, canonical UTF-8/LF/sealed where applicable, ordered, mutually accountable, and have
   synchronized row counts, owner partitions, strategies, hashes, schemas, and current-total
   projections. Generated rows rise only from 17,463 to 17,470; scalar-immediate rows rise only
   from 256 to 263; total rows rise only from 17,636 to 17,643; rejected rows remain 173; and
   oracle-proved rows remain 2,252.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [canonical glossary](../../../../glossary.md#term-canonical-form); [UTF-8 glossary](../../../../glossary.md#term-utf-8); [sealed glossary](../../../../glossary.md#term-sealed); [schemas glossary](../../../../glossary.md#term-schema); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
5. Every direct inventory SHA-256 consumer and direct scalar-immediate/count or
   semantic/structural closure consumer listed above is synchronized to the regenerated facts,
   while retaining fail-closed digest binding and stale-digest negative checks.<span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [digest glossary](../../../../glossary.md#term-digest); [binding glossary](../../../../glossary.md#term-binding))</span>
6. No production source, schema, capability, lowering, generator behavior, selected-route policy,
   materialization decision, semantic/structural disposition, or performance fact changes.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization))</span>
7. A separate clean documentation-focused context reviews the final diff under the General and
   Planning profiles, records no-change conclusions for Javadocs, guides, glossary, architecture,
   ADRs, Gradle, and architecture/conformance/integration tests, and validates planning links and
   status synchronization.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation context, after the final test/resource edits, runs once:

```bash
./gradlew :backends:cpu:test --tests '*CpuScalarImmediateClampMatrixTest' --tests '*CpuGeneratedCoverageCheckpointTest' --tests '*CpuGeneratedDirectEvidenceClosureTest' --tests '*CpuScalarImmediateClampCleanJavaStructuralEquivalenceTest' --tests '*CpuPointwiseSemanticClosureManifestTest' --tests '*CpuGeneratedStructuralOracleCatalogTest' --tests '*CpuAffineGeneratedCoverageFixtureTest' --tests '*CpuAffineMovementIndexingScatterRandomStructuralOracleTest' --tests '*CpuMaskedAdvancedReductionSemanticClosureTest' --tests '*CpuOrdinaryMovementFoldSemanticClosureTest' --tests '*CpuPool1dCompositionSemanticClosureTest' --tests '*CpuPool2d3dSemanticClosureTest' --tests '*CpuRandomOneHotSemanticClosureTest' --tests '*CpuGeneratedCoverageEvidenceTest' --tests '*CpuGeneratedCoverageUnsupportedBoundaryTest'
```

It also verifies live resource regeneration/sealing, exact [matrix](../../../../glossary.md#term-matrix-meanings)/inventory counts and [owner](../../../../glossary.md#term-owner)
partitions, all eight [scalar](../../../../glossary.md#term-scalar)-immediate [operation](../../../../glossary.md#operation)-form witnesses (including ADD), exact totals of
17,470 generated, 263 scalar-immediate, 17,643 total, 173 rejected, and 2,252 [oracle](../../../../glossary.md#term-test-oracle)-proved,
every regenerated inventory [digest](../../../../glossary.md#term-digest)/count consumer, resource headers/tabs/LF/final newline, and:

```bash
git diff --check
git status --short -uall
```

The documentation context reuses successful Java evidence unless it changes executable Java. It
checks local Markdown links, anchors, headings, fences, terminology/glossary impact, exact path
[scope](../../../../glossary.md#term-scope), status/[dependency](../../../../glossary.md#term-dependency) ordering, and whitespace. No [Javadoc](../../../../glossary.md#term-javadoc) generation, full [CPU](../../../../glossary.md#term-cpu) suite,
repository suite, benchmark, architecture, [conformance](../../../../glossary.md#term-conformance), or integration [run](../../../../glossary.md#term-run) is required because
this correction changes no production or public/executable behavior beyond test evidence.

## Dependencies and follow-up tasks

- Complete CPU 0008Q1, CPU 0009, CPU 0009A, and CPU 0009G, including their recorded
  documentation passes; current scalar-immediate matrix/oracle and generated coverage inventory.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [scalar glossary](../../../../glossary.md#term-scalar); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- CPU 0009G1 completes the correction immediately after 0009G. CPU 0010 remains the next Draft
  CPU task; this change does not ready or execute it.<span data-glossary-navigation="true"> ([ready glossary](../../../../glossary.md#term-ready))</span>
- No follow-up is expected. A current-source inability to select one required witness, or evidence
  of a real capability/selection discrepancy, requires an immediate narrowly named Draft
  follow-up and leaves this task incomplete.

## Architecture impact

Expected impact: None. Stop for an architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), capability, [schema](../../../../glossary.md#term-schema),
selection-policy, or public-contract conflict.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009G1. Read AGENTS.md,
ARCHITECTURE.md, docs/architecture/current-architecture-plan.md, the Planning Guide, CPU master
plan, CPU 0008Q1, CPU 0009, CPU 0009A, CPU 0009G, this task, and the scalar-immediate matrix plus
generated-inventory checkpoint sources/resources. Implement exactly this evidence-only scalar
selection correction. Do not change production behavior, capability, lowering, generator schema,
strategy policy, performance facts, commit, stage, push, or use GSD. Stop for an architecture or
scope conflict. After focused validation, hand the stable diff and exact evidence to a separate
clean documentation-focused context following documentation-rules.md; do not mark this task
Complete until that review is recorded.
```

## Local decisions

- One direct finite scalar witness per missing operation form is sufficient. It proves selected
  scalar presence for that operation only; it does not expand the finite matrix into a per-form
  vector or parallel orchestration product.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [scalar glossary](../../../../glossary.md#term-scalar); [operation glossary](../../../../glossary.md#operation); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [vector glossary](../../../../glossary.md#term-vector))</span>
- `SCALAR_ADD` remains the established orchestration witness because its scalar-immediate finite
  matrix already proves all four current orchestration selections. The new witnesses are not
  substitutes for that evidence.
- “Meaningful scalar” is a live, source-derived checkpoint predicate, not a manually maintained
  list or an assumption that every generated operation must select every strategy.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Known limitations

The correction proves only [operation](../../../../glossary.md#operation)-specific selected-[scalar](../../../../glossary.md#term-scalar) evidence for current live forms. It
does not prove universal scalar availability, change scalar strategy quality, add a performance
claim, or broaden [finite](../../../../glossary.md#term-finite) immediate/bound coverage. Existing partial structural and performance
dispositions remain unchanged.

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation added seven live-prepared, selected-[`SCALAR`](../../../../glossary.md#term-scalar) [owners](../../../../glossary.md#term-owner):
`SCALAR-WITNESS-SUB`, `SCALAR-WITNESS-MUL`, `SCALAR-WITNESS-DIV`, `SCALAR-WITNESS-MIN`,
`SCALAR-WITNESS-MAX`, `SCALAR-WITNESS-POW`, and `SCALAR-WITNESS-CLAMP`. Existing
`ORCHESTRATION-SCALAR` remains the ADD witness and the existing four-[owner](../../../../glossary.md#term-owner) ADD orchestration
basis remains unchanged. The [checkpoint](../../../../glossary.md#term-checkpoint) now derives every live scalar-meaningful [operation](../../../../glossary.md#operation) form
and requires selected-scalar coverage, while separately sealing exactly eight direct witness
owners: ADD orchestration plus the seven operation-specific witnesses.

The regenerated inventory [SHA-256](../../../../glossary.md#term-sha-256) is
`1dcb69796c00fe3793d86f3f4cc3e816176062a45312ddbbaadfba9f8036cf20`.
The sealed evidence records 17,470 generated rows, 263 scalar-immediate rows, 17,643 total rows,
173 rejected rows, 2,252 [oracle](../../../../glossary.md#term-test-oracle)-proved rows, 15,218 oracle-partial rows, and 17,470
performance-partial rows. The [pointwise](../../../../glossary.md#term-pointwise) [manifest](../../../../glossary.md#term-manifest) contains 1,108 generated rows and the structural
catalog contains 6,016 categories. Live resource regeneration and sealing, [canonical](../../../../glossary.md#term-canonical-form) [UTF-8](../../../../glossary.md#term-utf-8)/LF
and final-newline checks, old-[digest](../../../../glossary.md#term-digest) absence across the 21 [CPU](../../../../glossary.md#term-cpu) paths, exact owner/count
reconciliation, and `git diff --check` passed in the implementation/coordinator [handoff](../../../../glossary.md#term-handoff).

After the final Java edits, the coordinator ran the focused command in this task exactly once. It
passed 87 tests with zero failures; [Gradle](../../../../glossary.md#term-gradle) reported `BUILD SUCCESSFUL` in 1 minute 38 seconds. No
failed Gradle [run](../../../../glossary.md#term-run) was handed off. The recorded intermediate failure was a [planning](../../../../glossary.md#term-planning) [scope](../../../../glossary.md#term-scope) check:
the original five-path ceiling omitted the transitive SHA/count closure. The implementation
corrected that condition by staying within the expanded, enumerated 21-path CPU scope and
preserving every [fail-closed](../../../../glossary.md#term-fail-closed) digest assertion.

Clean documentation-focused finalization ran in context `/root`. It read the governing
architecture, current architecture explanation, documentation rules, General and Planning
profiles, planning guide, roadmap, CPU master plan, this task, the complete final diff, and all
21 CPU test/resource paths. It independently confirmed the inventory digest, exact resource
counts, seven witness rows, unchanged ADD orchestration basis, exact eight-owner checkpoint, and
absence of changed Java comments or Javadocs. It then validated local Markdown links and anchors,
headings, fences, status/[dependency](../../../../glossary.md#term-dependency) [synchronization](../../../../glossary.md#term-synchronization), exact 24-path scope, no production/build/
public-document paths, whitespace, and final worktree status without rerunning Gradle.

## Implementation notes

- Added one direct, live-prepared selected-[`SCALAR`](../../../../glossary.md#term-scalar) witness for each of SUB, MUL, DIV, MIN, MAX,
  POW, and CLAMP, retaining ADD as the orchestration basis.
- Made scalar-form coverage source-derived and fail closed, while sealing the distinct exact
  eight-owner direct witness basis.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [owner glossary](../../../../glossary.md#term-owner))</span>
- Regenerated all six authorized resources and synchronized the 15 authorized direct Java
  consumers without changing production behavior, schemas, strategy policy, or evidence
  dispositions.<span data-glossary-navigation="true"> ([schemas glossary](../../../../glossary.md#term-schema))</span>

## Completion summary

- Completed changes: Added seven operation-specific selected-scalar witnesses, derived and sealed
  live scalar-form coverage, retained the ADD orchestration basis, and synchronized the complete
  finite digest/count evidence closure.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [scalar glossary](../../../../glossary.md#term-scalar); [sealed glossary](../../../../glossary.md#term-sealed); [finite glossary](../../../../glossary.md#term-finite); [digest glossary](../../../../glossary.md#term-digest))</span>
- Files changed or created: The 21 CPU test/resource paths enumerated under Affected files, this
  task, the CPU master plan, and the repository roadmap; 24 paths total.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Tests and validation: The coordinator's final focused Gradle command passed 87 tests with zero
  failures and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Gradle glossary](../../../../glossary.md#term-gradle))</span>`BUILD SUCCESSFUL` in 1 minute 38 seconds. Resource regeneration/sealing,
  digest/count/owner reconciliation, canonical line endings/final newlines, stale-digest absence,
  Markdown links/anchors/headings/fences, exact scope, status/dependencies, <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [canonical glossary](../../../../glossary.md#term-canonical-form); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`,
  and `git status --short -uall` passed.
- Documentation-agent review: Clean context `/root` applied the General and Planning profiles to
  the complete final diff and all 21 CPU evidence paths. It changed only the three authorized
  planning files and did not rerun Gradle.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: This task and the two status surfaces now record the completed correction
  and exact evidence. No public guide or architecture explanation changed because the work is
  test/evidence-only and changes no public, runtime, backend, or architectural contract.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- Javadoc review: The 15 changed Java files are tests or a test oracle, their changed lines add
  evidence logic rather than API contracts, and no comments or Javadocs changed or became stale.
  No Javadoc update or generation is needed.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [test oracle glossary](../../../../glossary.md#term-test-oracle); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Glossary impact: No reusable terminology or public concept changed; no glossary update is
  needed.
- Additional no-change conclusions: No architecture/ADR, production source, Gradle, architecture
  test, backend-conformance test, integration test, benchmark, or generator-schema change is
  needed.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test); [schema glossary](../../../../glossary.md#term-schema))</span>
- Unresolved issues: None.
- Follow-up required: None for CPU 0009G1. CPU 0010 remains the next Draft CPU task and was not
  readied or executed here.

Status: Complete
