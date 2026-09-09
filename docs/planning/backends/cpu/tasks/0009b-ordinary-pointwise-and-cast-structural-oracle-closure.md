# Task 0009B: Ordinary Pointwise and CAST Structural-Oracle Closure

## Status

Complete

## Goal

Close truthful generated-support and clean-Java semantic evidence for the current ordinary pointwise and `CAST` CPU inventory, then promote only exact Class-File structural projections that a genuinely non-tautological generated-versus-clean-Java comparator proves. The owned generated set is exactly the 845 `pointwise-matrix` rows: 665 ordinary pointwise and 180 ordered `CAST`. Rejections are adjacent unsupported boundaries, not generated coverage: 152 are `rejected:pointwise` and 21 are other-category rows, 173 total. CPU 0009A's 256 scalar-immediate/clamp rows are excluded.

For every owned generated row, establish exact provenance; generation, loading, and typed invocation; independently authored typed optimal clean-Java execution; meaningful value/range behavior; every applicable type, ordered Java-array/`MemorySegment` carrier role, contiguous/non-contiguous access regime, and scalar/vector/parallel-scalar/parallel-vector strategy; active hot-path hygiene; deterministic evidence; and explicit rejected/not-applicable boundaries. Promote a structural disposition only for the exact projection the comparator proves. Otherwise retain `PARTIAL_CLASSFILE_NO_CLEAN_JAVA_ORACLE`. Performance remains `PARTIAL_NO_REPRESENTATIVE_BENCHMARK` unless exact reusable row evidence already exists; this task neither runs nor requires five-fork benchmarks.

## Scope

- Retain the 17,636-row inventory (17,463 `GENERATED`, 173 exact rejections). This task owns only 845 generated `pointwise-matrix` rows. The cross-category rejection total is 173 rows: 152 `rejected:pointwise` rows and 21 rejected rows in other categories. The digest-bound semantic manifest partitions the generated rows into 665 `pointwise` and 180 `cast` owners and must remain globally joined.
- Cover current ordinary arithmetic, comparison, classification, logical, unary/activation, and `WHERE` forms with their source-defined data-type and ordered input/output roles; cover all 36 ordered `CAST` source/target pairs, including represented-bit same-type identities and existing cross-type boundaries.
- Derive all fixtures from provider, preparer, lowering, generator, and checked inventory—not names, form IDs, hashes, or catalog categories. Preserve evidence key, operation/form, IR structural key, typed entry descriptor, ordered carrier/access roles, requested/selected strategy, materialization, class SHA-256, normalized key, and exact disposition.
- Generate, define/load, and invoke every exact entry. Compare it to an independently authored typed clean-Java counterpart that calls neither generated code, lowering, reference kernels, nor a generic carrier bridge. Test normal and applicable exceptional values, empty/non-empty ranges, nonzero starts/tails, and untouched sentinels. Include signed zero, NaN, infinity, subnormal, overflow/underflow, integer bounds, BOOL canonicalization, and represented BFLOAT16 cases where applicable.
- Cover every applicable array/segment and ordered mixed carrier role, dense/contiguous and admitted non-contiguous regime, and all selected scalar/vector/parallel-scalar/parallel-vector strategies. Record unavailable cases as exact provider, preparer, binding, selection, semantic, or not-applicable boundaries; never infer them from nearby forms.
- Add narrow active Class-File hygiene for paired selected entries. A closed invocation/overhead allowlist must inspect actual targets and have a failing mutation or equivalent independent negative control. Reject helper/reference leakage, allocation, boxing, reflection, method-handle/dynamic invocation, string/map or generic-carrier dispatch, and avoidable virtual/interface dispatch. Permitted numerical, Vector API, and Foreign Function and Memory API calls must be exact and justified.
- A structural comparator may promote only an exact projection it extracts from both independent artifacts: typed ABI, ordered carrier accesses/stores, selected realization, access/address regime, conversion/numerical order, active invokes, and narrow direct hot-loop/dataflow facts. It must fail for changed compared facts and reject a deliberately changed allowed projection. Form IDs, hashes, normalized-body identity, catalog membership, and self-comparison are never equivalence proof.
- Synchronize only row-exact inventory, disposition-registry, and evidence-ledger facts required by a proof. Unproved structural rows remain partial. Preserve deterministic schema, headers, tabs, LF endings, digests, and accounting.
- Use a separate clean documentation-focused context after executable work stabilizes to finalize planning/evidence, Javadocs, guide, glossary, and documentation-impact review.

## Out of scope

- Production Java; generators/lowering; Model, Compiler, Prepare, Runtime, Engine, cache/schema, capability, admission, selection, materialization, native-route, or public-API changes.
- CPU 0009A scalar-immediate/clamp; affine, movement, indexing, scatter, random, aggregate, scan, ordering, fold, reduction, normalization, loss, MATMUL, convolution, pooling, attention, batch norm, and later 0009 children.
- General bytecode verification, CFG isomorphism, full abstract interpretation, arbitrary-bytecode proof, a bounded predecessor scan presented as general dataflow proof, literal generated-versus-`javac` CFG equality, or mutation for every structural dimension.
- New performance evidence, performance promotion, or five-fork runs. Existing performance is reusable only after exact row/hash/counterpart/protocol/projection matching.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): CPU preparation owns lowering/selection; generated code preserves optimal clean-Java semantics, hot-loop/dataflow, and avoidable-overhead profile without byte, `javac`, CFG, or JIT identity requirements.
- [Current architecture navigation](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md), [General profile](../../../../developer-guide/documentation/general-style.md), [Planning profile](../../../../developer-guide/documentation/planning-style.md), [CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [CPU 0009A](0009a-scalar-immediate-clamp-clean-java-structural-equivalence.md), and completed [CPU 0008J](0008j-bfloat16-scalar-pointwise-closure.md), [CPU 0008K](0008k-cross-type-cast-execution.md), and [CPU 0008L](0008l-pointwise-simd-mask-output-closure.md).

## Architecture constraints

- Evidence is test-only and must not alter capability, admission, selection, identity, route choice, lowering, or runtime behavior. CPU prepare owns lowering/selection; Runtime invokes prepared artifacts only.
- The clean-Java specialized implementation is independently authored and typed; shared fixture data is allowed, shared execution or identity is not.
- Provider support, admission, binding/invocation, selected strategy, and production selection are independent fail-closed facts.
- Support/semantic closure is neither structural nor performance closure; dispositions are independent.

## Package impact

Existing test-only package:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` owns fixtures, clean-Java counterparts, generated-entry execution, paired structural comparison, hygiene, and inventory evidence.

Packages added or changed:

- None. No production type, package, public Javadoc, or module dependency is authorized.

Type placement:

- `...CpuPointwiseSemanticClosureManifestTest` — exact 665/180/845 semantic ownership.
- `...CpuPointwiseGeneratedKernelTest` and `...CpuCastGeneratedKernelTest` — direct generated-entry sources, changed only for missing source-derived witnesses.
- `...CpuOrdinaryPointwiseCastCleanJavaOracle` (new test-only) — independent typed clean-Java artifacts and ABI/semantic bindings.
- `...CpuOrdinaryPointwiseCastStructuralOracleTest` (new test-only) — paired projection comparator, hygiene, allowlist, and negative controls.

## Affected files and maximum scope

Expected CPU test/evidence paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseSemanticClosureManifestTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCastGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastStructuralOracleTest.java`
- Existing checkpoint/registry tests and checked pointwise manifest, inventory, disposition-registry, or evidence-ledger resources only when row-exact synchronization requires them.

Planning paths: this task, [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), [CPU master plan](../master-plan.md), and [roadmap](../../../roadmap.md).

At most ten CPU test/evidence paths and four planning paths. No production Java, Gradle, architecture docs/ADRs, public/API Javadocs, guides, glossary, architecture, conformance, or integration tests. If more paths, a production change, a new semantic family, or an unproved projection is needed, stop, retain that row partial, and propose a narrowly named Draft follow-up.

## Acceptance criteria

- Inventory remains 17,636 rows: 17,463 generated and 173 exact rejections. Exactly 845 generated pointwise-matrix rows reconcile once: 665 ordinary pointwise and 180 CAST; 0009A's 256 scalar-immediate/clamp rows are absent.
- Every owned generated row has exact provenance, generated hash/typed ABI, generation/load/invocation, and independent typed clean-Java agreement for all applicable normal, range, special-value, and outside-range-sentinel cases.
- Evidence covers all applicable data types, ordered array/segment carrier roles, contiguous/admitted non-contiguous regimes, and selected four strategy modes. Each unavailable/inapplicable case is explicit; the 152 `rejected:pointwise` rows and 21 other-category rejected rows remain exact and fail closed.
- Paired entries pass active hygiene and a closed invocation/overhead allowlist. Tests reject forbidden mechanisms and an unused/dead allowlist entry, preventing dead comparator/allowlist evidence.
- A promoted structural disposition has an exact declared, non-tautological paired comparator with negative controls. It cannot use IDs, hashes, normalized-body keys, or catalog categories as proof. All other rows remain `PARTIAL_CLASSFILE_NO_CLEAN_JAVA_ORACLE`.
- Resources fail closed for stale digest, duplicate/orphan/missing owner, altered accounting, malformed schema, changed ABI/hash, or unknown disposition. Performance stays `PARTIAL_NO_REPRESENTATIVE_BENCHMARK` absent exact reusable row evidence.
- A separate documentation-focused pass records reasoned no-change conclusions for API Javadocs, guides, glossary, architecture/ADRs, architecture/conformance/integration tests, Gradle, and production packages.

## Tests / validation

Implementation context runs the manifest, direct pointwise/CAST matrices, independent clean-Java matrix, paired structural/hygiene/negative controls, inventory checkpoint, and unsupported boundaries:

```bash
./gradlew :backends:cpu:test --tests '*CpuPointwiseSemanticClosureManifestTest' --tests '*CpuPointwiseGeneratedKernelTest' --tests '*CpuCastGeneratedKernelTest' --tests '*CpuOrdinaryPointwiseCastStructuralOracleTest' --tests '*CpuGeneratedCoverageCheckpointTest' --tests '*CpuGeneratedCoverageUnsupportedBoundaryTest'
```

It validates headers/tabs/LF/final newline/digests, ownership/disposition joins, exact 665/180/845/173 counts, and `git diff --check`.

Documentation context reuses successful Java evidence unless executable Java changes; it validates Markdown links, anchors, fences, path scope, statuses/dependencies, terminology, whitespace, and:

```bash
git diff --check
```

Repository-wide/capability-checkpoint validation and performance closure defer to CPU 0009G/CI. No five-fork command is required.

## Dependencies

- Complete CPU 0009A and its documentation pass.
- Complete CPU 0008J, 0008K, and 0008L; current checked inventory, semantic manifest, disposition registry, and evidence ledger.
- Java 26 Class-File, Vector API, and Foreign Function and Memory API test environment.

## Follow-up tasks

- CPU 0009C is the active Ready child with its detailed bounded specification; it owns affine, movement, indexing, scatter, and random closure. CPU 0009D remains the next ordered Draft child.
- A row needing production/semantic work or a comparator beyond this scope remains partial and becomes a narrow Draft follow-up only at the active frontier.
- CPU 0009G owns representative performance and final checkpoint closure.

## Architecture impact

Expected impact: None. Stop for architecture, dependency, production-selection, schema, generator, capability, admission, or semantics changes.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU task 0009B. Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md, the Planning Guide, CPU master plan, CPU 0009, CPU 0009A, CPU 0008J/0008K/0008L, and this task. Implement exactly this bounded ordinary-pointwise/CAST test-evidence closure; do not change production behavior or claim general bytecode/CFG equivalence. Promote structural evidence only through an exact non-tautological paired clean-Java comparator; leave all other structural rows partial. Do not run or promote new five-fork performance evidence. Stop for scope or architecture conflict. Do not commit or push. After stable evidence, hand the diff and tests to a separate clean documentation-focused context under docs/developer-guide/documentation-rules.md. Update this task with results; do not mark Complete before that pass finishes.
```

## Local decisions

- Ownership is 845 generated rows, not the 6,009 catalog or the 1,101-pointwise-opcode universe; the remaining 256 are CPU 0009A's scalar-immediate/clamp rows.
- Structural evidence is narrow paired observation. A bounded predecessor walk may support local hygiene but is not general dataflow proof.
- Prior CAST/pointwise performance is not a shortcut; reuse requires exact row/hash/counterpart/protocol/projection matching.

## Known limitations

No arbitrary-bytecode equivalence, arbitrary immediate values, broader-family behavior, or complete generated/direct performance parity is proved. Semantic support without an exact paired comparator remains structurally partial. The 6,009 catalog is discovery/accounting support, never a closure denominator or equivalence proof.

## Validation evidence

Implementation context final evidence (reused; executable Java has not changed afterward):

- Focused six-suite command covering the pointwise semantic manifest, direct pointwise and CAST matrices, ordinary paired structural oracle, generated-coverage checkpoint, and unsupported-boundary checks passed with 60 tests, 1 expected skip, and 0 failures/errors (`BUILD SUCCESSFUL` in 58s).
- Full CPU validation passed with 184 suites, 899 tests, 27 expected skips, and 0 failures/errors (`BUILD SUCCESSFUL` in 2m22s).
- The implementation's independent technical review found no HIGH findings and approved the result after typed `MemorySegment` access, real forward-branch direction, and rejection-accounting corrections.
- This documentation pass did not rerun Java/Gradle suites. The final `hasForwardSemanticBranch` to `hasForwardBranch` terminology change in the implementation diff changed no executable behavior, so duplicate suite execution is prohibited by the planning rule.
- Documentation validation: reviewed exact final diff and the two new test-only Javadocs/comments; checked task/master-plan/roadmap status and dependency synchronization; checked Markdown links, anchors, fenced blocks, and terminology; searched affected documentation and Java comments for stale `173 rejected pointwise` and `semantic branch` wording; `git diff --check` passed.

No-change conclusions: `ARCHITECTURE.md`, architecture explanations, ADRs, and architecture tests remain accurate because this is test-evidence/documentation work with no architecture or dependency change. Public API Javadocs and the glossary need no update because no public terminology or behavior changed. Backend conformance and integration tests need no update because no backend contract or end-to-end behavior changed. Gradle/build configuration and unrelated modules need no update because no build or module change occurred.

## Implementation notes

The test-only implementation supplies independent typed optimal clean-Java semantic execution for each of the 845 owned selected rows over their recorded ranges, including source-derived array-to-segment and segment-to-array carrier-role witnesses and legal safe aliases. Its bounded paired Class-File projection retains typed ABI; ordered carrier reads and writes; typed `MemorySegment` `ValueLayout`, width, and direction; conversions; numerical invokes; vector-versus-scalar realization facts; and actual forward/backward branch direction. Negative controls reject carrier/layout/order, store, conversion, invoke, and topology drift.

The projection is deliberately bounded. It does not prove literal Class-File or CFG identity, JIT assembly identity, universal alias/overlap semantics, or generated/direct performance parity. Performance remains partial and fail-closed; no five-fork benchmark is claimed.

## Completion summary

- Completed changes: finalized the test-evidence closure record for 845 generated rows (665 ordinary pointwise and 180 ordered CAST), corrected rejection accounting to 152 pointwise-category plus 21 other-category rows, and recorded the bounded structural proof, validation, review, and limitations.
- Files changed or created: this task; CPU master plan; roadmap; parent CPU 0009 task; and the implementation-owned test-only oracle and structural-oracle test sources.
- Tests and validation: reused focused six-suite evidence (60 tests, 1 expected skip, 0 failures/errors, 58s) and full CPU evidence (184 suites, 899 tests, 27 expected skips, 0 failures/errors, 2m22s); Markdown/static review and `git diff --check` passed.
- Documentation-agent review: completed in this clean context after the implementation evidence stabilized.
- Documentation impact: no architecture, guide, public API, glossary, configuration, conformance, integration, or unrelated-module update is required for this test-only closure.
- Javadoc review: reviewed the two new test-only contracts; no documentation-context Java comment edit was needed because branch terminology is `forward branch`, and vector/scalar and bounded-projection boundaries are explicit.
- Glossary impact: None; no reusable public term changed.
- Unresolved issues: None within CPU 0009B's bounded scope.
- Follow-up required: Implement the active Ready CPU 0009C specification before creating the next ordered Draft child, CPU 0009D.

Status: Complete
