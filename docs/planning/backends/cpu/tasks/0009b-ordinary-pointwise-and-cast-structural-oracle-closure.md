# Task 0009B: Ordinary [Pointwise](../../../../glossary.md#term-pointwise) and CAST Structural-[Oracle](../../../../glossary.md#term-test-oracle) Closure

## Reader context

This task records ordinary pointwise and `CAST` closure for 845 generated rows: 665 pointwise and
180 `CAST`, excluding 0009A's scalar-immediate/clamp forms.
`CpuOrdinaryPointwiseCastCleanJavaOracle` supplies independently authored typed execution.
`CpuOrdinaryPointwiseCastStructuralOracleTest` instead compares a bounded Class-File projection,
including ordered carrier accesses, conversions, invokes, realization, and branch direction.
Negative controls test that changed compared facts are rejected; hashes and catalog membership are
not equivalence proof.

CPU Prepare owns lowering and selection, while Runtime invokes prepared artifacts. Array and
`MemorySegment` input/output roles belong to the typed entry boundary, including layout width and
access direction. The codegen.emit test package owns this evidence without changing production
capability or admission. Semantic, structural, and performance dispositions remain independent;
recorded strategy coverage does not select production defaults.

Current-topic navigation (not a replacement for the historical record):

- [current ordered CAST conversion meanings](../../../../reference/tensor/selection-casts.md#cast-expressions) — Explains direct rounding, saturation, bit preservation, and why conversion pairs are directional.
- [current generated-class boundary](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit) — Explains emitters, typed entries, hidden classes, and the limits of Class-File inspection.

## Status

Complete

## Goal

Close truthful generated-support and clean-Java semantic evidence for the current ordinary [pointwise](../../../../glossary.md#term-pointwise) and `CAST` [CPU](../../../../glossary.md#term-cpu) inventory, then promote only exact Class-File structural projections that a genuinely non-tautological generated-versus-clean-Java comparator proves. The owned generated set is exactly the 845 `pointwise-matrix` rows: 665 ordinary pointwise and 180 ordered `CAST`. Rejections are adjacent unsupported boundaries, not generated coverage: 152 are `rejected:pointwise` and 21 are other-category rows, 173 total. CPU 0009A's 256 [scalar](../../../../glossary.md#term-scalar)-immediate/clamp rows are excluded.

For every owned generated row, establish exact [provenance](../../../../glossary.md#provenance); generation, loading, and typed invocation; independently authored typed optimal clean-Java execution; meaningful value/range behavior; every applicable type, ordered Java-array/[`MemorySegment`](../../../../glossary.md#term-memorysegment) [carrier](../../../../glossary.md#term-storage-carrier) role, [contiguous](../../../../glossary.md#term-contiguous)/non-contiguous access regime, and scalar/[vector](../../../../glossary.md#term-vector)/parallel-scalar/parallel-vector strategy; active hot-path hygiene; deterministic evidence; and explicit rejected/not-applicable boundaries. Promote a structural disposition only for the exact projection the comparator proves. Otherwise retain `PARTIAL_CLASSFILE_NO_CLEAN_JAVA_ORACLE`. Performance remains `PARTIAL_NO_REPRESENTATIVE_BENCHMARK` unless exact reusable row evidence already exists; this task neither runs nor requires five-[fork](../../../../glossary.md#term-fork-meanings) benchmarks.

## [Scope](../../../../glossary.md#term-scope)

- Retain the 17,636-row inventory (17,463 `GENERATED`, 173 exact rejections). This task owns only 845 generated `pointwise-matrix` rows. The cross-category rejection total is 173 rows: 152 `rejected:pointwise` rows and 21 rejected rows in other categories. The [digest](../../../../glossary.md#term-digest)-bound semantic [manifest](../../../../glossary.md#term-manifest) partitions the generated rows into 665 [`pointwise`](../../../../glossary.md#term-pointwise) and 180 `cast` [owners](../../../../glossary.md#term-owner) and must remain globally joined.
- Cover current ordinary arithmetic, comparison, classification, logical, [unary](../../../../glossary.md#term-unary)/[activation](../../../../glossary.md#term-activation), and `WHERE` forms with their source-defined data-type and ordered input/output roles; cover all 36 ordered `CAST` source/[target](../../../../glossary.md#term-target) pairs, including [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) same-type [identities](../../../../glossary.md#term-identity-meanings) and existing cross-type boundaries.
- Derive all fixtures from provider, preparer, [lowering](../../../../glossary.md#lowering), generator, and checked inventory—not names, form IDs, [hashes](../../../../glossary.md#term-hash), or catalog categories. Preserve evidence key, [operation](../../../../glossary.md#operation)/form, [IR](../../../../glossary.md#term-intermediate-representation-ir) structural key, typed entry [descriptor](../../../../glossary.md#term-descriptor), ordered [carrier](../../../../glossary.md#term-storage-carrier)/access roles, requested/selected strategy, [materialization](../../../../glossary.md#materialization), class [SHA-256](../../../../glossary.md#term-sha-256), normalized key, and exact disposition.
- Generate, define/load, and invoke every exact entry. Compare it to an independently authored typed clean-Java counterpart that calls neither generated code, lowering, reference [kernels](../../../../glossary.md#kernel), nor a generic carrier bridge. Test normal and applicable exceptional values, empty/non-empty ranges, nonzero starts/tails, and untouched sentinels. Include [signed zero](../../../../glossary.md#term-signed-zero), [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [subnormal](../../../../glossary.md#term-subnormal), [overflow](../../../../glossary.md#term-overflow)/[underflow](../../../../glossary.md#term-underflow), integer bounds, [BOOL](../../../../glossary.md#data-type--datatype) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), and represented [BFLOAT16](../../../../glossary.md#term-bfloat16) cases where applicable.
- Cover every applicable array/segment and ordered mixed carrier role, dense/[contiguous](../../../../glossary.md#term-contiguous) and admitted non-contiguous regime, and all selected [scalar](../../../../glossary.md#term-scalar)/[vector](../../../../glossary.md#term-vector)/parallel-scalar/parallel-vector strategies. Record unavailable cases as exact provider, preparer, [binding](../../../../glossary.md#term-binding), selection, semantic, or not-applicable boundaries; never infer them from nearby forms.
- Add narrow active Class-File hygiene for paired selected entries. A closed invocation/overhead allowlist must inspect actual targets and have a failing mutation or equivalent independent negative control. Reject helper/reference leakage, allocation, boxing, reflection, method-handle/dynamic invocation, string/map or generic-carrier [dispatch](../../../../glossary.md#term-dispatch), and avoidable virtual/interface dispatch. Permitted numerical, [Vector API](../../../../glossary.md#term-vector-api), and [Foreign Function and Memory API](../../../../glossary.md#application-binary-interface-and-native-bridge) calls must be exact and justified.
- A structural comparator may promote only an exact projection it extracts from both independent [artifacts](../../../../glossary.md#term-artifact): typed [ABI](../../../../glossary.md#term-application-binary-interface-abi), ordered carrier accesses/stores, selected realization, access/address regime, conversion/numerical order, active invokes, and narrow direct hot-loop/[dataflow](../../../../glossary.md#term-dataflow) facts. It must fail for changed compared facts and reject a deliberately changed allowed projection. Form IDs, hashes, normalized-body [identity](../../../../glossary.md#term-identity-meanings), catalog membership, and self-comparison are never equivalence proof.
- Synchronize only row-exact inventory, disposition-[registry](../../../../glossary.md#term-registry), and evidence-ledger facts required by a proof. Unproved structural rows remain partial. Preserve deterministic [schema](../../../../glossary.md#term-schema), headers, tabs, LF endings, digests, and accounting.
- Use a separate clean documentation-focused context after executable work stabilizes to finalize [planning](../../../../glossary.md#term-planning)/evidence, Javadocs, guide, glossary, and documentation-impact review.

## Out of [scope](../../../../glossary.md#term-scope)

- Production Java; generators/[lowering](../../../../glossary.md#lowering); [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Engine](../../../../glossary.md#term-engine), [cache](../../../../glossary.md#term-cache)/[schema](../../../../glossary.md#term-schema), capability, [admission](../../../../glossary.md#term-admission), selection, [materialization](../../../../glossary.md#materialization), native-route, or public-[API](../../../../glossary.md#term-application-programming-interface-api) changes.
- [CPU](../../../../glossary.md#term-cpu) 0009A [scalar](../../../../glossary.md#term-scalar)-immediate/clamp; affine, movement, indexing, scatter, random, aggregate, [scan](../../../../glossary.md#term-cumulative-scan), ordering, fold, reduction, [normalization](../../../../glossary.md#term-normalization), [loss](../../../../glossary.md#loss), [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), [convolution](../../../../glossary.md#term-channels-first-convolution-layers), [pooling](../../../../glossary.md#term-pooling), attention, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [norm](../../../../glossary.md#term-norm), and later 0009 children.
- General [bytecode](../../../../glossary.md#term-bytecode) verification, [CFG](../../../../glossary.md#term-control-flow-graph-cfg) isomorphism, full abstract interpretation, arbitrary-bytecode proof, a bounded predecessor scan presented as general [dataflow](../../../../glossary.md#term-dataflow) proof, literal generated-versus-`javac` CFG equality, or mutation for every structural [dimension](../../../../glossary.md#dimension).
- New performance evidence, performance promotion, or five-[fork](../../../../glossary.md#term-fork-meanings) runs. Existing performance is reusable only after exact row/[hash](../../../../glossary.md#term-hash)/counterpart/protocol/projection matching.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): [CPU](../../../../glossary.md#term-cpu) preparation owns [lowering](../../../../glossary.md#lowering)/selection; generated code preserves optimal clean-Java semantics, hot-loop/[dataflow](../../../../glossary.md#term-dataflow), and avoidable-overhead profile without byte, `javac`, [CFG](../../../../glossary.md#term-control-flow-graph-cfg), or [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) [identity](../../../../glossary.md#term-identity-meanings) requirements.
- [Current architecture navigation](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [documentation rules](../../../../developer-guide/documentation-rules.md), [General profile](../../../../developer-guide/documentation/general-style.md), [Planning profile](../../../../developer-guide/documentation/planning-style.md), [CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [CPU 0009A](0009a-scalar-immediate-clamp-clean-java-structural-equivalence.md), and completed [CPU 0008J](0008j-bfloat16-scalar-pointwise-closure.md), [CPU 0008K](0008k-cross-type-cast-execution.md), and [CPU 0008L](0008l-pointwise-simd-mask-output-closure.md).

## Architecture constraints

- Evidence is test-only and must not alter capability, [admission](../../../../glossary.md#term-admission), selection, [identity](../../../../glossary.md#term-identity-meanings), route choice, [lowering](../../../../glossary.md#lowering), or [runtime](../../../../glossary.md#term-runtime) behavior. [CPU](../../../../glossary.md#term-cpu) [prepare](../../../../glossary.md#prepare) owns lowering/selection; Runtime invokes prepared [artifacts](../../../../glossary.md#term-artifact) only.
- The clean-Java specialized implementation is independently authored and typed; shared [fixture](../../../../glossary.md#term-fixture) data is allowed, shared execution or identity is not.
- Provider support, admission, [binding](../../../../glossary.md#term-binding)/invocation, selected strategy, and production selection are independent [fail-closed](../../../../glossary.md#term-fail-closed) facts.
- Support/semantic closure is neither structural nor performance closure; dispositions are independent.

## Package impact

Existing test-only package:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` owns fixtures, clean-Java counterparts, generated-entry execution, paired structural comparison, hygiene, and inventory evidence.

Packages added or changed:

- None. No production type, package, public [Javadoc](../../../../glossary.md#term-javadoc), or [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency) is authorized.

Type placement:

- `...CpuPointwiseSemanticClosureManifestTest` — exact 665/180/845 semantic [ownership](../../../../glossary.md#term-ownership-meanings).
- `...CpuPointwiseGeneratedKernelTest` and `...CpuCastGeneratedKernelTest` — direct generated-entry sources, changed only for missing source-derived witnesses.
- `...CpuOrdinaryPointwiseCastCleanJavaOracle` (new test-only) — independent typed clean-Java [artifacts](../../../../glossary.md#term-artifact) and [ABI](../../../../glossary.md#term-application-binary-interface-abi)/semantic [bindings](../../../../glossary.md#term-binding).
- `...CpuOrdinaryPointwiseCastStructuralOracleTest` (new test-only) — paired projection comparator, hygiene, allowlist, and negative controls.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

Expected [CPU](../../../../glossary.md#term-cpu) test/evidence paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseSemanticClosureManifestTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCastGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastStructuralOracleTest.java`
- Existing [checkpoint](../../../../glossary.md#term-checkpoint)/[registry](../../../../glossary.md#term-registry) tests and checked [pointwise](../../../../glossary.md#term-pointwise) [manifest](../../../../glossary.md#term-manifest), inventory, disposition-registry, or evidence-ledger resources only when row-exact [synchronization](../../../../glossary.md#term-synchronization) requires them.

[Planning](../../../../glossary.md#term-planning) paths: this task, [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), [CPU master plan](../master-plan.md), and [roadmap](../../../roadmap.md).

At most ten CPU test/evidence paths and four planning paths. No production Java, [Gradle](../../../../glossary.md#term-gradle), architecture docs/ADRs, public/[API](../../../../glossary.md#term-application-programming-interface-api) Javadocs, guides, glossary, architecture, [conformance](../../../../glossary.md#term-conformance), or integration tests. If more paths, a production change, a new semantic family, or an unproved projection is needed, stop, retain that row partial, and propose a narrowly named Draft follow-up.

## Acceptance criteria

- Inventory remains 17,636 rows: 17,463 generated and 173 exact rejections. Exactly 845 generated [pointwise](../../../../glossary.md#term-pointwise)-[matrix](../../../../glossary.md#term-matrix-meanings) rows reconcile once: 665 ordinary pointwise and 180 CAST; 0009A's 256 [scalar](../../../../glossary.md#term-scalar)-immediate/clamp rows are absent.
- Every owned generated row has exact [provenance](../../../../glossary.md#provenance), generated [hash](../../../../glossary.md#term-hash)/typed [ABI](../../../../glossary.md#term-application-binary-interface-abi), generation/load/invocation, and independent typed clean-Java agreement for all applicable normal, range, special-value, and outside-range-sentinel cases.
- Evidence covers all applicable data types, ordered array/segment [carrier](../../../../glossary.md#term-storage-carrier) roles, [contiguous](../../../../glossary.md#term-contiguous)/admitted non-contiguous regimes, and selected four strategy modes. Each unavailable/inapplicable case is explicit; the 152 `rejected:pointwise` rows and 21 other-category rejected rows remain exact and [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).
- Paired entries pass active hygiene and a closed invocation/overhead allowlist. Tests reject forbidden mechanisms and an unused/dead allowlist entry, preventing dead comparator/allowlist evidence.
- A promoted structural disposition has an exact declared, non-tautological paired comparator with negative controls. It cannot use IDs, hashes, normalized-body keys, or catalog categories as proof. All other rows remain `PARTIAL_CLASSFILE_NO_CLEAN_JAVA_ORACLE`.
- Resources fail closed for stale [digest](../../../../glossary.md#term-digest), duplicate/orphan/missing [owner](../../../../glossary.md#term-owner), altered accounting, malformed [schema](../../../../glossary.md#term-schema), changed ABI/hash, or unknown disposition. Performance stays `PARTIAL_NO_REPRESENTATIVE_BENCHMARK` absent exact reusable row evidence.
- A separate documentation-focused pass records reasoned no-change conclusions for [API](../../../../glossary.md#term-application-programming-interface-api) Javadocs, guides, glossary, architecture/ADRs, architecture/[conformance](../../../../glossary.md#term-conformance)/integration tests, [Gradle](../../../../glossary.md#term-gradle), and production packages.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation context runs the [manifest](../../../../glossary.md#term-manifest), direct [pointwise](../../../../glossary.md#term-pointwise)/CAST [matrices](../../../../glossary.md#term-matrix-meanings), independent clean-Java matrix, paired structural/hygiene/negative controls, inventory [checkpoint](../../../../glossary.md#term-checkpoint), and unsupported boundaries:

```bash
./gradlew :backends:cpu:test --tests '*CpuPointwiseSemanticClosureManifestTest' --tests '*CpuPointwiseGeneratedKernelTest' --tests '*CpuCastGeneratedKernelTest' --tests '*CpuOrdinaryPointwiseCastStructuralOracleTest' --tests '*CpuGeneratedCoverageCheckpointTest' --tests '*CpuGeneratedCoverageUnsupportedBoundaryTest'
```

It validates headers/tabs/LF/final newline/digests, [ownership](../../../../glossary.md#term-ownership-meanings)/disposition joins, exact 665/180/845/173 counts, and `git diff --check`.

Documentation context reuses successful Java evidence unless executable Java changes; it validates Markdown links, anchors, fences, path [scope](../../../../glossary.md#term-scope), statuses/dependencies, terminology, whitespace, and:

```bash
git diff --check
```

Repository-wide/capability-checkpoint validation and performance closure defer to [CPU](../../../../glossary.md#term-cpu) 0009G/[CI](../../../../glossary.md#term-continuous-integration-ci). No five-[fork](../../../../glossary.md#term-fork-meanings) command is required.

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0009A and its documentation pass.
- Complete CPU 0008J, 0008K, and 0008L; current checked inventory, semantic [manifest](../../../../glossary.md#term-manifest), disposition [registry](../../../../glossary.md#term-registry), and evidence ledger.
- Java 26 Class-File, [Vector API](../../../../glossary.md#term-vector-api), and [Foreign Function and Memory API](../../../../glossary.md#application-binary-interface-and-native-bridge) test environment.

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0009C is the active [Ready](../../../../glossary.md#term-ready) child with its detailed bounded specification; it owns affine, movement, indexing, scatter, and random closure. CPU 0009D remains the next ordered Draft child.
- A row needing production/semantic work or a comparator beyond this [scope](../../../../glossary.md#term-scope) remains partial and becomes a narrow Draft follow-up only at the active frontier.
- CPU 0009G owns representative performance and final [checkpoint](../../../../glossary.md#term-checkpoint) closure.

## Architecture impact

Expected impact: None. Stop for architecture, [dependency](../../../../glossary.md#term-dependency), production-selection, [schema](../../../../glossary.md#term-schema), generator, capability, [admission](../../../../glossary.md#term-admission), or semantics changes.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU task 0009B. Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md, the Planning Guide, CPU master plan, CPU 0009, CPU 0009A, CPU 0008J/0008K/0008L, and this task. Implement exactly this bounded ordinary-pointwise/CAST test-evidence closure; do not change production behavior or claim general bytecode/CFG equivalence. Promote structural evidence only through an exact non-tautological paired clean-Java comparator; leave all other structural rows partial. Do not run or promote new five-fork performance evidence. Stop for scope or architecture conflict. Do not commit or push. After stable evidence, hand the diff and tests to a separate clean documentation-focused context under docs/developer-guide/documentation-rules.md. Update this task with results; do not mark Complete before that pass finishes.
```

## Local decisions

- [Ownership](../../../../glossary.md#term-ownership-meanings) is 845 generated rows, not the 6,009 catalog or the 1,101-[pointwise](../../../../glossary.md#term-pointwise)-[opcode](../../../../glossary.md#term-opcode) universe; the remaining 256 are [CPU](../../../../glossary.md#term-cpu) 0009A's [scalar](../../../../glossary.md#term-scalar)-immediate/clamp rows.
- Structural evidence is narrow paired observation. A bounded predecessor walk may support local hygiene but is not general [dataflow](../../../../glossary.md#term-dataflow) proof.
- Prior CAST/pointwise performance is not a shortcut; reuse requires exact row/[hash](../../../../glossary.md#term-hash)/counterpart/protocol/projection matching.

## Known limitations

No arbitrary-[bytecode](../../../../glossary.md#term-bytecode) equivalence, arbitrary immediate values, broader-family behavior, or complete generated/direct performance parity is proved. Semantic support without an exact paired comparator remains structurally partial. The 6,009 catalog is discovery/accounting support, never a closure denominator or equivalence proof.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context final evidence (reused; executable Java has not changed afterward):

- Focused six-suite command covering the [pointwise](../../../../glossary.md#term-pointwise) semantic [manifest](../../../../glossary.md#term-manifest), direct pointwise and CAST [matrices](../../../../glossary.md#term-matrix-meanings), ordinary paired structural [oracle](../../../../glossary.md#term-test-oracle), generated-coverage [checkpoint](../../../../glossary.md#term-checkpoint), and unsupported-boundary checks passed with 60 tests, 1 expected skip, and 0 failures/errors (`BUILD SUCCESSFUL` in 58s).
- Full [CPU](../../../../glossary.md#term-cpu) validation passed with 184 suites, 899 tests, 27 expected skips, and 0 failures/errors (`BUILD SUCCESSFUL` in 2m22s).
- The implementation's independent technical review found no HIGH findings and approved the result after typed [`MemorySegment`](../../../../glossary.md#term-memorysegment) access, real forward-branch direction, and rejection-accounting corrections.
- This documentation pass did not rerun Java/[Gradle](../../../../glossary.md#term-gradle) suites. The final `hasForwardSemanticBranch` to `hasForwardBranch` terminology change in the implementation diff changed no executable behavior, so duplicate suite execution is prohibited by the [planning](../../../../glossary.md#term-planning) rule.
- Documentation validation: reviewed exact final diff and the two new test-only Javadocs/comments; checked task/master-plan/roadmap status and [dependency](../../../../glossary.md#term-dependency) [synchronization](../../../../glossary.md#term-synchronization); checked Markdown links, anchors, fenced blocks, and terminology; searched affected documentation and Java comments for stale `173 rejected pointwise` and `semantic branch` wording; `git diff --check` passed.

No-change conclusions: `ARCHITECTURE.md`, architecture explanations, ADRs, and architecture tests remain accurate because this is test-evidence/documentation work with no architecture or dependency change. Public [API](../../../../glossary.md#term-application-programming-interface-api) Javadocs and the glossary need no update because no public terminology or behavior changed. [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) and integration tests need no update because no backend contract or end-to-end behavior changed. Gradle/build configuration and unrelated [modules](../../../../glossary.md#term-module) need no update because no build or module change occurred.

## Implementation notes

The test-only implementation supplies independent typed optimal clean-Java semantic execution for each of the 845 owned selected rows over their recorded ranges, including source-derived array-to-segment and segment-to-array [carrier](../../../../glossary.md#term-storage-carrier)-role witnesses and legal safe [aliases](../../../../glossary.md#aliasing-and-arena). Its bounded paired Class-File projection retains typed [ABI](../../../../glossary.md#term-application-binary-interface-abi); ordered carrier reads and writes; typed [`MemorySegment`](../../../../glossary.md#term-memorysegment) `ValueLayout`, width, and direction; conversions; numerical invokes; [vector](../../../../glossary.md#term-vector)-versus-[scalar](../../../../glossary.md#term-scalar) realization facts; and actual forward/backward branch direction. Negative controls reject carrier/[layout](../../../../glossary.md#layout)/order, store, conversion, invoke, and [topology](../../../../glossary.md#term-topology) drift.

The projection is deliberately bounded. It does not prove literal Class-File or [CFG](../../../../glossary.md#term-control-flow-graph-cfg) [identity](../../../../glossary.md#term-identity-meanings), [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) assembly identity, universal alias/overlap semantics, or generated/direct performance parity. Performance remains partial and [fail-closed](../../../../glossary.md#term-fail-closed); no five-[fork](../../../../glossary.md#term-fork-meanings) benchmark is claimed.

## Completion summary

- Completed changes: finalized the test-evidence closure record for 845 generated rows (665 ordinary [pointwise](../../../../glossary.md#term-pointwise) and 180 ordered CAST), corrected rejection accounting to 152 pointwise-category plus 21 other-category rows, and recorded the bounded structural proof, [validation](../../../../glossary.md#term-validation), review, and limitations.
- Files changed or created: this task; [CPU](../../../../glossary.md#term-cpu) master plan; roadmap; parent CPU 0009 task; and the implementation-owned test-only [oracle](../../../../glossary.md#term-test-oracle) and structural-oracle test sources.
- Tests and validation: reused focused six-suite evidence (60 tests, 1 expected skip, 0 failures/errors, 58s) and full CPU evidence (184 suites, 899 tests, 27 expected skips, 0 failures/errors, 2m22s); Markdown/static review and `git diff --check` passed.
- Documentation-agent review: completed in this clean context after the implementation evidence stabilized.
- Documentation impact: no architecture, guide, public [API](../../../../glossary.md#term-application-programming-interface-api), glossary, configuration, [conformance](../../../../glossary.md#term-conformance), integration, or unrelated-[module](../../../../glossary.md#term-module) update is required for this test-only closure.
- [Javadoc](../../../../glossary.md#term-javadoc) review: reviewed the two new test-only contracts; no documentation-context Java comment edit was needed because branch terminology is `forward branch`, and [vector](../../../../glossary.md#term-vector)/[scalar](../../../../glossary.md#term-scalar) and bounded-projection boundaries are explicit.
- Glossary impact: None; no reusable public term changed.
- Unresolved issues: None within CPU 0009B's bounded [scope](../../../../glossary.md#term-scope).
- Follow-up required: Implement the active [Ready](../../../../glossary.md#term-ready) CPU 0009C specification before creating the next ordered Draft child, CPU 0009D.

Status: Complete
