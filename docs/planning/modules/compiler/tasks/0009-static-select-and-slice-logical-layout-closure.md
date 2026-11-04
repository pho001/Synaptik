# Task 0009 Static SELECT and [SLICE](../../../../glossary.md#slice) logical [layout](../../../../glossary.md#layout) closure

> **Editorial reading key — not part of the historical record.**
>
> SELECT and SLICE close as logical views: their offsets and strides describe how later execution
> addresses existing logical data, not a newly dense copy. Compiler reuses family inference after
> preceding outputs resolve, preserving noncontiguous geometry. Providers and preparation still
> qualify actual physical bindings. The task's public CPU regressions establish their recorded
> execution slice; they do not widen cold reconstruction or silently migrate NN recurrent
> builders.
>
> Current reading: [exact graph transformations and final logical layouts](../../../../reference/compiler/optimization.md#current-package-private-canonicalization-and-whole-graph-optimization).

## Status

Complete — Feynman final Class C APPROVE, zero findings; main's combined native root gate passed on the unchanged reviewed [snapshot](../../../../glossary.md#term-snapshot) (2026-10-19).

## Change class and goal

Class C — final [compile](../../../../glossary.md#compile) [descriptors](../../../../glossary.md#term-descriptor) cross the [Planning](../../../../glossary.md#term-planning)/[Prepare](../../../../glossary.md#prepare) boundary. Close exact logical [views](../../../../glossary.md#term-view-meanings)
([shape](../../../../glossary.md#shape), [strides](../../../../glossary.md#term-stride) and [storage offset](../../../../glossary.md#term-storage-offset), not physical storage) for SELECT and positive-step [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
[SLICE](../../../../glossary.md#slice) after their inputs resolve, so existing recurrent cells work without forced [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [contiguous](../../../../glossary.md#term-contiguous) copies.

## [Scope](../../../../glossary.md#term-scope) and non-goals

Extend only `StaticResultLogicalLayoutClosure.closeAffineResult` [admission](../../../../glossary.md#term-admission)/routing: SELECT uses
`IndexingInference.infer`; [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [SLICE](../../../../glossary.md#slice) uses `LayoutInference.infer`, alongside existing crop
and [shape](../../../../glossary.md#shape)/[axis](../../../../glossary.md#term-axis) transforms. Reuse family [inference](../../../../glossary.md#term-inference); do not duplicate selection formulas or turn
LayoutInference into a generic dispatcher. Preserve [topological](../../../../glossary.md#term-topological-order) propagation and existing guards.
No new production/public type, abstraction, [module](../../../../glossary.md#term-module) edge, [kernel](../../../../glossary.md#kernel), [adjoint](../../../../glossary.md#term-adjoint), format, [backend route](../../../../glossary.md#backend-route),
[materialization](../../../../glossary.md#materialization) or [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) workaround. No [scan](../../../../glossary.md#term-cumulative-scan) [backpropagation through time](../../../../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../../../../glossary.md#backpropagation-through-time-and-recurrent-state)), dynamic [widening](../../../../glossary.md#term-widening),
formal-proof exercise, coverage campaign, native build or performance work. Cold source admission
and reconstruction remain unchanged, including their eight-kind [persistence](../../../../glossary.md#term-persistence) boundary.

## Contracts

Start with [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read root `Core lifecycle`, `Core invariants`, `Module-ownership routing`, `Dependency rules`,
`Explicit non-goals`, and `Testing requirements`; these preserve logical versus physical [ownership](../../../../glossary.md#term-ownership-meanings).
Read [Compiler contract](../../../../architecture/contracts/compiler-autograd.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span> headings
`modules/compiler`, [`Compile artifacts`](../../../../glossary.md#compile-artifacts), `Training graph model`, `Compiler-owned automatic differentiation`,
and `Compile lifecycle`; and [Engine contract](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
heading `Public reusable prepared execution` for the public integration [regression](../../../../glossary.md#term-regression).
Contract conclusion: existing exact family [inference](../../../../glossary.md#term-inference) and [compile](../../../../glossary.md#compile) [lifecycle](../../../../glossary.md#lifecycle) permit this repair;
no authority cutover or new physical [owner](../../../../glossary.md#term-owner) is needed. Preserve broad exclusions. Stop and report
an actual ambiguity or [scope](../../../../glossary.md#term-scope) expansion; do not edit contracts to bypass it.

## Dependencies and integration

- Depends on: Compiler0006B11 and Compiler0008, Complete in [Compiler master](../master-plan.md); [CPU0025](../../../backends/cpu/tasks/0025-scatter-validation-coordinate-rank.md) Complete after independent APPROVE, zero findings. NOT NN0029.
- Conflicts with: [Compiler](../../../../glossary.md#term-compiler) closure/[inference](../../../../glossary.md#term-inference) and scoped tests/docs; NN0029 implementation and shared [planning](../../../../glossary.md#term-planning) writers serialize. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) partial remains frozen.
- Parallel group: None; sole clean implementation writer, then independent final Class C review.
- Common base revision: `7e880fb1044168043e45b10419a570d64c5283ae` plus the full current dirty [snapshot](../../../../glossary.md#term-snapshot), including accepted CP0009, frozen NN0029 partial and completed CPU0025. HEAD alone is not the base; preserve all unrelated changes.
- Integration order: Compiler0009 implementation/[validation](../../../../glossary.md#term-validation)/final independent review → main accepts repair → NN0029 [resume](../../../../glossary.md#schema-serialization-and-checkpoint)/completion/review → single shared root gate.
- Integration validation: main's single native-enabled root command in [NN0029](../../../extensions/nn/tasks/0029-fluent-model-builder-and-checkpoint-independent-execution.md#validation-and-documentation) passed at the reviewed join; no duplicate worker root [run](../../../../glossary.md#term-run).
- Shared-document integration [owner](../../../../glossary.md#term-owner): main; workers do not edit masters/roadmap/NN brief. Main may serially delegate the targeted Compiler explanations below.
- Launch verification (2026-10-19): main read the actual [Ready](../../../../glossary.md#term-ready) brief and plan updates and verified user approval, Ready/frontier, Complete predecessors/current [CPU](../../../../glossary.md#term-cpu) final review, current HEAD/full dirty base, exact contracts permitting existing family inference, frozen NN and no other writer/conflict. Main's snapshot check found only the expected five planning-file changes; native [hash](../../../../glossary.md#term-hash) unchanged. Main accepts this bounded packet; no authority cutover, extra planning review or agent launch by this context.

## Files and symbols

[Compiler](../../../../glossary.md#term-compiler) package paths below use `io/github/pho001/synaptik/compiler/`:

- Production: `modules/compiler/src/main/java/` + package + `StaticResultLogicalLayoutClosure.java`, especially `closeAffineResult` and affected [Javadoc](../../../../glossary.md#term-javadoc).
- Read-only [inference](../../../../glossary.md#term-inference) [owners](../../../../glossary.md#term-owner) at the same path: `IndexingInference.java` (`infer`/`select`), `LayoutInference.java` (`infer`/[`slice`](../../../../glossary.md#slice)); `CapturedGraphInference.java` and `GraphCompiler.java` for [validation](../../../../glossary.md#term-validation)/order. Do not broaden these owners.
- Tests: `modules/compiler/src/test/java/` + package + `StaticResultLogicalLayoutClosureTest.java`; reuse `LayoutInferenceTest.java`, `IndexingInferenceTest.java`, and `GraphCompilerTest.java` as focused reference/coverage, extending only if needed for this repair.
- Add `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/CompilerSelectSliceLayoutIntegrationTest.java`; use public [Engine](../../../../glossary.md#term-engine)/[Tensor](../../../../glossary.md#tensor) APIs, no [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) imports or new build [dependency](../../../../glossary.md#term-dependency). Existing `EngineNumericalSemanticsIntegrationTest.java` in that directory supplies public backward/[materialization](../../../../glossary.md#materialization) patterns.

## Acceptance criteria

1. Replace the intentional unresolved assertions in `derivesAffineViewFamiliesFromResolvedInputsAndLeavesOtherViewsUnresolved`: existing [SLICE](../../../../glossary.md#slice) [fixture](../../../../glossary.md#term-fixture) becomes `[2,2]`, [strides](../../../../glossary.md#term-stride) `[3,1]`, offset `5`, [view](../../../../glossary.md#term-view-meanings)=true; SELECT becomes `[3]`, stride `[1]`, offset `8`, view=true. Rename the test to match its new intent.
2. Cover materialized MATMUL/CONCAT → extraction → downstream view propagation in the same pass.
   NN fixtures: RNN SELECT <span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>`[3,2,2]` / strides `[4,2,1]` at [axis](../../../../glossary.md#term-axis) 0 index 2 gives `[2,2]` / `[2,1]` / offset `8`;
   GRU projection `[2,6]` and [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) projection `[2,8]` sliced to `[2,2]` retain strides `[6,1]` and `[8,1]` respectively. Include a nonzero gate start and verify its exact offset.
3. Assert exact noncontiguous and nonzero-offset geometry, positive nonunit steps, and view flags;
   never replace an inferred view with canonical geometry. Negative-step and empty extraction remain
   unresolved under current inference, as do unknown/dynamic or otherwise unproved layouts. No new signed-stride or empty-layout support.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [inference glossary](../../../../glossary.md#term-inference))</span>
4. Preserve [dtype](../../../../glossary.md#data-type--datatype), [shape](../../../../glossary.md#shape) and requiresGrad; accept a derived [layout](../../../../glossary.md#layout) only when inferred output dtype/shape agree with the captured output. Keep resolved [descriptors](../../../../glossary.md#term-descriptor) and caller-bindable [identities](../../../../glossary.md#term-identity-meanings) untouched. Preserve value/boundary order, IDs, exact [node](../../../../glossary.md#node)/[operation](../../../../glossary.md#operation) references, phases, [constants](../../../../glossary.md#term-constant), constraints and [derivative](../../../../glossary.md#term-derivative) sidecars. Assert exact input [identity](../../../../glossary.md#term-identity-meanings) when no descriptor changes; retain checked offset/stride arithmetic and contextual failure behavior.
5. Small public [Engine](../../../../glossary.md#term-engine) [CPU](../../../../glossary.md#term-cpu) integration cases exercise both closed SELECT and SLICE through [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare)/[run](../../../../glossary.md#term-run), including reuse, plus [scalar](../../../../glossary.md#term-scalar)-[objective](../../../../glossary.md#term-objective) backward through the extracted values. Compare values and nonzero [gradients](../../../../glossary.md#term-gradient) to independent small-array arithmetic; no forced [contiguous](../../../../glossary.md#term-contiguous)/copies, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [dependency](../../../../glossary.md#term-dependency), new [adjoint](../../../../glossary.md#term-adjoint) or [backend](../../../../glossary.md#backend) change. NN0029 later owns actual [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/LSTM builder acceptance and valid static-[unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) training, not this worker.

## [Validation](../../../../glossary.md#term-validation)

Use focused tests during development, then one final [Compiler](../../../../glossary.md#term-compiler) [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) [run](../../../../glossary.md#term-run). [Gradle](../../../../glossary.md#term-gradle) [identities](../../../../glossary.md#term-identity-meanings)
below are verified in `settings.gradle.kts`; separate filtered test invocation from [compile](../../../../glossary.md#compile) tasks:

```bash
./gradlew :modules:compiler:test --tests '*StaticResultLogicalLayoutClosureTest' --tests '*LayoutInferenceTest' --tests '*IndexingInferenceTest' --max-workers=2 --console=plain
./gradlew :modules:compiler:test :modules:compiler:javadoc --max-workers=2 --console=plain
./gradlew :testing:integration-tests:compileTestJava --max-workers=2 --console=plain
./gradlew :testing:integration-tests:test --tests '*CompilerSelectSliceLayoutIntegrationTest' --max-workers=2 --console=plain
git diff --check
```

Main's single native-enabled root gate passed at the combined [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) join. No worker root/native
repeat. Review reuses successful source-matched executable evidence; final Javadoc follows final edits.
No [dependency](../../../../glossary.md#term-dependency)/[backend](../../../../glossary.md#backend) change is planned; existing architecture/[conformance](../../../../glossary.md#term-conformance) checks belong to that gate.

## Documentation and review impact

Clean implementation and one independent final Class C code/[Javadoc](../../../../glossary.md#term-javadoc)/documentation review required.
Update closure Javadoc to describe family routing and exact-[view](../../../../glossary.md#term-view-meanings) limits. Review the static-result
closure paragraph in `docs/api/compile-api.md` and relevant `docs/architecture/lifecycle.md` text;
update affected explanations in the same overall change, under main's serial [ownership](../../../../glossary.md#term-ownership-meanings). Search
glossary [`Layout`](../../../../glossary.md#layout), [`Scalar select`](../../../../glossary.md#scalar-select), [`Slice`](../../../../glossary.md#slice); record a reasoned no-change conclusion if still accurate.
Apply documentation-rules and general/[API](../../../../glossary.md#term-application-programming-interface-api)/architecture profiles as applicable. No contract/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) edit,
new terminology or extra review campaign. Check links/anchors/whitespace; no stage/[commit](../../../../glossary.md#term-commit)/push.

## Result

Status: Complete

Extended only existing closure [admission](../../../../glossary.md#term-admission)/routing: SELECT delegates to IndexingInference;
[coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [SLICE](../../../../glossary.md#slice) delegates to LayoutInference alongside existing crop and [shape](../../../../glossary.md#shape)/[axis](../../../../glossary.md#term-axis) [views](../../../../glossary.md#term-view-meanings).
No formulas, production types, physical decisions, public [API](../../../../glossary.md#term-application-programming-interface-api), contracts, [backend](../../../../glossary.md#backend)/[ABI](../../../../glossary.md#term-application-binary-interface-abi), [codec](../../../../glossary.md#term-codec),
build configuration or [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) source changed. Exact matching geometry, captured [gradient](../../../../glossary.md#term-gradient) eligibility,
resolved/caller [descriptors](../../../../glossary.md#term-descriptor), IDs/order/references/phases and sidecars are preserved.

Changed files: production closure and its scoped test; new public [CPU](../../../../glossary.md#term-cpu)-only
`CompilerSelectSliceLayoutIntegrationTest`; targeted [compile](../../../../glossary.md#compile)-api/[lifecycle](../../../../glossary.md#lifecycle) explanations;
the [Scalar select](../../../../glossary.md#scalar-select) glossary paragraph; this brief's Status/Result and authorized six→five correction.
Tests cover same-pass [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/CONCAT extraction and downstream views, [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) geometry,
nonunit/zero [strides](../../../../glossary.md#term-stride) and offsets, negative/empty/unknown/dynamic exclusions, descriptor guards,
[metadata](../../../../glossary.md#term-metadata) preservation, unchanged-input [identity](../../../../glossary.md#term-identity-meanings), and contextual checked stride [overflow](../../../../glossary.md#term-overflow).
Integration uses no NN imports or forced [contiguous](../../../../glossary.md#term-contiguous) [operation](../../../../glossary.md#operation), reuses preparations, and compares
values plus both nonzero input gradients against independent array arithmetic.

[Validation](../../../../glossary.md#term-validation) (2026-10-19; exact commands):

- `./gradlew :modules:compiler:test --tests '*StaticResultLogicalLayoutClosureTest' --tests '*LayoutInferenceTest' --tests '*IndexingInferenceTest' --max-workers=2 --console=plain`: RED 19 tests, 6 expected failures before production edits; GREEN 19 tests, no failures/skips after repair.
- `./gradlew :modules:compiler:test :modules:compiler:javadoc --max-workers=2 --console=plain`: PASS, 42 suites / 315 tests, zero failures/errors/skips; final [Javadoc](../../../../glossary.md#term-javadoc) after frozen edits, no Javadoc warnings. Initial sandbox attempt denied existing [Gradle](../../../../glossary.md#term-gradle) wrapper [cache](../../../../glossary.md#term-cache) lock; rerun with proper require_escalated approval succeeded.
- `./gradlew :testing:integration-tests:compileTestJava --max-workers=2 --console=plain`: PASS before red integration; final frozen-source check PASS, all 19 tasks up to date, using approved Gradle cache access.
- `./gradlew :testing:integration-tests:test --tests '*CompilerSelectSliceLayoutIntegrationTest' --max-workers=2 --console=plain`: RED 2 tests, 2 failures at SELECT/SLICE [ownership](../../../../glossary.md#term-ownership-meanings) selection before repair; GREEN 2 tests, zero failures/errors/skips after repair. Successful source-matched evidence reused; no duplicate green [run](../../../../glossary.md#term-run). Existing incubator-[vector](../../../../glossary.md#term-vector) and restricted [FFM](../../../../glossary.md#term-ffm) [OpenBLAS](../../../../glossary.md#term-openblas) [lookup](../../../../glossary.md#term-lookup) warnings only.
- `git diff --check`: PASS; targeted local Markdown [targets](../../../../glossary.md#term-target)/anchors (8) and new-file trailing whitespace checks PASS. Protected THIRD_PARTY_NOTICES and CPU clean-Java [oracle](../../../../glossary.md#term-test-oracle) [hashes](../../../../glossary.md#term-hash) match launch values. Worker performed no native build/library replacement, root or benchmark run; main's later join is recorded below.

Documentation impact: updated closure Javadoc (family routing, proof limits, [parameters](../../../../glossary.md#term-parameter)/result and
contextual overflow), compile-api and lifecycle explanations under general/API/architecture profiles.
Targeted glossary [Layout](../../../../glossary.md#layout)/Scalar select/Slice review found stale planned-execution wording in Scalar
select; main authorized that paragraph alone to reflect the existing bounded static resolved CPU
domain and exact [Compiler](../../../../glossary.md#term-compiler) view closure. Layout/Slice meanings remain accurate; no new terminology
or glossary-wide rewrite. Existing CP0009/NN/CPU0025 dirty changes preserved.

Frozen SHA256 (source/tests/explanations; brief finalized separately for [handoff](../../../../glossary.md#term-handoff)):

| Path | SHA256 |
|---|---|
| `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StaticResultLogicalLayoutClosure.java` | `83be096d4e23e9ad381c0c230c0e0d44fcf2c14be39b2f4e4298dbd55b3f25bb` |
| `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/StaticResultLogicalLayoutClosureTest.java` | `4d3affe377cca15c3ad01084cc49316cb5733e3fc971301ea629c039921f16f5` |
| `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/CompilerSelectSliceLayoutIntegrationTest.java` | `b2a1d8b63e9415101324a210ead21f403241296c1e3389b031a52e2d01ae8220` |
| `docs/api/compile-api.md` | `0f8426685156434764d116b1cbabd13442d95c8d000aee66b8bd77f88dcdaa7b` |
| `docs/architecture/lifecycle.md` | `012564dbdd08243092dd609891a0c7b008ba546d8649f7aff66a3b262552e2e0` |
| `docs/glossary.md` | `b721e66641776da0b2534551a4a65b966a8d06c99be3baf16fc359b194af094b` |

Final independent code/Javadoc/documentation review: Feynman APPROVE, zero findings (2026-10-19);
all seven frozen hashes matched. Review reused the successful 315-test Compiler/Javadoc and
two-test integration evidence above. Main accepted the local repair after confirming the [snapshot](../../../../glossary.md#term-snapshot)
unchanged. No unresolved Compiler issue or follow-up; NN0029 is Complete, frontier None.
Main's exact NN0029 native root command passed: exit 0, BUILD SUCCESSFUL, 5m14s; 107 tasks,
18 executed/89 UP-TO-DATE. Current XML: 4279 tests, zero failures/errors, 41 skips; includes
UP-TO-DATE reports, not all newly run. Reviewed 2291-file snapshot/native [hash](../../../../glossary.md#term-hash) unchanged;
both configured [Metal](../../../../glossary.md#term-metal) training cases executed without failure/skip. The shared join is complete.
