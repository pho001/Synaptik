# Task CPU0025 Scatter [validation](../../../../glossary.md#term-validation) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [rank](../../../../glossary.md#term-rank) repair

## Reader context — historical integration boundary

This failure concerns validator traversal state, not scatter's numerical kernel. Indices and updates
can have different logical ranks, while one reusable coordinate array is large enough for both.
CPU's addressing helper must use the currently selected extents/strides to decide how many
coordinates participate; scratch capacity is only storage capacity, not active geometry. The
consumer's complete index validation still runs after producers and before scatter writes.
Compiler-generated gradients are real framework callers of this path: a Gather adjoint can supply
exactly the rank combination that exposed the defect without an application directly invoking a
validator.

Current reading: [Scatter validation and update domains](../../../../backend-guide/cpu-backend.md#current-functional-scatter-family); [Metadata binding versus produced-value checks](../../../../backend-guide/cpu-backend.md#binding-metadata-and-validating-produced-values).

## Status

Complete — independent Class C APPROVE, zero findings, and main's combined native root gate passed (2026-10-19).

## Change class and goal

Class C — memory-safe addressing and allocation-free [validation](../../../../glossary.md#term-validation) before [target](../../../../glossary.md#term-target) writes; existing private correctness repair, not capability, math, architecture, [ABI](../../../../glossary.md#term-application-binary-interface-abi), adjoints or formats.

## [Scope](../../../../glossary.md#term-scope) and evidence

`CpuPreparedExecutable.advanceAddress` (line 1942) loops over `coordinates.length`.
`ScatterValidation` (2571) allocates scratch at max(index [rank](../../../../glossary.md#term-rank), update rank); `bounds` (2597)
passes index-rank extents/[strides](../../../../glossary.md#term-stride) with that oversized scratch, causing ArrayIndexOutOfBoundsException.
Legal [GATHER](../../../../glossary.md#gather) [adjoint](../../../../glossary.md#term-adjoint) SCATTER_ADD has hidden rank 2, indices rank 1, updates rank 2.
Frozen [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) reproduction: `Model.builder().input(DataType.FLOAT32,3,1,2).seed(29).rnn(2).build()` then configure/train [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd).
`/private/tmp/nn0029-blocker-handoff.txt` records exact inputs and 17 frozen tests: 13 pass, 4 fail. CPU0025 owns only the scatter failure, not SELECT/[SLICE](../../../../glossary.md#slice) diagnosis.
Use active geometry rank for addressing; inspect every helper caller and scratch reuse.
No NN, [Compiler](../../../../glossary.md#term-compiler), [kernel](../../../../glossary.md#kernel)/emitter, capability, build, public-type, native or contract changes.

## Contracts

- [Root](../../../../../ARCHITECTURE.md): `Scope-indexed normative contracts`, `Core invariants`, `Testing requirements`.
- [Backend execution](../../../../architecture/contracts/backend-execution.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>: `Concrete backend modules`, `CPU backend routes`.
- [Foundational modules](../../../../architecture/contracts/foundational-modules.md)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>: `modules/model`.
- Existing [Model](../../../../glossary.md#term-model) semantics: [Gather-compatible Scatter Add expressions](../../../../api/tensor-api.md#gather-compatible-scatter-add-expressions)<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [Scatter Add glossary](../../../../glossary.md#scatter-add))</span>; no semantic change.
Stop on an actual contract ambiguity or required [scope](../../../../glossary.md#term-scope) expansion; do not invent architecture.

## Dependencies and integration

- Depends on: CPU0024 and existing scatter foundation CPU0006B1/0007A0C/0007A1B, all Complete in [CPU](../../../../glossary.md#term-cpu) master.
- Conflicts with: CPU executable/scatter tests and shared [planning](../../../../glossary.md#term-planning) writers; NN0029 writes remain frozen.
- Parallel group: None; sole serial implementation writer, then independent targeted review.
- Common base revision: `7e880fb1044168043e45b10419a570d64c5283ae` + accepted uncommitted CP0009 + frozen NN0029 partial; preserve all.
- Integration order: CPU0025 implementation/tests/review accepted → NN0029 [layout](../../../../glossary.md#layout) decision/resume/review → shared main gate. CPU0025 has no [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [dependency](../../../../glossary.md#term-dependency).
- Integration [validation](../../../../glossary.md#term-validation): main's shared NN0029 native root gate passed on the unchanged reviewed [snapshot](../../../../glossary.md#term-snapshot).
- Shared-document integration [owner](../../../../glossary.md#term-owner): main; workers do not edit master/roadmap or NN brief.
- Launch verification (2026-10-19): main rechecked [Ready](../../../../glossary.md#term-ready)/frontier, unchanged HEAD/frozen source, Complete predecessors, exact contracts and inactive conflicts. CPU0025 is sole CPU writer; NN remains frozen. No dependency cycle.

## Files and symbols

- `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`: advanceAddress, IndexValidation caller, ScatterValidation bounds/duplicates/scratch.
- `modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/`: CpuScatterGeneratedKernelTest, CpuScatterSemanticClosureTest, CpuIndexingGeneratedKernelTest.
- `modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScatterLoweringTest.java`: existing admitted geometry, read-only unless [regression](../../../../glossary.md#term-regression) needs extension.
- Add `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuScatterValidationConformanceTest.java` for direct prepared [CPU](../../../../glossary.md#term-cpu) regression, without [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [dependency](../../../../glossary.md#term-dependency).

## Acceptance criteria

- Direct SCATTER_ADD succeeds for index [rank](../../../../glossary.md#term-rank) < update rank, [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype), including admitted strided/offset indices; compare actual values to an independent direct [oracle](../../../../glossary.md#term-test-oracle).
- Late negative and upper-bound-invalid indices fail with intended bounds diagnostics; output/[target](../../../../glossary.md#term-target) sentinel bytes remain unchanged, including earlier valid indices.
- Full bounds scan precedes writes; preserve duplicate-index accumulation, ScatterElements/[ND](../../../../glossary.md#term-nd) uniqueness and nearby indexing behavior, repeated invocation resets and [scalar](../../../../glossary.md#term-scalar)/empty cases already admitted.
- [Validation](../../../../glossary.md#term-validation) remains allocation-free per invocation/[element](../../../../glossary.md#term-element); no generated body, hot-loop, capability or [identity](../../../../glossary.md#term-identity-meanings) change. Review all advanceAddress callers, not just the failing line.
- Meaningful helper [Javadoc](../../../../glossary.md#term-javadoc) states active-rank versus scratch-capacity invariants, mutation/[ownership](../../../../glossary.md#term-ownership-meanings), inputs, result and failures; inspect final source and tests independently.

## [Validation](../../../../glossary.md#term-validation)

Focused development, then one final affected-[module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc)/[conformance](../../../../glossary.md#term-conformance) [run](../../../../glossary.md#term-run) after executable edits stabilize:

```bash
./gradlew :backends:cpu:test --tests '*CpuScatter*Test' --tests '*CpuIndexingGeneratedKernelTest' --max-workers=2 --console=plain
./gradlew :backends:cpu:test :backends:cpu:javadoc --max-workers=2 --console=plain
./gradlew :testing:backend-conformance:test --tests '*CpuScatterValidationConformanceTest' --max-workers=2 --console=plain
git diff --check
```

Main ran NN0029's recorded root command once at the shared join; outcome is recorded below.
No worker full-repository repeat, benchmark campaign or native build; reviewer reuses valid test evidence.

## Documentation and review impact

Clean implementation context and independent targeted Class C code/[Javadoc](../../../../glossary.md#term-javadoc)/documentation review required.
Review affected [API](../../../../glossary.md#term-application-programming-interface-api) explanations and glossary [Scatter Add](../../../../glossary.md#scatter-add)/[Scatter Elements](../../../../glossary.md#scatter-elements)/[Scatter-ND](../../../../glossary.md#scatter-nd); record
`Documentation impact: none; existing contract unchanged by private correctness repair` if confirmed.
No contract/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) edits. Preserve THIRD_PARTY_NOTICES.md [SHA-256](../../../../glossary.md#term-sha-256) `8efe7e3c1f83bd5b30b913d447fa3ee95f9c0ee60485c30de7c0008638e7037f`
and CpuOrdinaryPointwiseCastCleanJavaOracle.java SHA-256 `57ce4dc701a83d2ed01b96c725197a72f5aa752fe5e200f3a354f9d273f8f842`; no Git mutations.

## Result

- `CpuPreparedExecutable.advanceAddress` now uses active `extents.length`, not scratch capacity;
  helper Javadoc documents geometry, scratch ownership/mutation, result and failure invariants.
  Both callers and scratch resets were audited; validation still precedes writes, without allocation
  or generated-body changes. Added <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>`CpuScatterValidationTest` and `CpuScatterValidationConformanceTest`.
- Reused frozen-source evidence (2026-10-19), exact commands in Validation above: focused 37 passed;
  full CPU 1,079 tests = 1,049 passed + 30 assumption skips, zero failures/errors; Javadoc passed
  with 94 warnings in unchanged code; new conformance 2 passed, zero failures/errors/skips.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Independent targeted Class C code/Javadoc/documentation review: APPROVE, zero findings (2026-10-19).
  Local prerequisite accepted; no unresolved CPU implementation issue and no NN dependency.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Documentation impact: existing API explanations, glossary and other Javadoc remain accurate;
  private correctness repair changes no contract, terminology, architecture or ADR. Status records
  synchronized separately; successful executable/Javadoc evidence reused without reruns.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- All three frozen CPU source/test hashes and both protected hashes match the supplied handoff.
  No code or Git writes in status integration; accepted dirty CP0009 and frozen NN0029 preserved.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- Main's exact [NN0029 root command](../../../extensions/nn/tasks/0029-fluent-model-builder-and-checkpoint-independent-execution.md#validation-and-documentation) passed: exit 0, 5m14s; 107 tasks, 18 executed/89 UP-TO-DATE.
  Current XML: 4279 tests, zero failures/errors, 41 skips; includes UP-TO-DATE evidence, not all newly run.
  Reviewed 2291-file snapshot/native hash unchanged; both configured Metal training cases executed
  without failure/skip. Compiler0009 resolved layout closure; NN0029 is Complete, frontier None.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [snapshot glossary](../../../../glossary.md#term-snapshot); [hash glossary](../../../../glossary.md#term-hash); [Metal glossary](../../../../glossary.md#term-metal); [layout glossary](../../../../glossary.md#layout))</span>

Status: Complete
