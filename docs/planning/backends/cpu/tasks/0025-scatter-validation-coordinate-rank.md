# Task CPU0025 Scatter validation coordinate rank repair

## Status

Complete — independent Class C APPROVE, zero findings, and main's combined native root gate passed (2026-10-19).

## Change class and goal

Class C — memory-safe addressing and allocation-free validation before target writes; existing private correctness repair, not capability, math, architecture, ABI, adjoints or formats.

## Scope and evidence

`CpuPreparedExecutable.advanceAddress` (line 1942) loops over `coordinates.length`.
`ScatterValidation` (2571) allocates scratch at max(index rank, update rank); `bounds` (2597)
passes index-rank extents/strides with that oversized scratch, causing ArrayIndexOutOfBoundsException.
Legal GATHER adjoint SCATTER_ADD has hidden rank 2, indices rank 1, updates rank 2.
Frozen NN reproduction: `Model.builder().input(DataType.FLOAT32,3,1,2).seed(29).rnn(2).build()` then configure/train SGD.
`/private/tmp/nn0029-blocker-handoff.txt` records exact inputs and 17 frozen tests: 13 pass, 4 fail. CPU0025 owns only the scatter failure, not SELECT/SLICE diagnosis.
Use active geometry rank for addressing; inspect every helper caller and scratch reuse.
No NN, Compiler, kernel/emitter, capability, build, public-type, native or contract changes.

## Contracts

- [Root](../../../../../ARCHITECTURE.md): `Scope-indexed normative contracts`, `Core invariants`, `Testing requirements`.
- [Backend execution](../../../../architecture/contracts/backend-execution.md): `Concrete backend modules`, `CPU backend routes`.
- [Foundational modules](../../../../architecture/contracts/foundational-modules.md): `modules/model`.
- Existing Model semantics: [Gather-compatible Scatter Add expressions](../../../../api/tensor-api.md#gather-compatible-scatter-add-expressions); no semantic change.
Stop on an actual contract ambiguity or required scope expansion; do not invent architecture.

## Dependencies and integration

- Depends on: CPU0024 and existing scatter foundation CPU0006B1/0007A0C/0007A1B, all Complete in CPU master.
- Conflicts with: CPU executable/scatter tests and shared planning writers; NN0029 writes remain frozen.
- Parallel group: None; sole serial implementation writer, then independent targeted review.
- Common base revision: `7e880fb1044168043e45b10419a570d64c5283ae` + accepted uncommitted CP0009 + frozen NN0029 partial; preserve all.
- Integration order: CPU0025 implementation/tests/review accepted → NN0029 layout decision/resume/review → shared main gate. CPU0025 has no NN dependency.
- Integration validation: main's shared NN0029 native root gate passed on the unchanged reviewed snapshot.
- Shared-document integration owner: main; workers do not edit master/roadmap or NN brief.
- Launch verification (2026-10-19): main rechecked Ready/frontier, unchanged HEAD/frozen source, Complete predecessors, exact contracts and inactive conflicts. CPU0025 is sole CPU writer; NN remains frozen. No dependency cycle.

## Files and symbols

- `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`: advanceAddress, IndexValidation caller, ScatterValidation bounds/duplicates/scratch.
- `modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/`: CpuScatterGeneratedKernelTest, CpuScatterSemanticClosureTest, CpuIndexingGeneratedKernelTest.
- `modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScatterLoweringTest.java`: existing admitted geometry, read-only unless regression needs extension.
- Add `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuScatterValidationConformanceTest.java` for direct prepared CPU regression, without NN dependency.

## Acceptance criteria

- Direct SCATTER_ADD succeeds for index rank < update rank, INT32 and INT64, including admitted strided/offset indices; compare actual values to an independent direct oracle.
- Late negative and upper-bound-invalid indices fail with intended bounds diagnostics; output/target sentinel bytes remain unchanged, including earlier valid indices.
- Full bounds scan precedes writes; preserve duplicate-index accumulation, ScatterElements/ND uniqueness and nearby indexing behavior, repeated invocation resets and scalar/empty cases already admitted.
- Validation remains allocation-free per invocation/element; no generated body, hot-loop, capability or identity change. Review all advanceAddress callers, not just the failing line.
- Meaningful helper Javadoc states active-rank versus scratch-capacity invariants, mutation/ownership, inputs, result and failures; inspect final source and tests independently.

## Validation

Focused development, then one final affected-module/Javadoc/conformance run after executable edits stabilize:

```bash
./gradlew :backends:cpu:test --tests '*CpuScatter*Test' --tests '*CpuIndexingGeneratedKernelTest' --max-workers=2 --console=plain
./gradlew :backends:cpu:test :backends:cpu:javadoc --max-workers=2 --console=plain
./gradlew :testing:backend-conformance:test --tests '*CpuScatterValidationConformanceTest' --max-workers=2 --console=plain
git diff --check
```

Main ran NN0029's recorded root command once at the shared join; outcome is recorded below.
No worker full-repository repeat, benchmark campaign or native build; reviewer reuses valid test evidence.

## Documentation and review impact

Clean implementation context and independent targeted Class C code/Javadoc/documentation review required.
Review affected API explanations and glossary Scatter Add/Scatter Elements/Scatter-ND; record
`Documentation impact: none; existing contract unchanged by private correctness repair` if confirmed.
No contract/ADR edits. Preserve THIRD_PARTY_NOTICES.md SHA-256 `8efe7e3c1f83bd5b30b913d447fa3ee95f9c0ee60485c30de7c0008638e7037f`
and CpuOrdinaryPointwiseCastCleanJavaOracle.java SHA-256 `57ce4dc701a83d2ed01b96c725197a72f5aa752fe5e200f3a354f9d273f8f842`; no Git mutations.

## Result

- `CpuPreparedExecutable.advanceAddress` now uses active `extents.length`, not scratch capacity;
  helper Javadoc documents geometry, scratch ownership/mutation, result and failure invariants.
  Both callers and scratch resets were audited; validation still precedes writes, without allocation
  or generated-body changes. Added `CpuScatterValidationTest` and `CpuScatterValidationConformanceTest`.
- Reused frozen-source evidence (2026-10-19), exact commands in Validation above: focused 37 passed;
  full CPU 1,079 tests = 1,049 passed + 30 assumption skips, zero failures/errors; Javadoc passed
  with 94 warnings in unchanged code; new conformance 2 passed, zero failures/errors/skips.
- Independent targeted Class C code/Javadoc/documentation review: APPROVE, zero findings (2026-10-19).
  Local prerequisite accepted; no unresolved CPU implementation issue and no NN dependency.
- Documentation impact: existing API explanations, glossary and other Javadoc remain accurate;
  private correctness repair changes no contract, terminology, architecture or ADR. Status records
  synchronized separately; successful executable/Javadoc evidence reused without reruns.
- All three frozen CPU source/test hashes and both protected hashes match the supplied handoff.
  No code or Git writes in status integration; accepted dirty CP0009 and frozen NN0029 preserved.
- Main's exact [NN0029 root command](../../../extensions/nn/tasks/0029-fluent-model-builder-and-checkpoint-independent-execution.md#validation-and-documentation) passed: exit 0, 5m14s; 107 tasks, 18 executed/89 UP-TO-DATE.
  Current XML: 4279 tests, zero failures/errors, 41 skips; includes UP-TO-DATE evidence, not all newly run.
  Reviewed 2291-file snapshot/native hash unchanged; both configured Metal training cases executed
  without failure/skip. Compiler0009 resolved layout closure; NN0029 is Complete, frontier None.

Status: Complete
