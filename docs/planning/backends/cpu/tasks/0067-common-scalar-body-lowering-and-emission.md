# CPU0067: Common scalar body lowering and emission

> Superseded by CPU0070's [replacement plan](../existing-emitter-composition-plan.md) and
> [CPU0071 foundation](0071-common-matmul-binding-foundation.md). The original Draft below is
> archived, not executable. Its four named types and diagnostic budget are not carried forward.

## Status and change class

Superseded. The following pre-supersession status and remaining brief are archived:

Draft; not a source frontier. Requires reviewed/MAIN-accepted CPU0066, verified dependencies,
**user** approval of all four proposed named types (including nested Body/Accumulation) and the
exact diagnostic budget. Private generation-only types are not exempt from the scoped stop.
No launch/build/measurement authorized; extraction alone needs no new root/ADR rule.
Class C: generated-code shape, exact arithmetic/resource requirements and temporal binding.

## Goal and bounded scope

Deliver one common scalar math path for ordinary and existing captured Scan bodies, including
typed lowering, emission, tests and documentation. Scan retains traversal and temporal bindings,
not its own pointwise/MATMUL/SUM algorithms or a private optimizer behind a shared facade.
Follow the [proposal's call flow](../shared-body-emission-proposal.md#concrete-common-call-flow-and-proposed-types).
Preserve static all-valid F32/F64, symbolic all-valid F32 and static/symbolic length F32/F64
forward/actual first-order reverse, with current layout/read/binder exclusions.

Non-goals: capability expansion, vector/parallel Scan, uniform reduction numerics, ordinary
symbolic F64 MATMUL/epilogues, Compiler-body rewrites, Model/API/persistence/Runtime/build changes,
NN packing/migration, native/low Scan, tuning or unrelated CPU0048–0052 repairs.

## Contracts and pre-code gate

Read AGENTS; root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
[scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts); backend
[Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes) and
[Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract);
Scan [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
[Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic)
and [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories);
Runtime/Prepare [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).

Before Java, MAIN verifies a Ready packet, stable common seams and a changed-artifact policy.
No root/ADR change is required merely to factor private codegen classes/methods. Preserve existing
byte obligations where possible; if affected, enumerate changed classes, advance both identity
streams and coordinate only affected root/backend version/retention facts before conflicting
generation. ADR only for a significant actual decision. Scan
[Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates)
requires user approval of new named types; MAIN cannot self-approve the four-type packet below.
Any numerical/ownership/Runtime ambiguity stops execution. This Draft authorizes no authority edit.

## Dependencies and integration

- Depends on: CPU0066 accepted; user-approved named-type/diagnostic packet; CPU0064/0065 Complete;
  retained Compiler0015/0016 and Prepare0015.
- Conflicts with: CPU common-body, Scan, MATMUL/aggregate, cache, workers and CPU0048–0052 writers.
- Parallel group: None; clean sole implementation writer, independent read-only review.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty tree;
  MAIN must record a fresh preservation snapshot/contract set before Ready, not assume a merge.
- Integration order: reviewed seams/any necessary compatibility updates → scalar migration → independent
  review/fixes → MAIN checkpoint/acceptance → consider CPU0068. No automatic launch.
- Integration validation: CPU0067 checkpoint below, scoped docs/links/whitespace/preservation.
- Shared-document integration owner: MAIN; delegate owned explanation/Javadoc exclusively.

## Files and symbols

Source prefix: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`.

- `lowering/CpuPartitionLowering`, `CpuMatmulLowering`, `CpuAggregateLowering`: extract shared
  facts-only instruction/geometry/axis validators; `CpuGeneralScanLowering.analyze*` delegates
  these before its temporal/binder/resource proof. Existing `CpuAffineLayoutLowering` and
  `CpuAccessPlan` provide layout vocabulary; do not move element enumeration into run binding.
- `ir/CpuKernelIr`, `CpuMatmulIr`, `CpuAggregateIr`, `CpuGeneralScanIr.Stage/Stages/View/Cut`:
  retain existing typed IR; add only nested `CpuAggregateIr.Body`/`Accumulation` as proposed.
  Sequential SUM is not an ordinary exact-SUM object with its state guards disabled.
- New `codegen/emit/CpuBodyBindings` and `CpuPointwiseBodyEmitter`: package-private,
  generation-only types with proposal signatures. Existing `CpuLoopEmitter`/`CpuCarrierEmitter`
  own reusable coordinate/address/carrier emission, including v9's dominating constant facts.
- `CpuClassFileKernelGenerator`, `CpuMatmulEmitter`, `CpuAggregateEmitter`, `CpuExactSumEmitter`:
  ordinary entries adapt to the same body methods used by Scan; preserve other family routes.
- `CpuGeneralScanEmitter.Shaped`: retain time/carry/activity/history bindings; delegate math.
  Retire `CpuGeneralScanMatmulEmitter` and `CpuGeneralScanReductionEmitter`, not rename them.
  Factor common fusion/cut proofs from `CpuGeneralScanLowering.fuse*` and
  `CpuGeneralScanStageSchedule`; temporal constraints remain explicit additional inputs.
- `prepare/CpuPartitionPreparer`, `CpuPartitionFinalizer`, `executable/CpuGeneralScanPreparedExecutable`,
  `cache/CpuKernelSpecialization`: adapt requirements/bindings and both identity streams only
  as required. No executor/worker or shared Prepare/Runtime source changes in this stage.
- Focused tests below and retained `CpuSymbolicLengthScanCleanJavaOracle`/fixtures. Add shared
  cross-caller fixtures within existing tests; no new general proof framework/test DSL.

## Acceptance criteria

1. Both ordinary and Scan callers reach common family normalization and body emission. Inspect
   call sites and removed source: no duplicate Scan arithmetic loop/axis/opcode implementation,
   renamed Scan-only backend or runtime operation dispatch. Existing ordinary non-Scan families
   retain supported behavior; historical paths need not be rewritten beyond shared seams.
2. MAIN's SUM audit is preserved: exact ordinary SUM/MEAN keep exact limbs/rounding/requirements;
   represented-sequential Scan SUM keeps ordered fadd/dadd, raw singleton/no-axis movement and
   empty/special/zero behavior. Tests include cancellation-sensitive values that distinguish the
   variants. Ordinary-unrolling comparisons account for this algorithm difference; exact checks
   use matching-algorithm oracles. Algorithm/resource mismatches reject before generation;
   no accuracy policy API.
3. Body-local IDs/ordinals never become outer slots. Exact cuts, carries and demanded ports use
   predeclared workspace/outer bindings with all-B inclusive lifetime/size/alias proofs. No
   undeclared scratch or staging increase, per-step geometry allocation or new RunState.
4. Preserve typed actual AD, both traversals/seeds, tied positional inputs/roots, two carries,
   shared parameter sums, dead/demanded histories, fan-out and exact saved sites. Poison padding
   proves activity before both loads/arithmetic, including ROW(1) contractions and SHARED work.
5. Preserve v9 direct stable carriers/temporal bases, explicit B, zero-base/coefficient proofs,
   indexed contractions, affine-prefix/cell sharing and safe bank-free schedule. Compare generated
   loop/dataflow and allocations/calls with specialized clean Java; do not substitute an entry
   geometry decoder or generic helper in the hot body. Failed proofs retain legal fallbacks.
6. Exercise T=0/B=0/1/empty K/zero features, 64→40→1→64 reusable geometry, scalar-to-dense carry,
   forwarded final pre-step views, aliases/overlap, overflow, failure/retry/close/concurrent runs.
   Preserve ordinaryBatch F32/direct-only guard and reject all-valid symbolic F64 Scan.
7. Frozen unaffected artifacts keep keys/bytes; explicitly approved changed artifacts advance
   both streams and reject stale ClassFiles. Test differing algorithm/roles/demand/access/carry/
   resource facts and compatible reuse. No actual B/L/addresses/slots enter canonical identity.

## Validation

Commands below are future worker validation after Ready, not permission to run during CPU0066.
Focused development selection, then one final affected-module test/Javadoc run:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScanLoweringTest' --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuGeneralScanExecutionTest' --tests '*CpuAggregateGeneratedKernelTest' --tests '*CpuMatmulTanhEpilogueTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

MAIN CPU0067 checkpoint after stabilized code/review (once):

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*CpuSymbolicScanConformanceTest' --tests '*CpuMatmulStrategyConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuSymbolicScanIntegrationTest'
```

Record actual skips/environment, do not alter unrelated gates. Specialized Java/Class-File
assertions live in focused generated tests. Proposed fresh diagnostic, only after explicit launch
approval: add `sharedBodyScalarDiagnostic` to existing `CpuGeneralScanTimingTest`, selected alone:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScanTimingTest.sharedBodyScalarDiagnostic' -Dsynaptik.cpu.generalScan.performance=true
```

Budget: one test JVM, no automatic retry/forks, at most 120 seconds of diagnostic body. Freeze
matching work before source edits: the proposal's two-layer MLP plus combined length RNN
T=4/B=7/F=3/H=8 forward and actual reverse, F32/F64, comparing accepted generated baseline,
new generated and specialized clean Java. Retain baseline ClassFiles/identities before migration;
run untimed raw/finite checks before warmup and interleaved samples. Separate cold generation,
admission/entry and kernel costs; report variance without a universal speed claim. This method is
a new scoped grant, not other historical timing selectors. MAIN must accept this exact finite
protocol before Ready/launch. Root validation only if actual scope changes justify it.

## Documentation, review and result

Update affected implementation Javadocs and CPU backend explanation; update compatibility
authority only if affected, ADR only if significant. Use matching documentation profiles;
target glossary terms, not a full read. Independent Class C code/docs review must confirm real
common ownership, preserved numerics/resources and v9 shape, reusing successful test evidence.
Result empty until execution; report changed files, exact checks/limits and unresolved follow-up,
ending `Status: Complete` or `Status: Incomplete` plus `Follow-up required: ...`.
