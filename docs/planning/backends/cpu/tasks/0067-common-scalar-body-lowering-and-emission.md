# CPU0067: Common [scalar](../../../../glossary.md#term-scalar) body [lowering](../../../../glossary.md#lowering) and emission

## Reader context

Read the named types and signatures below as archived proposals, not APIs to instantiate.
The intended callers were ordinary CPU entry generation and Scan's temporal emitter, with
shared arithmetic generation between them. Body-local value IDs describe the captured
computation; they are not outer physical memory slots. A saved history or carry crosses that
boundary only through predeclared storage and a binding whose lifetime covers every use.

Today's [common pointwise and SUM explanation](../../../../backend-guide/cpu-backend.md#common-pointwise-and-sum-composition)
names the actual shared owners: `CpuClassFileKernelGenerator.emitBoundPointwise` and
`CpuAggregateEmitter.emitRepresentedSum`. It also explains why common ownership preserves
ordinary exact SUM and Scan's represented sequential SUM as different algorithms. No command
or diagnostic proposal in this superseded Draft grants a new execution slot.

> Superseded by CPU0070's [replacement plan](../existing-emitter-composition-plan.md) and
> [CPU0071 foundation](0071-common-matmul-binding-foundation.md). The original Draft below is
> archived, not executable. Its four named types and diagnostic budget are not carried forward.

## Status and change class

Superseded. The following pre-supersession status and remaining brief are archived:

Draft; not a source frontier. Requires reviewed/MAIN-accepted CPU0066, verified dependencies,
**user** approval of all four proposed named types (including nested Body/Accumulation) and the
exact diagnostic budget. Private generation-only types are not exempt from the scoped stop.
No launch/build/measurement authorized; extraction alone needs no new root/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) rule.
Class C: generated-code [shape](../../../../glossary.md#shape), exact arithmetic/resource requirements and temporal [binding](../../../../glossary.md#term-binding).

## Goal and bounded [scope](../../../../glossary.md#term-scope)

Deliver one common [scalar](../../../../glossary.md#term-scalar) math path for ordinary and existing captured [Scan](../../../../glossary.md#term-cumulative-scan) bodies, including
typed [lowering](../../../../glossary.md#lowering), emission, tests and documentation. Scan retains traversal and temporal [bindings](../../../../glossary.md#term-binding),
not its own [pointwise](../../../../glossary.md#term-pointwise)/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/SUM algorithms or a private [optimizer](../../../../glossary.md#term-optimizer) behind a shared [facade](../../../../glossary.md#term-facade).
Follow the [proposal's call flow](../shared-body-emission-proposal.md#concrete-common-call-flow-and-proposed-types).
Preserve static all-valid [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding), symbolic all-valid F32 and static/symbolic length F32/F64
forward/actual first-order reverse, with current [layout](../../../../glossary.md#layout)/read/binder exclusions.

Non-goals: capability expansion, [vector](../../../../glossary.md#term-vector)/parallel Scan, uniform reduction numerics, ordinary
symbolic F64 MATMUL/epilogues, [Compiler](../../../../glossary.md#term-compiler)-body rewrites, [Model](../../../../glossary.md#term-model)/[API](../../../../glossary.md#term-application-programming-interface-api)/[persistence](../../../../glossary.md#term-persistence)/[Runtime](../../../../glossary.md#term-runtime)/build changes,
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) packing/migration, native/low Scan, tuning or unrelated CPU0048–0052 repairs.

## Contracts and pre-code gate

Read AGENTS; root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
[scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>; [backend](../../../../glossary.md#backend)
[Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>,
[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> and
[Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>;
[Scan](../../../../glossary.md#term-cumulative-scan) [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
[Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic)
and [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories);
[Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare) [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.

Before Java, MAIN verifies a [Ready](../../../../glossary.md#term-ready) packet, stable common seams and a changed-[artifact](../../../../glossary.md#term-artifact) policy.
No root/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) change is required merely to factor private codegen classes/methods. Preserve existing
byte obligations where possible; if affected, enumerate changed classes, advance both [identity](../../../../glossary.md#term-identity-meanings)
streams and coordinate only affected root/backend version/retention facts before conflicting
generation. ADR only for a significant actual decision. Scan
[Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates)<span data-glossary-navigation="true"> ([Qualification glossary](../../../../glossary.md#term-qualification))</span>
requires user approval of new named types; MAIN cannot self-approve the four-type packet below.
Any numerical/[ownership](../../../../glossary.md#term-ownership-meanings)/Runtime ambiguity stops execution. This Draft authorizes no authority edit.

## Dependencies and integration

- Depends on: CPU0066 accepted; user-approved named-type/diagnostic packet; CPU0064/0065 Complete;
  retained Compiler0015/0016 and Prepare0015.
- Conflicts with: [CPU](../../../../glossary.md#term-cpu) common-body, [Scan](../../../../glossary.md#term-cumulative-scan), [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/aggregate, [cache](../../../../glossary.md#term-cache), workers and CPU0048–0052 writers.
- Parallel group: None; clean sole implementation writer, independent read-only review.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty tree;
  MAIN must record a fresh preservation snapshot/contract set before Ready, not assume a merge.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Integration order: reviewed seams/any necessary compatibility updates → scalar migration → independent
  review/fixes → MAIN checkpoint/acceptance → consider CPU0068. No automatic launch.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Integration [validation](../../../../glossary.md#term-validation): CPU0067 [checkpoint](../../../../glossary.md#term-checkpoint) below, scoped docs/links/whitespace/preservation.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; delegate owned explanation/[Javadoc](../../../../glossary.md#term-javadoc) exclusively.

## Files and symbols

Source prefix: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`.

- `lowering/CpuPartitionLowering`, `CpuMatmulLowering`, `CpuAggregateLowering`: extract shared
  facts-only instruction/geometry/axis validators; <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>`CpuGeneralScanLowering.analyze*` delegates
  these before its temporal/binder/resource proof. Existing `CpuAffineLayoutLowering` and
  `CpuAccessPlan` provide [layout](../../../../glossary.md#layout) vocabulary; do not move [element](../../../../glossary.md#term-element) enumeration into [run](../../../../glossary.md#term-run) [binding](../../../../glossary.md#term-binding).
- `ir/CpuKernelIr`, `CpuMatmulIr`, `CpuAggregateIr`, `CpuGeneralScanIr.Stage/Stages/View/Cut`:
  retain existing typed IR; add only nested <span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>`CpuAggregateIr.Body`/`Accumulation` as proposed.
  Sequential SUM is not an ordinary exact-SUM object with its state guards disabled.
- New `codegen/emit/CpuBodyBindings` and `CpuPointwiseBodyEmitter`: package-private,
  generation-only types with proposal signatures. Existing <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>`CpuLoopEmitter`/`CpuCarrierEmitter`
  own reusable coordinate/address/carrier emission, including v9's dominating constant facts.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [carrier glossary](../../../../glossary.md#term-storage-carrier); [constant glossary](../../../../glossary.md#term-constant))</span>
- `CpuClassFileKernelGenerator`, `CpuMatmulEmitter`, `CpuAggregateEmitter`, `CpuExactSumEmitter`:
  ordinary entries adapt to the same body methods used by Scan; preserve other family routes.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- `CpuGeneralScanEmitter.Shaped`: retain time/carry/activity/history bindings; delegate math.
  Retire <span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry))</span>`CpuGeneralScanMatmulEmitter` and `CpuGeneralScanReductionEmitter`, not rename them.
  Factor common fusion/cut proofs from <span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>`CpuGeneralScanLowering.fuse*` and
  `CpuGeneralScanStageSchedule`; temporal constraints remain explicit additional inputs.
- `prepare/CpuPartitionPreparer`, `CpuPartitionFinalizer`, `executable/CpuGeneralScanPreparedExecutable`,
  `cache/CpuKernelSpecialization`: adapt requirements/bindings and both identity streams only
  as required. No executor/worker or shared Prepare/Runtime source changes in this stage.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Focused tests below and retained `CpuSymbolicLengthScanCleanJavaOracle`/fixtures. Add shared
  cross-caller fixtures within existing tests; no new general proof framework/test DSL.<span data-glossary-navigation="true"> ([DSL glossary](../../../../glossary.md#term-dsl))</span>

## Acceptance criteria

1. Both ordinary and Scan callers reach common family normalization and body emission. Inspect
   call sites and removed source: no duplicate Scan arithmetic loop/axis/opcode implementation,
   renamed Scan-only backend or runtime operation dispatch. Existing ordinary non-Scan families
   retain supported behavior; historical paths need not be rewritten beyond shared seams.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization); [axis glossary](../../../../glossary.md#term-axis); [opcode glossary](../../../../glossary.md#term-opcode); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
2. MAIN's SUM audit is preserved: exact ordinary SUM/MEAN keep exact limbs/rounding/requirements;
   represented-sequential Scan SUM keeps ordered fadd/dadd, raw singleton/no-axis movement and
   empty/special/zero behavior. Tests include cancellation-sensitive values that distinguish the
   variants. Ordinary-unrolling comparisons account for this algorithm difference; exact checks
   use matching-algorithm oracles. Algorithm/resource mismatches reject before generation;
   no accuracy policy API.<span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean); [rounding glossary](../../../../glossary.md#term-rounding); [singleton glossary](../../../../glossary.md#term-singleton); [cancellation glossary](../../../../glossary.md#term-cancellation); [unrolling glossary](../../../../glossary.md#term-unrolling); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
3. Body-local IDs/ordinals never become outer slots. Exact cuts, carries and demanded ports use
   predeclared workspace/outer bindings with all-B inclusive lifetime/size/alias proofs. No
   undeclared scratch or staging increase, per-step geometry allocation or new RunState.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [lifetime glossary](../../../../glossary.md#term-lifetime); [alias glossary](../../../../glossary.md#aliasing-and-arena); [RunState glossary](../../../../glossary.md#run-state--runstate))</span>
4. Preserve typed actual AD, both traversals/seeds, tied positional inputs/roots, two carries,
   shared parameter sums, dead/demanded histories, fan-out and exact saved sites. Poison padding
   proves activity before both loads/arithmetic, including ROW(1) contractions and SHARED work.<span data-glossary-navigation="true"> ([AD glossary](../../../../glossary.md#autograd); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [parameter glossary](../../../../glossary.md#term-parameter); [fan-out glossary](../../../../glossary.md#term-fan-out); [padding glossary](../../../../glossary.md#padding))</span>
5. Preserve v9 direct stable carriers/temporal bases, explicit B, zero-base/coefficient proofs,
   indexed contractions, affine-prefix/cell sharing and safe bank-free schedule. Compare generated
   loop/dataflow and allocations/calls with specialized clean Java; do not substitute an entry
   geometry decoder or generic helper in the hot body. Failed proofs retain legal fallbacks.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow))</span>
6. Exercise T=0/B=0/1/empty K/zero features, 64→40→1→64 reusable geometry, scalar-to-dense carry,
   forwarded final pre-step views, aliases/overlap, overflow, failure/retry/close/concurrent runs.
   Preserve ordinaryBatch F32/direct-only guard and reject all-valid symbolic F64 Scan.<span data-glossary-navigation="true"> ([features glossary](../../../../glossary.md#term-software-feature); [scalar glossary](../../../../glossary.md#term-scalar); [carry glossary](../../../../glossary.md#term-carry); [views glossary](../../../../glossary.md#term-view-meanings); [overflow glossary](../../../../glossary.md#term-overflow); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [guard glossary](../../../../glossary.md#term-guard); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
7. Frozen unaffected artifacts keep keys/bytes; explicitly approved changed artifacts advance
   both streams and reject stale ClassFiles. Test differing algorithm/roles/demand/access/carry/
   resource facts and compatible reuse. No actual B/L/addresses/slots enter canonical identity.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## [Validation](../../../../glossary.md#term-validation)

Commands below are future worker validation after [Ready](../../../../glossary.md#term-ready), not permission to [run](../../../../glossary.md#term-run) during CPU0066.
Focused development selection, then one final affected-[module](../../../../glossary.md#term-module) test/[Javadoc](../../../../glossary.md#term-javadoc) run:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScanLoweringTest' --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuGeneralScanExecutionTest' --tests '*CpuAggregateGeneratedKernelTest' --tests '*CpuMatmulTanhEpilogueTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

MAIN CPU0067 [checkpoint](../../../../glossary.md#term-checkpoint) after stabilized code/review (once):

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

Budget: one test [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), no automatic retry/[forks](../../../../glossary.md#term-fork-meanings), at most 120 seconds of diagnostic body. Freeze
matching work before source edits: the proposal's two-layer [MLP](../../../../glossary.md#term-multilayer-perceptron-mlp) plus combined length [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)
T=4/B=7/F=3/H=8 forward and actual reverse, [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding), comparing accepted generated baseline,
new generated and specialized clean Java. Retain baseline ClassFiles/[identities](../../../../glossary.md#term-identity-meanings) before migration;
run untimed raw/[finite](../../../../glossary.md#term-finite) checks before [warmup](../../../../glossary.md#term-warmup) and interleaved [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters). Separate cold generation,
[admission](../../../../glossary.md#term-admission)/entry and [kernel](../../../../glossary.md#kernel) costs; report [variance](../../../../glossary.md#term-variance) without a universal speed claim. This method is
a new scoped grant, not other historical timing selectors. MAIN must accept this exact finite
protocol before Ready/launch. Root validation only if actual [scope](../../../../glossary.md#term-scope) changes justify it.

## Documentation, review and result

Update affected implementation Javadocs and [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) explanation; update compatibility
authority only if affected, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) only if significant. Use matching documentation profiles;
[target](../../../../glossary.md#term-target) glossary terms, not a full read. Independent Class C code/docs review must confirm real
common [ownership](../../../../glossary.md#term-ownership-meanings), preserved numerics/resources and v9 [shape](../../../../glossary.md#shape), reusing successful test evidence.
Result empty until execution; report changed files, exact checks/limits and unresolved follow-up,
ending `Status: Complete` or `Status: Incomplete` plus `Follow-up required: ...`.
