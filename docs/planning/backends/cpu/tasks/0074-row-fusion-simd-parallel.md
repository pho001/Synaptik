# Task CPU0074: Row-[fusion](../../../../glossary.md#term-fusion) [SIMD](../../../../glossary.md#simd) and parallel realizations

## Reader context

This task adds ways to execute a previously selected fused row region. SIMD means one instruction
processes several feature cells; parallel execution means worker ranges process disjoint rows
and private partial results. Neither name means that every Scan body becomes vectorized or that
adding threads is profitable.

CPU preparation chooses the strategy and declares scratch storage. A bound Scan invocation
borrows a sufficient live reservation from the existing CPU concurrency budget, dispatches
ranges through `CpuWorkerGroup`, and waits for completion before merging partials or reusing
scratch. That internal invocation is the borrower; the run's existing owner retains cleanup
responsibility. “Quiescence” means the workers have stopped using the borrowed resources,
including on failure, so returning from dispatch cannot leave writes racing with cleanup.

Read the [selected-region execution explanation](../../../../backend-guide/cpu-backend.md#implemented-and-accepted-row-fusion-exec-v1)
for range/merge relationships and the distinction between functional qualification and the
unqualified profitability reported here. The broader
[current Scan composition boundary](../../../../backend-guide/cpu-backend.md#current-general-scan-composition)
is separate from this historical region's dtype and strategy matrix.

## Status

Complete — MAIN accepted functional capability and bounded [validation](../../../../glossary.md#term-validation), 2026-10-30.
Steady-state profitability is not qualified; [scalar](../../../../glossary.md#term-scalar)/thread-count-one defaults remain unchanged.

## Change class

Class C — generated arithmetic grouping, worker [admission](../../../../glossary.md#term-admission), emitted entry [ABI](../../../../glossary.md#term-application-binary-interface-abi) and scratch [lifetime](../../../../glossary.md#term-lifetime).
One cohesive [CPU](../../../../glossary.md#term-cpu) capability; clean implementation and independent targeted review are mandatory.

## Goal

Extend CPU0073's selected row-map/multiple-terminal regions from [scalar](../../../../glossary.md#term-scalar) to [SIMD](../../../../glossary.md#simd) ([single
instruction, multiple data](../../../../glossary.md#simd)), parallel-scalar and parallel-SIMD, reusing existing math [owners](../../../../glossary.md#term-owner).
Keep the emitted Scan time loop, one [RunState](../../../../glossary.md#run-state--runstate) and separate per-active-time [carry](../../../../glossary.md#term-carry) ADDs.
The [source-bound proposal](../row-fusion-simd-parallel-proposal.md) records seams, resources,
numerical limits and the accepted authority map; it is not authority itself.

## [Scope](../../../../glossary.md#term-scope)

The rows below bound eligible [fused](../../../../glossary.md#specialization-and-fusion) regions, not every [operation](../../../../glossary.md#operation) or whole Scan. Each new
realization needs its own complete [binding](../../../../glossary.md#term-binding) and forward/actual-reverse [qualification](../../../../glossary.md#term-qualification).

| Existing domain | Arithmetic | New realizations when proved |
|---|---|---|
| Ordinary static | [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) | [Vector](../../../../glossary.md#term-vector), parallel-[scalar](../../../../glossary.md#term-scalar), parallel-vector |
| Ordinary named-B symbolic | F32 only | Same; no ordinary symbolic F64 [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) |
| Static all-valid Scan | F32/F64 | Same, only at selected fused regions |
| All-valid [symbolic Scan](../../../../glossary.md#symbolic-scan) | F32 only | Same; no all-valid symbolic F64 |
| Static-length / combined symbolic-length Scan | F32/F64 | Same; retain exact [INT64](../../../../glossary.md#data-type--datatype) control |

Retain CPU0073 scalar as baseline/fallback. Ordinary arrays/MemorySegments and Scan's checked
MemorySegments retain existing eligibility; no new [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout)/[dtype](../../../../glossary.md#data-type--datatype). Vectorize bounded
[feature](../../../../glossary.md#term-feature)/output cells inside one active row, with [masks](../../../../glossary.md#term-boolean-mask) for feature tails and ordered feature K.
Primary actual F3/H2 saved-[primal](../../../../glossary.md#term-primal) [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) reverse must select and execute each new region realization
under explicit eligible configuration, including mixed lengths. Qualify a genuinely fused forward
[fixture](../../../../glossary.md#term-fixture) separately (a forward-only SHARED-[carry](../../../../glossary.md#term-carry) body may qualify). Ordinary RNN forward
`x@W+h@U+b+tanh` may have no terminal region and must remain scalar in that case.

## Non-goals

No general [pointwise](../../../../glossary.md#term-pointwise)-fuser repair, blanket all-[operation](../../../../glossary.md#operation) Scan vectorization, new public numerical
mode/profile/certification, implicit [FMA](../../../../glossary.md#term-fused-multiply-add-fma)/[TF32](../../../../glossary.md#term-tf32)/lower [precision](../../../../glossary.md#term-precision), [feature](../../../../glossary.md#term-feature)-K or standalone [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)-K
[reassociation](../../../../glossary.md#term-reassociation), exact ordinary SUM/[MEAN](../../../../glossary.md#term-mean) change, across-time [carry](../../../../glossary.md#term-carry) promotion, low types, wider
[layouts](../../../../glossary.md#layout), [OpenBLAS](../../../../glossary.md#term-openblas)/native route, new executor/global Job/pool/cohort framework, interpreter,
per-op [runtime](../../../../glossary.md#term-runtime) [lookup](../../../../glossary.md#term-lookup), per-B [cache](../../../../glossary.md#term-cache), T-fold code/[buffers](../../../../glossary.md#term-buffer), mandatory Job reuse or [commit](../../../../glossary.md#term-commit)/push.
Keep CPU0010M production [scalar](../../../../glossary.md#term-scalar)/thread-count-one defaults. Existing explicit configuration/tuning
selects eligible work cold; adding capability does not force workers on B7/B32.

## Contracts

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>.
- [Backend](../../../../glossary.md#backend) [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>,
  [CPU0073 accepted row-map and multiple-terminal reduction fusion](../../../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>,
  [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>.
- Scan [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>,
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates),
  [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
  [Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>,
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>.
- Foundational [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives)
  and [Aggregate floor](../../../../architecture/contracts/foundational-modules.md#aggregate-floor)<span data-glossary-navigation="true"> ([Aggregate floor glossary](../../../../glossary.md#term-aggregate-floor))</span>, unchanged.
- [Runtime](../../../../glossary.md#term-runtime) [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.
- Implemented exceptions: [CPU mechanism](../../../../architecture/contracts/backend-execution.md#cpu0074-selected-row-region-execution-exception),
  [Scan semantics](../../../../architecture/contracts/recurrent-scan.md#cpu0074-selected-terminal-grouping-exception),
  [run reservation](../../../../architecture/contracts/runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation).

[Scalar](../../../../glossary.md#term-scalar)/ordered/no-[dispatch](../../../../glossary.md#term-dispatch) rules still govern outside the exact selected exec-v1 exception.
The staged cutover changes the bounded CPU0073/Scan override, not global [Model](../../../../glossary.md#term-model) aggregate semantics.
Stop on ambiguity or additional types/seams; implementation stays within this accepted packet.

## Dependencies and integration

- Depends on: CPU0073 Complete; retained CPU0070–0072, CPU0064/0065,
  Compiler0015/0016 and Prepare0015 Complete baselines; authority accepted by MAIN receipt below.
- Conflicts with: held CPU0048–0052; common emission, Scan binding/resources/identity,
  worker/budget ownership and shared authority/documents. Pointwise audit findings stay outside.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Parallel group: None; exclusive serial writer, independent audit read-only.
- Common base revision: HEAD=origin/main `07a281edd3086efde2853211b71b48288db027e1`
  plus accepted dirty base, index empty; preserve the authentic CPU0073 2565-path inventory.
- Integration order: mechanism APPROVE/P2 closure → exclusively delegated actual contract/docs
  patch → final independent literal-diff review → MAIN authority/Ready/frontier verification → clean CPU
  implementation → independent code/Javadoc/docs review → MAIN named join and closure.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Integration validation: final affected CPU module/Javadoc once; named conformance/Engine join
  below. MAIN runs one root checkpoint if boundary impact requires it, never duplicate campaigns.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; this exclusively delegated [planning](../../../../glossary.md#term-planning)/docs writer only.

Historical MAIN [Ready](../../../../glossary.md#term-ready) verification, 2026-10-30: dependencies/common base/preservation verified;
authority/docs APPROVE 0 P1/P2, 51 links, no conflicting writer, Parallel group None, sole CPU0074 frontier.

## Files and symbols

All production paths below are relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.

- `internal/prepare/CpuPartitionPreparer`, `CpuPartitionPreparationPlan.ExecutionUnitPlan`,
  `CpuPartitionFinalizer` — existing strategy/range policy, declarations and [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) resources.
- `internal/lowering/CpuRowMapReductionLowering`, `CpuGeneralScanLowering.shapedResources/Geometry/Facts.pack`
  and `internal/ir/CpuGeneralScanIr.Stages` — vector/alias/temporal proof and exact workspace.
  Keep <span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>`CpuRowMapReductionIr`'s [canonical](../../../../glossary.md#term-canonical-form) [topology](../../../../glossary.md#term-topology)/key free of worker policy.
- `internal/cache/CpuKernelSpecialization` — nested `RowFusionEmission` and optional
  emission payload, selected entry type and both identity streams; old constructors delegate empty.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `internal/codegen/emit/CpuRowMapReductionEmitter`, `CpuClassFileKernelGenerator`,
  `CpuGeneralScanEmitter.emitRegionalStage`, `CpuGeneratedKernel` — range/merge methods,
  existing emitted outer loop and explicit cold helper accessors, not a registry.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>
- Addenda: `CpuGeneralScanEmitter.emitRowRange` and ordinary symbolic/Scan executable overloads
  transport existing range/minimum-row/borrowed-worker facts. Old six-argument forms retain inline
  behavior; exact signatures are in the CPU contract. Policy stays in the unit, not IR/Facts/spec.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [signatures glossary](../../../../glossary.md#term-signature); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Existing `CpuVectorInstructionEmitter`, `CpuVectorMath`, `CpuLoopEmitter`, `CpuCarrierEmitter`,
  `CpuMatmulEmitter`, `CpuAggregateEmitter` — reuse, including checked masked F32/F64 byte access
  through <span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`CpuCarrierEmitter.vectorLoadBytes/vectorStoreBytes` for selected Scan/ordinary partial scratch.
- `internal/executable/CpuPreparedExecutable.bindCompatible/bindOrdinaryBatch/Invocation`,
  `CpuGeneralScanPreparedExecutable.Invocation` plus nested `StepDispatch`, `CpuWorkerGroup.execute`,
  `CpuConcurrencyBudget.Lease` — cold [run](../../../../glossary.md#term-run) [binding](../../../../glossary.md#term-binding), borrowed reservation, join and ordered merge.
- `internal/ir/CpuFusionDecision`, `internal/lowering/CpuFusionProfitabilitySelector` and plan validation —
  guarded ordinary-only <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`ROW_REGION_STATE` [metadata](../../../../glossary.md#term-metadata); retain indivisible row boundaries and [pointwise](../../../../glossary.md#term-pointwise) scoring.
- Existing row-[fusion](../../../../glossary.md#term-fusion)/[lowering](../../../../glossary.md#lowering)/Scan/worker/budget tests and oracles; add test-only
  `CpuRowFusionSimdParallelTest`, `CpuRowFusionSimdParallelJavaOracle`,
  `CpuRowFusionSimdParallelTimingTest` in existing [CPU](../../../../glossary.md#term-cpu) `internal.codegen.emit` test package.

## Acceptance criteria

1. Exercise all four realizations in every admitted matrix domain using eligible fixtures,
   including actual F3/H2 reverse and a separate fused forward. Assert selected strategy and
   actual vector/worker execution, not just equal outputs. Negative/no-region cases remain scalar.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [fused glossary](../../../../glossary.md#specialization-and-fusion); [vector glossary](../../../../glossary.md#term-vector); [scalar glossary](../../../../glossary.md#term-scalar))</span>
2. Only fused terminal R/B contributors may regroup. Preserve dtype, each MUL/ADD site, sequential
   time and one distinct carry ADD per active time. Ordinary exact SUM/MEAN, singleton/no-axis
   movement, masks/indices/casts, inactive raw identity and selected-realization saved bits stay exact.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [carry glossary](../../../../glossary.md#term-carry); [MEAN glossary](../../../../glossary.md#term-mean); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [masks glossary](../../../../glossary.md#term-boolean-mask); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
3. Range policy is cold and fixed across time; bind numeric B bounds once. Workers own disjoint
   rows/partials; deterministic range-index merge skips invalid empty/inactive partials. No atomic
   floating updates, B×parameter banks or cross-strategy/thread-count/device bitwise claim.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [parameter glossary](../../../../glossary.md#term-parameter); [device glossary](../../../../glossary.md#term-device))</span>
4. Prove transpose lifetime: W/U cost is 10 cells (40/80 bytes before alignment) per run only
   when immutable across T. Changing cuts/SHARED histories stage at the correct active time or
   fall back. Reuse staged data only through its proved lifetime; retain saved/public stores.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [lifetime glossary](../../../../glossary.md#term-lifetime); [alignment glossary](../../../../glossary.md#term-alignment); [run glossary](../../../../glossary.md#term-run); [immutable glossary](../../../../glossary.md#term-immutable))</span>
5. Before writes, retain full after-producer preflight and accessible worker segments. One RunState
   owns arrays/scratch/calls. Borrow one same-budget live sufficient reservation synchronously;
   no nested acquisition/submission, concurrent borrowing or release before quiescence. Budget 1,
   failure, interruption, close races, cleanup/retry and concurrent mixed-B runs have durable tests.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [preflight glossary](../../../../glossary.md#term-preflight); [RunState glossary](../../../../glossary.md#run-state--runstate); [borrowing glossary](../../../../glossary.md#term-borrowing); [release glossary](../../../../glossary.md#term-release); [quiescence glossary](../../../../glossary.md#term-quiescence))</span>
6. Cold-prove all resources/aliases/lifetimes and CPU0073 limits before slots. Generated outer T
   remains emitted; no graph/geometry rediscovery or body allocation per step. Scheduler Job/queue
   metadata is the only per-dispatch allocation exception; Job reuse is not required.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [graph glossary](../../../../glossary.md#graph); [metadata glossary](../../../../glossary.md#term-metadata); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
7. Test generated versus optimal clean Java for each selected realization; inspect all helper
   ClassFiles for arithmetic/dataflow, FMA absence, hidden allocation/boxing/reflection/dispatch
   and avoidable elementwise overhead. Test vector feature tails, B=0/1, T=0, all-inactive,
   poisoned padding, cancellation, NaN/Inf, signed zeros, subnormals and overflow/underflow.
   Finite per-operation/dtype tolerances are test-only; class/sign/raw checks remain separate.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [elementwise glossary](../../../../glossary.md#term-elementwise); [feature glossary](../../../../glossary.md#term-feature); [padding glossary](../../../../glossary.md#padding); [cancellation glossary](../../../../glossary.md#term-cancellation); [NaN glossary](../../../../glossary.md#term-nan); [subnormals glossary](../../../../glossary.md#term-subnormal); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [Finite glossary](../../../../glossary.md#term-finite); [operation glossary](../../../../glossary.md#operation))</span>
8. Preserve scalar CPU0073 keys/ClassFiles/resources and noneligible controls literally. New
   realization payloads bind all emitted facts in both streams; stale/load/reuse and B/L-independent
   cache tests pass. No global envelope/projection bump; no compatibility evidence regenerated
   from candidate code. MAIN's frozen 590 production classes are available as literal controls.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>

## [Validation](../../../../glossary.md#term-validation)

Owning commands; this documentation pass launches no Java/test/benchmark work:

```bash
./gradlew :backends:cpu:test --tests '*CpuRowFusionSimdParallelTest' --tests '*CpuRowMapReduction*Test' --tests '*CpuGeneralScanExecutionTest' --tests '*CpuWorkerGroupTest' --tests '*CpuConcurrencyBudgetTest' --offline
./gradlew :backends:cpu:test :backends:cpu:javadoc --offline
```

MAIN named join (receipts below; no successful [Engine](../../../../glossary.md#term-engine) rerun):

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*CpuSymbolicScanConformanceTest' --tests '*CpuOrdinaryDynamicBatchConformanceTest' :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuSymbolicScanIntegrationTest' --tests '*EngineCpuOrdinaryDynamicBatchIntegrationTest' --tests '*EngineNumericalSemanticsIntegrationTest' --offline
```

No [Model](../../../../glossary.md#term-model)/[dependency](../../../../glossary.md#term-dependency)/native-[ABI](../../../../glossary.md#term-application-binary-interface-abi)/public-workflow change; MAIN records bounded coverage, no root campaign.
The proposal's sole [finite](../../../../glossary.md#term-finite) performance budget is exhausted; no retry, extra [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) or old campaign reopens.
This docs pass checks links/whitespace and frozen source [hashes](../../../../glossary.md#term-hash); it launches no executable work.

## Documentation and review impact

Independent review [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) root/contracts, explanations, ADR0053, guide, glossary and Javadocs.
MAIN's final [CPU](../../../../glossary.md#term-cpu)/[Javadoc](../../../../glossary.md#term-javadoc) gate passed after the comment freeze; this pass duplicates no executable evidence.
Size note: this Class C brief keeps exact contracts, acceptance and final [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) limitations together (>15 KB).

## Result

Status: Complete
MAIN accepted functional capability and bounded [validation](../../../../glossary.md#term-validation); independent review has no open source findings.
Evidence root: `/private/tmp/synaptik-cpu0074-run.brjyRN` (relative [artifact](../../../../glossary.md#term-artifact) paths below).

- Final post-fix [CPU](../../../../glossary.md#term-cpu): 1438 tests, 0 failures/errors, 55 skips, 6m8s; [Javadoc](../../../../glossary.md#term-javadoc) PASS, 94 historical warnings (`main-final-cpu-javadoc.log`). Historical module2 is superseded.
- [Engine](../../../../glossary.md#term-engine) 27 PASS/[conformance](../../../../glossary.md#term-conformance) 17 PASS: `main-integration-receipts.txt` and its named [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings). Focused 99/0 failures/errors/1 opt-in skip: `impl.VJzWki/short-ordinary-frozen/`.
- Retention: 356 authentic [scalar](../../../../glossary.md#term-scalar) controls; all 32 final ClassFiles match reviewed [capture](../../../../glossary.md#term-capture-meanings), all 8 Scan B pairs identical; all 512 source/resources match the accepted comment freeze.
- MAIN preservation: original 2565 paths present, current 2574; 34 existing changes + 9 new = 43 scoped paths, zero unexpected/missing. [Scope](../../../../glossary.md#term-scope): 17 CPU production + 7 CPU tests + 3 resources + 2 conformance + 14 docs; new files: 3 docs, 3 tests, 3 resources. HEAD=origin/main `07a281e`, index empty.
- Sole performance root: `/private/tmp/synaptik-cpu0074-measure.zow14d`; `REPORT.txt`/`REPORT.json`, `samples.csv`, `JIT-SUMMARY.txt`, `JIT-NMETHODS.json`, `INTERPRETATION.txt`. Exit 0: 256 windows (64 warm/192 measured), 32 pairs; no censor/failure/retry/extra [fork](../../../../glossary.md#term-fork-meanings).
- Raw generated/Java [median](../../../../glossary.md#term-median) ratios 1.016–296.438 are [nonstationary](../../../../glossary.md#term-nonstationary): no steady-state parity, universal speedup or stable strategy [ranking](../../../../glossary.md#term-ranking); profitability remains unqualified.
- Independent [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) review found 43 failed generated [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) attempts: `out of virtual registers in LIR generator`, plus deoptimizations/traps. Missing standalone nmethods do not prove absent caller [inlining](../../../../glossary.md#term-inlining). This concrete pressure must not be dismissed as merely [warm-up](../../../../glossary.md#benchmark-measurement-vocabulary).
- Caller allocation observations exclude workers and do not identify sites. Generated [admission](../../../../glossary.md#term-admission)/[preflight](../../../../glossary.md#term-preflight) is not subtracted. Optimal Java retains twelve typed [carry](../../../../glossary.md#term-carry) cells across T; generator carry promotion remains deferred.
- Existing bounds remain: the [nonlinear](../../../../glossary.md#term-nonlinear) static all-valid [F64](../../../../glossary.md#floating-point-precision-and-rounding) [fixture](../../../../glossary.md#term-fixture) exceeds the old live limit; an admitted `h+xW` witness tests that domain. [Owner](../../../../glossary.md#term-owner) fault injection is [F32](../../../../glossary.md#floating-point-precision-and-rounding) parallel-[vector](../../../../glossary.md#term-vector) Scan, not exhaustive every-owner coverage.
- Separate JIT-pressure/steady-state investigation is recommended, not a new task/frontier/campaign or reopened budget. Defaults stay scalar/max1; no rollout. The two [pointwise](../../../../glossary.md#term-pointwise) audit findings remain separate and unimplemented.
