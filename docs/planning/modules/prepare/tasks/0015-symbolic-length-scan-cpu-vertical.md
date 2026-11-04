# Task Prepare0015: Symbolic-length [Scan](../../../../glossary.md#term-cumulative-scan) [CPU](../../../../glossary.md#term-cpu) vertical

## Status

Complete — MAIN accepted 2026-10-27 after independent Class C code/docs APPROVE (0 P1/P2), owning [validation](../../../../glossary.md#term-validation), the sole bounded timing probe and configured root test/[Javadoc](../../../../glossary.md#term-javadoc) PASS.
[Ready](../../../../glossary.md#term-ready) [admission](../../../../glossary.md#term-admission): independent [planning](../../../../glossary.md#term-planning) APPROVE, 0 open P1/P2; MAIN accepted 2026-10-26. User “pokracuj” authorizes bounded complete implementation including this gate.
PlanningAuthorized=true; AuthorityAuthorized=true (bounded [qualification](../../../../glossary.md#term-qualification)); ImplementationAuthorized=true.
Active frontier: None. JavaLaunchAuthorized=true records the completed clean implementation admission, not a successor launch.
Size exception: 241 lines/~21KB (<25KB); one [atomic](../../../../glossary.md#resource-transaction-and-rollback) cross-[owner](../../../../glossary.md#term-owner) qualification keeps support gates, typed seams,
acceptance and validation evidence together. The pre-Java documentation phase changed no Java work.

## Change class

Class C — cross-[owner](../../../../glossary.md#term-owner) geometry/generated hot paths; clean implementation and independent code/docs review.

## Goal and [scope](../../../../glossary.md#term-scope)

One named B + [runtime](../../../../glossary.md#term-runtime) discrete non-[gradient](../../../../glossary.md#term-gradient) [INT64](../../../../glossary.md#data-type--datatype)[B] lengths, fixed T/[features](../../../../glossary.md#term-feature), [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) ([F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding))
forward AND actual [Compiler](../../../../glossary.md#term-compiler)-generated first-order reverse: [compile](../../../../glossary.md#compile) once → [prepare](../../../../glossary.md#prepare) once → fresh
runValues [descriptors](../../../../glossary.md#term-descriptor)/resources. First qualify bounded outer F64/control/ROW(1) inside this task,
then combined projection/recipes/execution/[publication](../../../../glossary.md#term-publication). All-valid symbolic F64 is not assumed or
required; its current F32-only [qualification](../../../../glossary.md#term-qualification) stays separate.

New [API](../../../../glossary.md#term-application-programming-interface-api)/type/package/session/executor/[memo](../../../../glossary.md#cache-and-memoization)/[dependency](../../../../glossary.md#term-dependency)/representation: None. Retain static/all-valid
routes. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training, cold/copy/wire/v5, native/[OpenBLAS](../../../../glossary.md#term-openblas)/[Metal](../../../../glossary.md#term-metal)/mixed-[owner](../../../../glossary.md#term-owner), variable T/features,
low/mixed [dtypes](../../../../glossary.md#data-type--datatype), [vector](../../../../glossary.md#term-vector)/parallel/tuning, fluent migration and higher-order/createGraph stay held.

## Contracts

Start at [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>.
Exact required headings (including applicable subordinate rules):

- Root: `Core lifecycle`, `Core invariants`, `Dependency rules`, `Testing requirements`.
- [recurrent-scan.md](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>: [`General differentiable scan`](../../../../glossary.md#general-differentiable-scan), `Accepted symbolic-batch scan`, `Length-aware whole-batch scan`, especially `Runtime preflight and implementation gates`.
- [runtime-prepare-engine.md](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine))</span>: `modules/runtime`, `modules/prepare`, `modules/engine`, `Accepted symbolic-batch resource contract`, `Prepare lifecycle`, `Run lifecycle`, `Public reusable prepared execution`.
- [backend-execution.md](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>: `CPU backend routes`, `Accepted ordinary symbolic-batch CPU contract`.
- [compiler-autograd.md](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd))</span>: `Accepted symbolic-batch metadata binding`; no new [AD](../../../../glossary.md#autograd)/binder algorithm/[API](../../../../glossary.md#term-application-programming-interface-api).

Design context only: [length-aware proposal](../../../length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration),
`Minimal coordinated delta and serial capability order`, `Numeric resources, admission and compatibility`,
`Proportional future validation, not runs now`.

## Dependencies and integration

- Depends on: Model0044; Compiler0015; Prepare0011/0013/0014; CPU0042 (retaining CPU0033/0037/0041); Runtime0016; Engine0021; Planning0008 — Complete.
- Conflicts with: [Scan](../../../../glossary.md#term-cumulative-scan) projection/roles/demand; [CPU](../../../../glossary.md#term-cpu) typed closure/[IR](../../../../glossary.md#term-intermediate-representation-ir)/emission/resources/[identities](../../../../glossary.md#term-identity-meanings); [Engine](../../../../glossary.md#term-engine) [binding](../../../../glossary.md#term-binding)/[publication](../../../../glossary.md#term-publication)/[lifecycle](../../../../glossary.md#lifecycle); owning contracts/shared docs. Serialize, no concurrent writer.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus the accepted 2483-file dirty worktree (`prepare0015-baseline.json`), not merged predecessors; index empty.
- Integration order: MAIN review/[Ready](../../../../glossary.md#term-ready) → user approval → in-task pre-Java review/MAIN acceptance → clean implementation → independent code/docs review → MAIN combined gate/acceptance.
- Integration [validation](../../../../glossary.md#term-validation): the single MAIN combined [checkpoint](../../../../glossary.md#term-checkpoint) under Validation, reusing successful worker evidence.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; exclusive serial final docs/[Javadoc](../../../../glossary.md#term-javadoc) delegation.

Launch verification 2026-10-26: MAIN reverified final Ready [planning](../../../../glossary.md#term-planning) unchanged, Complete predecessors/
reviews/gates, sole frontier/inactive conflicts, no other writer. This clean context confirmed predecessor
Status/Result, all 2483 baseline [hashes](../../../../glossary.md#term-hash) and HEAD/origin/index unchanged. MAIN accepted the frozen 19-file [qualification](../../../../glossary.md#term-qualification) after independent APPROVE; sole code-writer launch follows with stable base/conflicts.

## Pre-Java support-gate [qualification](../../../../glossary.md#term-qualification) (same task)

Enacted qualification is implemented and validated at these exact owning headings. Independent
pre-Java review and MAIN acceptance passed on 2026-10-26, before the clean implementation launch.

- Root `Core lifecycle`, `Core invariants`, `Scope-indexed normative contracts`: matching bounded support/status links, successors held.
- recurrent-[scan](../../../../glossary.md#term-cumulative-scan) `Accepted symbolic-batch scan` / `Qualification and mechanism gates`, `Length-aware whole-batch scan` / `Runtime preflight and implementation gates`: combined projection/execution only; unchanged header/equations/roles/[AD](../../../../glossary.md#autograd) guards/saved sites/[preflight](../../../../glossary.md#term-preflight)/resource algorithms.
- [runtime](../../../../glossary.md#term-runtime)-[prepare](../../../../glossary.md#prepare)-[engine](../../../../glossary.md#term-engine) `modules/prepare`, `Prepare lifecycle`, `Run lifecycle`, `Public reusable prepared execution`: static-or-one-B length projection/[admission](../../../../glossary.md#term-admission)/[publication](../../../../glossary.md#term-publication); unchanged [handoff](../../../../glossary.md#term-handoff)/[lifecycle](../../../../glossary.md#lifecycle), no [memo](../../../../glossary.md#cache-and-memoization).
- [backend](../../../../glossary.md#backend)-execution `CPU backend routes`, `Accepted ordinary symbolic-batch CPU contract`: combined [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) and bounded outer/control/ROW(1); retain old F32 bytes/static reduction sizing.
- [compiler](../../../../glossary.md#term-compiler)-[autograd](../../../../glossary.md#autograd) `Accepted symbolic-batch metadata binding`: downstream status only.

Same-change explanations/decision evidence; no authority-only successor. A different clean code context
followed review/MAIN acceptance. No missing contract, unavoidable new type or substantive decision remained.

## Files and symbols

Prefix `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/`: analysis/[PrepareContext](../../../../glossary.md#prepare-context-preparecontext) validateScan/validateRowShape; [GraphPreparation](../../../../glossary.md#graph-preparation--graphpreparation) projection/demand/[handoff](../../../../glossary.md#term-handoff).
[CPU](../../../../glossary.md#term-cpu) prefix `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`: CpuCapabilityProvider; below `internal/`, existing [owners](../../../../glossary.md#term-owner) by directory (all basenames `.java`):

- `lowering/`: CpuPartitionLowering supportsOrdinaryBatch/batchByteSizes/lowerOrdinaryBatch;
  CpuAggregateLowering.BatchGeometry; CpuGeneralScanLowering analyze/lower/Facts.pack/row proof.
- `prepare/`: CpuPartitionPreparer, CpuPartitionFinalizer; `ir/`: CpuGeneralScanIr, CpuPortableKernelIr;
  `cache/`: CpuKernelSpecialization non-[F32](../../../../glossary.md#floating-point-precision-and-rounding) ordinary gate and BOTH [identity](../../../../glossary.md#term-identity-meanings) streams;
  `memory/`: CpuRepresentationRecipes; `route/portable/`: CpuPortableRoutePlan.
- `codegen/emit/`: CpuGeneralScanEmitter **separate combined branch preserving old bytes**;
  CpuGeneralScanReductionEmitter, CpuGeneralScanMatmulEmitter; CpuAggregateEmitter float locals/five-limb/
  carrier assumptions; CpuExactSumEmitter, CpuClassFileKernelGenerator.validate (currently ordinary-batch-v1 only), scalar/carrier emitters.<span data-glossary-navigation="true"> ([limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [carrier glossary](../../../../glossary.md#term-storage-carrier); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- `executable/`: CpuPreparedExecutable.bindOrdinaryBatch F32 span/base hooks;
  CpuGeneralScanPreparedExecutable.executeBound static hooks/stages.batch → run-aware spans/explicit B
  even <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`[B,0]`; CpuPreparedPartitionExecutable orchestration. No new representation/[Runtime](../../../../glossary.md#term-runtime) protocol.

[Engine](../../../../glossary.md#term-engine) prefix `modules/engine/src/main/java/io/github/pho001/synaptik/engine/` → [PreparedExecution](../../../../glossary.md#prepared-execution--preparedexecution)
numeric/[publication](../../../../glossary.md#term-publication) indexing, AdvancedEngine fresh [admission](../../../../glossary.md#term-admission), [RunResult](../../../../glossary.md#run-result--runresult) actual [descriptors](../../../../glossary.md#term-descriptor); retain [dtype](../../../../glossary.md#data-type--datatype)-generic mechanics.
Extend existing tests below; Compiler0015 fixtures are actual-[graph](../../../../glossary.md#graph) references, not redesign. Keep invalid raw-[IR](../../../../glossary.md#term-intermediate-representation-ir) guards.

## Acceptance criteria

1. Qualify outer typed seed/zero/one splats, ADD ties/cross-occurrence sums, required MUL/DIV,
   SUM/MEAN/sumToShape, expand, feature-only reshape/rank edits, transpose/contiguous/same-dtype CAST.
   Explicit family/dtype/rank/axis/attribute/access predicates admit <span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [occurrence glossary](../../../../glossary.md#term-occurrence); [MEAN glossary](../../../../glossary.md#term-mean); [feature glossary](../../../../glossary.md#term-software-feature); [rank glossary](../../../../glossary.md#term-rank); [transpose glossary](../../../../glossary.md#term-transpose); [contiguous glossary](../../../../glossary.md#term-contiguous); [axis glossary](../../../../glossary.md#term-axis))</span>`[T,B,F]` and ROW(1) `[F,B]`/
   `[T,F,B]` literals/zeros, not arbitrary rank-three or ordinary F64 MATMUL. Positive/negative
   provider tests INSIDE/OUTSIDE Scan give identical answers, no origin tag. Only non-gradient<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Scan glossary](../../../../glossary.md#term-cumulative-scan); [gradient glossary](../../../../glossary.md#term-gradient))</span>
   `[B]` [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding)→[INT64](../../../../glossary.md#data-type--datatype) CAST adds control production; direct L needs no [producer](../../../../glossary.md#term-producer).
2. Ordinary F64 exact SUM/MEAN uses existing typed exact arithmetic: signedBits=2099+ceil(log2(N)),
   zero cardinality term at N≤1; int-bounded N fits fixed 34 limbs, 280-byte single-range scratch
   when outputCount>0, else zero with slot/creator retained. F32 keeps five limbs/48 bytes;
   static F64 N=3 stays 33 limbs/272 bytes. Test rational MEAN, subnormals, ±0, NaN/Inf/empties.
   Scan producing-site SUM remains typed sequential, never this ordinary exact algorithm.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality); [subnormals glossary](../../../../glossary.md#term-subnormal); [NaN glossary](../../../../glossary.md#term-nan))</span>
3. Qualify complete forward/actual opposite-traversal Compiler siblings independently, both dtypes;
   no hand-authored reverse substitute. Preserve every-original-carry/sequence ROW(0) AD guards,
   generated mixed ROW(0)/ROW(1)/SHARED histories/constants/shared transpose, joint/missing seeds,
   ties/cross-occurrence sums, false-flag propagation and exact saved TANH/SIGMOID. No replay or
   weakened residual/higher-order preflight.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [carry glossary](../../../../glossary.md#term-carry); [AD glossary](../../../../glossary.md#autograd); [constants glossary](../../../../glossary.md#term-constant); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [TANH glossary](../../../../glossary.md#term-tanh); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [preflight glossary](../../../../glossary.md#term-preflight))</span>
4. Before analyzers validate all outer/local declarations/bindings/latent ports and complete demand,
   named-axis identity not F=B/H=B coincidence. L is separate eight-byte control, no body formal.
   Preindex numeric domains; before borrow/allocation check fixed-feature products behind empties,
   int extents/counts/strides/spans, long bytes/alignment/padding/aggregate overflow. No L-based shapes.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [identity glossary](../../../../glossary.md#term-identity-meanings); [strides glossary](../../../../glossary.md#term-stride); [alignment glossary](../../../../glossary.md#term-alignment); [padding glossary](../../../../glossary.md#padding); [overflow glossary](../../../../glossary.md#term-overflow); [shapes glossary](../../../../glossary.md#shape))</span>
5. Use existing fixed/[run](../../../../glossary.md#term-run)-size recipes and explicit [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-required mode even for `[B,0]`. Carry B
   explicitly in fresh numeric packing, never derive it from frame/features. Declare every bank,
   cut/temporary/output/history/zero-byte workspace before slots, with actual B/typed widths.
   Row history T×B×row-elements, SHARED T×P, dead history zero bytes/stores; accumulator/contribution
   banks O(P), never B×P. Prove cut lifetime/size dominance for every B or keep disjoint regions;
   raw carry backing stays independent of padded histories.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element); [accumulator glossary](../../../../glossary.md#accumulator-type); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
6. After producers recheck run-aware typed spans/access/alignment/overlap and ALL B lengths in
   [0,T] before first Scan output/padding/history/scratch write, including invalid-last/T=0;
   B=0 has no lengths. Scan-local rejection, not upstream rollback. Activity dominates sequence/
   seed/history padding loads AND arithmetic, including hoisted/shared work. Inactive carry/dInitial
   carry is raw initial/final-seed identity, emissions/dX raw +0. Reduce active contributors in
   increasing b, then one distinct represented shared-carry ADD per active time; inactive bypass.
   Preserve CPU0042 fusion/SUM-epilogue and exact saved-site guards.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [fusion glossary](../../../../glossary.md#term-fusion); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>
7. One prepared [Engine.runValues](../../../../glossary.md#fresh-value-binding--enginerunvalues) witness per [dtype](../../../../glossary.md#data-type--datatype)/direction/direct-or-produced ingress: T=4/F=3/H=2,
   `next=tanh(xStep@Wx+h@Wh+bias)`, joint final/emission seeds, nonuniform data; B=32→32→7→1→32,
   L respectively `b%5`, `4-b%5`, `b%5`, `[0]`, all 4, plus repeated L. Compare outputs/actual gradients
   to independent active-prefix unrolling. Final <span data-glossary-navigation="true"> ([unrolling glossary](../../../../glossary.md#term-unrolling))</span>`[B,2]`, emissions `[4,B,2]`, dX `[4,B,3]`; fixed
   parameter gradients; unchanged code/routes/slots/topology, actual-sized resources.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [topology glossary](../../../../glossary.md#term-topology))</span>
8. EVERY call, even same B: fresh complete Compiler bind/private obligations, unused/tied inputs,
   all forward/reverse/latent ports, current input-backed descriptors/publications/storage. No memo.
   One RunState, fresh mutable packed/access/carry arrays/resources/result; numeric-only Runtime.
   No per-B prepare/recompile/cache, graph walk, route/topology choice, per-step allocation/callback,
   unsafe FFM, boxing/reflection/map/string dispatch or avoidable fallback overhead.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [memo glossary](../../../../glossary.md#cache-and-memoization); [RunState glossary](../../../../glossary.md#run-state--runstate); [mutable glossary](../../../../glossary.md#term-mutable); [Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [cache glossary](../../../../glossary.md#term-cache); [graph glossary](../../../../glossary.md#graph); [callback glossary](../../../../glossary.md#term-callback); [FFM glossary](../../../../glossary.md#term-ffm); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
9. Extend existing tests: B=0/1, separate fixed-T=0 and T=0/B=0, zero features, fixed-product/byte/
   aggregate overflow, raw ±0/NaN carry, poisoned padding/seeds, zero/mixed/full/invalid-last L,
   live/dead/shared histories, aliases/read views/write guards, cleanup/failure/retry/close/lease
   precedence, concurrent mixed-B/results. Empty Scan skips no other required work. Mean uses B×H,
   not sum(L): empty mean NaN, empty row gradients, nonempty typed-zero parameter sums. Retain held gates.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [views glossary](../../../../glossary.md#term-view-meanings); [lease glossary](../../../../glossary.md#term-lease))</span>
10. Separate combined [canonical](../../../../glossary.md#term-canonical-form) `:symbolic-length-v1` and baked `|scan-symbolic-length-v1=` in BOTH
    specialization streams, including dtype/roles/terminal/demand/control/numeric-pack/recipe facts;
    exclude actual B/L/bytes/slots/addresses. Added outer F64/control/new rank-three cases use<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
    `ordinary-batch-typed-v1` in IR and both specialization streams, including exact dtype/carrier/
    mapping/reduction-state facts; positive/negative validated-generation and artifact load/reuse/stale
    rejection tests cover the typed namespace. Old F32 <span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [carrier glossary](../../../../glossary.md#term-storage-carrier); [artifact glossary](../../../../glossary.md#term-artifact))</span>`ordinary-batch-v1`, schema/family71, static
    stagedv4, all-valid symbolic canonicalv1/bakedv3, static-length v1, fixed/ordinaryF32 keys/bytes stay identical.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>

## [Validation](../../../../glossary.md#term-validation)

Executed commands and outcomes are recorded in Result. Settings/builds map :[backends](../../../../glossary.md#backend):cpu to [modules](../../../../glossary.md#term-module)/backend/cpu.
Focused development extends [PrepareContext](../../../../glossary.md#prepare-context-preparecontext)/[GraphPreparation](../../../../glossary.md#graph-preparation--graphpreparation)/[BackendPartitionPreparer](../../../../glossary.md#backend-partition-preparer--backendpartitionpreparer) and EngineSymbolicScan
tests; use test filters, not repeated full suites. [CPU](../../../../glossary.md#term-cpu) focused example:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScan*Test' --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuOrdinaryDynamicBatchTest' -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=2 --console=plain
```

After focused checks and executable freeze, once: `./gradlew :modules:prepare:test :backends:cpu:test :modules:engine:test :modules:prepare:javadoc :backends:cpu:javadoc :modules:engine:javadoc -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=2 --console=plain`. MAIN owns the sole campaign below after independent structural review, avoiding measurement of code still under correction.

Existing optimal clean-Java/ClassFile oracles check changed hot paths only: combined loops/[binding](../../../../glossary.md#term-binding)
and [F64](../../../../glossary.md#floating-point-precision-and-rounding) exact outer reductions; algorithm/hot-loop/[dataflow](../../../../glossary.md#term-dataflow) and hidden calls/allocation/[dispatch](../../../../glossary.md#term-dispatch).
One added CpuGeneralScanTimingTest method, one campaign/[JVM](../../../../glossary.md#term-java-virtual-machine-jvm): fixed witness/B32/7, both typed forward/
actual reverse and new exact reductions, equal carriers/inputs/saved sites/preallocation. Separate
cold generation/classload/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit), entry/[preflight](../../../../glossary.md#term-preflight) and [kernel](../../../../glossary.md#kernel) costs; disclose [censoring](../../../../glossary.md#benchmark-measurement-vocabulary)/[variance](../../../../glossary.md#term-variance).
Budget: ≤60s wall including setup; per case 20 [warmup](../../../../glossary.md#term-warmup) pairs, then 3 rounds ×30 measured pairs,
100 calls/side/pair, alternating order. Stop admitting work at deadline; report overshoot/censoring
and non-steady-state limits. No warm-to-stability, extra [fork](../../../../glossary.md#term-fork-meanings) or retry campaign.
Executed sole method command: `./gradlew :backends:cpu:test --tests '*.CpuGeneralScanTimingTest.symbolicLengthForwardReversePrepare0015' -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-prepare0015.gVDqlx/performance -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain`.
No five-fork/full-[matrix](../../../../glossary.md#term-matrix-meanings), JIT-cause or parity/speed promise; reuse unchanged evidence.

MAIN runs ONE combined [conformance](../../../../glossary.md#term-conformance)/integration/architecture/root gate. Extend CpuGeneralScanConformanceTest/
CpuSymbolicScanConformanceTest, EngineCpuGeneralScanIntegrationTest/EngineCpuSymbolicScanIntegrationTest;
reuse architecture guards and successful worker tasks:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib ./gradlew test javadoc -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=2 --console=plain
```

Existing explicit native environment only, no rebuild/build/gating/skip edits; report blockers.
Reuse successes, no duplicate reviewer suites; rerun only for changed code/stale evidence/concrete risk.
Final [Javadoc](../../../../glossary.md#term-javadoc) follows comments. Check links/anchors/size/whitespace and baseline-relative [scope](../../../../glossary.md#term-scope)/bytes.

## Documentation and review impact

Phase 1 enacts gates, focused explanations, existing [ADR](../../../../glossary.md#term-architecture-decision-record-adr) amendments and glossary support status; no [Javadoc](../../../../glossary.md#term-javadoc)/Java/[API](../../../../glossary.md#term-application-programming-interface-api) change.
Final independent docs context updates implemented support/Javadocs/API examples and `docs/user-guide/running-models.md`
after executable acceptance. Targeted glossary: [symbolic Scan](../../../../glossary.md#symbolic-scan)/row role/[primal](../../../../glossary.md#term-primal) [binding](../../../../glossary.md#term-binding); meanings unchanged. MAIN integration.

## Result

Status: Complete. MAIN accepted 2026-10-27; active [Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu)/[Engine](../../../../glossary.md#term-engine) frontiers None.
Delivered one named B plus [runtime](../../../../glossary.md#term-runtime) [INT64](../../../../glossary.md#data-type--datatype)[B] lengths, fixed T/[features](../../../../glossary.md#term-software-feature), [homogeneous](../../../../glossary.md#term-homogeneous) [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) forward
and actual [Compiler](../../../../glossary.md#term-compiler)-generated first-order reverse through existing Prepare/CPU/Engine [owners](../../../../glossary.md#term-owner).
The same prepared [owner](../../../../glossary.md#term-owner) supports changed/repeated B/L, actual-sized resources and publications;
bounded ordinary F64/control/ROW(1) prerequisites are included. No new production type/[API](../../../../glossary.md#term-application-programming-interface-api)/protocol,
[dependency](../../../../glossary.md#term-dependency), session, [memo](../../../../glossary.md#cache-and-memoization), native change or production deletion. All ten acceptance criteria passed.

Changed: 19 production Java paths (four comment-only), 18 test Java paths (two new test-only types),
the current v2 [capability ledger](../../../../glossary.md#capability-ledger), and 21 Markdown paths. Independent code/structural review (Rawls)
and documentation/[Javadoc](../../../../glossary.md#term-javadoc) review (Plato) APPROVE, 0 P1/P2. Final R1 row-copy activity is once per row;
R2 F64 workspace uses eight-byte cells in both geometry and packing. Six core [Scan](../../../../glossary.md#term-cumulative-scan) semantic sections
are unchanged. Old v1 ledger/all 272 old v2 rows and 28 static-length key/both-stream/ClassFile cases
remain exact. Generated code was compared with typed optimal Java; 16/16 measured Scan ClassFiles
equal the structurally reviewed [artifacts](../../../../glossary.md#term-artifact). No hidden allocation/helper [dispatch](../../../../glossary.md#term-dispatch) or [activation](../../../../glossary.md#term-activation) replay
was found in the inspected generated hot loops.

[Validation](../../../../glossary.md#term-validation): owning command above exit 0, 1,524 executed/46 skipped (1,570 discovered), three Javadocs
PASS; focused [conformance](../../../../glossary.md#term-conformance) 23 and integration 24 PASS. MAIN's sole configured root command above
exit 0: 4,988 executed/57 skipped (5,045 discovered), zero failures/errors, all Javadocs PASS;
119 tasks, 47 executed/72 UP-TO-DATE. Existing optional diagnostics/performance/[observer](../../../../glossary.md#term-observer) skips were
retained, not bypassed. Root XML was archived before the filtered timing test replaced CPU reports.
No native rebuild or Git mutation. Final links/anchors/whitespace and baseline-relative [scope](../../../../glossary.md#term-scope) PASS;
HEAD/origin/index and the other 2,426 accepted files remain unchanged.

Sole timing campaign: exit 0, 24/24 complete cases, 20 [warmup](../../../../glossary.md#term-warmup) pairs + 3 rounds ×30 measured pairs,
100 calls/side/pair; 10.117149416 seconds, no [censoring](../../../../glossary.md#benchmark-measurement-vocabulary)/overshoot. Ratios below are paired medians
of generated entry plus [fixture](../../../../glossary.md#term-fixture) list/spreader/[MethodHandle](../../../../glossary.md#native-numerical-library-and-instruction-architecture) overhead versus a typed direct Java call;
both include Runnable/deadline-clock overhead. They are not isolated inner-loop measurements.

| Cases | Generated/direct paired-[median](../../../../glossary.md#term-median) range |
|---|---|
| B=7, forward, both [dtypes](../../../../glossary.md#data-type--datatype)/traversals | 0.921–1.137 |
| B=7, actual reverse, both dtypes/traversals | 1.344–1.462 |
| B=32, forward, both dtypes/traversals | 0.811–1.059 |
| B=32, actual reverse, both dtypes/traversals | 41.129–78.837 |
| F32 exact SUM/[MEAN](../../../../glossary.md#term-mean), both B | 1.091–46.116 |
| F64 exact SUM/MEAN, both B | 1.020–1.081 |

The large B=32 reverse gaps are real observations in this short series and remain unexplained.
Their round ratios change sharply; neither steady-state parity nor a [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) cause is established.
No slow [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) was discarded, [oracle](../../../../glossary.md#term-test-oracle) weakened, extra [fork](../../../../glossary.md#term-fork-meanings)/retry/profiling campaign [run](../../../../glossary.md#term-run), or speed
[qualification](../../../../glossary.md#term-qualification) inferred. A separate bounded diagnosis is needed before any performance-parity claim;
it is not an unimplemented acceptance item in this explicitly non-parity capability task.
Detailed evidence is retained in `/private/tmp/synaptik-prepare0015.gVDqlx/`: implementation [handoff](../../../../glossary.md#term-handoff),
classfile [manifest](../../../../glossary.md#term-manifest), root [log](../../../../glossary.md#term-log)/XML/counts, full timing report, `30-performance-summary.json` and
`31-measured-classfile-identity.json`. Final docs/examples/glossary support status and affected
Javadocs are synchronized; no further glossary meaning or Java behavior change was needed.
No required follow-up remains for this bounded capability. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training, cold/wire/copy, low/native,
variable T/[features](../../../../glossary.md#term-feature), higher order and all-valid symbolic F64 remain held; no successor is authorized.
