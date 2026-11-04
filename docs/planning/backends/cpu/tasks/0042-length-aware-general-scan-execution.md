# Task CPU0042: Length-aware [general Scan](../../../../glossary.md#general-differentiable-scan) execution

## Status

Complete — MAIN accepted 2026-10-26 after independent Class C code/docs APPROVE and recorded gates.
User “ok, pust se do toho” authorized this bounded implementation after approved [planning](../../../../glossary.md#term-planning)/[admission](../../../../glossary.md#term-admission).
PlanningAuthorized=true; ImplementationAuthorized=true records CPU0042 only; source/test frontier None.
No successor, further measurement, [commit](../../../../glossary.md#term-commit) or push is authorized.

## Change class

Class C — generated hot loops, mixed typed [binding](../../../../glossary.md#term-binding), [producer](../../../../glossary.md#term-producer)-ordered [validation](../../../../glossary.md#term-validation) and saved-state boundaries.

## Goal

Execute fixed T/B/[features](../../../../glossary.md#term-software-feature) length-aware [Scan](../../../../glossary.md#term-cumulative-scan) forward and actual [Compiler](../../../../glossary.md#term-compiler)-generated first-order
reverse in [FLOAT32](../../../../glossary.md#term-float32) AND [FLOAT64](../../../../glossary.md#term-float64): [compile](../../../../glossary.md#compile) → [prepare](../../../../glossary.md#prepare) → generated [CPU](../../../../glossary.md#term-cpu) loop. [Runtime](../../../../glossary.md#term-runtime) lengths change activity.

## [Scope](../../../../glossary.md#term-scope)

Qualify closed [tensor](../../../../glossary.md#tensor)/[scalar](../../../../glossary.md#term-scalar) ADD/SUB/MUL/DIV, NEG/[TANH](../../../../glossary.md#term-tanh)/[SIGMOID](../../../../glossary.md#term-sigmoid), [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[transpose](../../../../glossary.md#term-transpose),
SUM/sumToShape, [feature](../../../../glossary.md#term-feature)-only edits/[CONTIGUOUS](../../../../glossary.md#term-contiguous)/same-[dtype](../../../../glossary.md#data-type--datatype) CAST and actual reverse closure.
Preserve saved sites/demand, ROW(0)/rank-two ROW(1)/SHARED [bindings](../../../../glossary.md#term-binding) and terminal SHARED suffixes.
ROW identifies [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); SHARED has none even at F=B. [PURE](../../../../glossary.md#term-pure) validity is not [CPU](../../../../glossary.md#term-cpu) support.
Scalar single-range route; declare resources before slots.
No new production type/package, supported/public [API](../../../../glossary.md#term-application-programming-interface-api) or [Runtime](../../../../glossary.md#term-runtime) [handoff](../../../../glossary.md#term-handoff). Listed CPU-internal record
fields/constructors/methods may extend; retain legacy overloads/default keys and the segment + packed
long geometry/start/end entry [ABI](../../../../glossary.md#term-application-binary-interface-abi) with checked [argument](../../../../glossary.md#term-argument)/[layout](../../../../glossary.md#layout) budget.

## Non-goals

No dynamic-B + lengths, variable T/[features](../../../../glossary.md#term-feature), higher-order/createGraph, fixed [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/fluent migration,
cold copy/reconstruction, v5/[Checkpoint](../../../../glossary.md#term-checkpoint), [memo](../../../../glossary.md#cache-and-memoization), control flow, new executor/[Runtime](../../../../glossary.md#term-runtime) [API](../../../../glossary.md#term-application-programming-interface-api), packing/sorting/
compaction, low types, [vector](../../../../glossary.md#term-vector)/parallel/[OpenBLAS](../../../../glossary.md#term-openblas)/[Metal](../../../../glossary.md#term-metal), dependencies, numerics or benchmark expansion.
NN0032's second-order/CONCAT/mixed-[engine](../../../../glossary.md#term-engine) gates are neither dependencies nor waived.

## Contracts

- [Root](../../../../../ARCHITECTURE.md): `Scope-indexed normative contracts`, `Core lifecycle`, `Core invariants`.
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [owner glossary](../../../../glossary.md#term-owner))</span>:
  `Source body, attachments and scoped identities`, `Canonical residual ports and demand`,
  `Reverse scan and exact primal bindings`, `Traversal, empty domains and declared qualification`,
  `Qualification, performance and migration gates`; preserve the separately qualified symbolic branch.
- [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>,
  **all five subheadings**: `Header, bindings and source admission`, `Bounded row-role analysis`,
  `Activity, raw identity and retained histories`, `First-order length-mode reverse`,
  `Runtime preflight and implementation gates` — already accepted semantics.
- [Backend execution](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>:
  `Concrete backend modules`, `CPU backend routes`, `Performance evidence and optimization tooling`.
- [Runtime/Prepare/Engine](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>:
  `modules/runtime`, `modules/prepare`, `Prepare lifecycle`, `Run lifecycle`, `Public reusable prepared execution`.
No authority redesign. Update factual support status only at future completion. Stop for MAIN on
contract/[dtype](../../../../glossary.md#data-type--datatype) ambiguity, required new type/supported [API](../../../../glossary.md#term-application-programming-interface-api), [Runtime](../../../../glossary.md#term-runtime) [handoff](../../../../glossary.md#term-handoff) or global [schema](../../../../glossary.md#term-schema) bump.

## Dependencies and integration

- Depends on: Complete [Model0037](../../../modules/model/tasks/0037-length-aware-general-scan-authority.md#result),
  [Model0038](../../../modules/model/tasks/0038-length-aware-scan-source-foundation.md#result),
  [Compiler0013](../../../modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result),
  [Prepare0011](../../../modules/prepare/tasks/0011-length-aware-scan-projection-and-resource-handoff.md#result),
  [CPU0033](0033-general-scan-f64-domain-closure.md#result), [CPU0037](0037-general-scan-dense-read-views.md#result),
  [CPU0041](0041-symbolic-scan-reverse-carrier-specialization.md#result),
  [Prepare0014](../../../modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result).
- Conflicts with: [scan](../../../../glossary.md#term-cumulative-scan) [admission](../../../../glossary.md#term-admission)/[binding](../../../../glossary.md#term-binding)/resources/[IR](../../../../glossary.md#term-intermediate-representation-ir)/emission/[identity](../../../../glossary.md#term-identity-meanings), ledger and shared docs; serialize.
- Parallel group: None; serial implementation then independent review.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` + accepted dirty work;
  2477-file `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/cpu0042-implementation-baseline.json` and `-files`.
- Integration order: approved plan + user implementation approval + MAIN admission → clean CPU
  implementation → independent Class C code/docs review → MAIN checkpoint.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Integration [validation](../../../../glossary.md#term-validation): named [checkpoint](../../../../glossary.md#term-checkpoint) below once; [CI](../../../../glossary.md#term-continuous-integration-ci) at next join, no extra root/native campaign.
- Shared-document integration owner: MAIN; final docs/status pass exclusively delegated to the
  independent reviewer after source freeze and MAIN's named checkpoint/sole probe.
MAIN launch verification 2026-10-26: Ready at CPU0042-only frontier; all eight dependencies Complete
with final reviewed/gated Results; contracts/source unchanged. All earlier agents closed; no active
conflicting writer/agent build per MAIN; global process inventory unavailable in sandbox.
Current 2477-file snapshot matches; versus 2476-file planning baseline only four approved docs changed.
HEAD=origin/main unchanged, index empty. Other frontiers unchanged; no automatic successor or commit/push.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Ready glossary](../../../../glossary.md#term-ready); [snapshot glossary](../../../../glossary.md#term-snapshot); [planning glossary](../../../../glossary.md#term-planning); [commit glossary](../../../../glossary.md#term-commit))</span>

## Files and symbols

[CPU](../../../../glossary.md#term-cpu) paths below are relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.

- `CpuCapabilityProvider.java` [dispatch](../../../../glossary.md#term-dispatch); `internal/lowering/CpuGeneralScanLowering.java` `analyze/lower`,
  `Facts/Geometry`, `shapedResources`: row/terminal proof, [aliases](../../../../glossary.md#aliasing-and-arena), typed spans/demand/storage/resources.
- `internal/lowering/CpuPartitionLowering.java` `LoweredPartition(Facts, boundaries)` currently uses
  homogeneous <span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous))</span>`nCopies`; preserve [INT64](../../../../glossary.md#data-type--datatype) through selection/[prepare](../../../../glossary.md#prepare) transport.
- `internal/ir/CpuGeneralScanIr.java` constructor/[cardinality](../../../../glossary.md#term-cardinality) checks, `encodedKernelIr`, `Stage/View/Stages`,
  `canonicalStages`, `bakedSpecializationIdentity`: encode INT64 control separately from homogeneous
  arithmetic; formal-only mappings plus explicit length read position and row/terminal/demand facts.
- `internal/prepare/{CpuPartitionPreparer,CpuPartitionPreparationPlan,CpuPartitionFinalizer}.java`:
  `analyzeGeneralScan` one-width declarations, unit/[runtime](../../../../glossary.md#term-runtime) checks, child [dtype](../../../../glossary.md#data-type--datatype)/size/[alignment](../../../../glossary.md#term-alignment)/write offsets.
- `internal/cache/CpuKernelSpecialization.java` generalScanIr constructor guards, `compatibilityBytes`,
  `classIdentityBytes`; `internal/codegen/emit/CpuClassFileKernelGenerator.java` [scan](../../../../glossary.md#term-cumulative-scan) path and
  `CpuCarrierEmitter.prepareSegmentLayouts`: checked all-segment [layout](../../../../glossary.md#layout)/[argument](../../../../glossary.md#term-argument)-slot budget.
- `internal/executable/CpuGeneralScanPreparedExecutable.java` constructor, accepts/bindCompatible,
  `Invocation.executeBound`: separate typed spans/slicing/read count; after-[producer](../../../../glossary.md#term-producer) value [preflight](../../../../glossary.md#term-preflight).
  `CpuPreparedPartitionExecutable.bindCompatible/executeBound`: bind-all then ordered children, no new callbacks.
- `internal/codegen/emit/CpuGeneralScanEmitter.java` entry/`Shaped.emit`, bind/publish/load/store;
  `CpuGeneralScan{Reduction,Matmul}Emitter.emit`: typed activity/hoist/contributor/history loops.
- Read-only [Compiler](../../../../glossary.md#term-compiler) `ScanBodyInference.validateRows`, `ScanGradientRules.preflight/apply`; Prepare
  `PrepareContext.validateScan`, `PrepareContextTest` length and `GraphPreparationTest` actual-length-reverse
  fixtures. No upstream production change; fixed recurrent Invocation is ordering evidence only.
- Extend existing CPU tests `CpuGeneralScan{Capability,Lowering,Preparation,Execution,GeneratedKernel}Test`,
  `CpuGeneralScanCleanJavaOracle` and one opt-in method in `CpuGeneralScanTimingTest`; no new test framework.
  Extend named conformance/Engine tests below and <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>`LowPrecisionCapabilityLedgerTest`, its [conformance](../../../../glossary.md#term-conformance)
  `withType` helper and `testing/backend-conformance/src/test/resources/low-precision-capability-ledger-v2.tsv`.

## Acceptance criteria

1. Qualify F32/F64 forward AND actual Compiler siblings independently. Original AD requires original
   carries/sequences ROW(0), not generated mixed-role reverse. Unsupported bodies reject even dead/empty.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [AD glossary](../../../../glossary.md#autograd))</span>
2. Outer actuals are `c+s+i+1`, local formals `c+s+i`: no fake lengths formal/Tensor, arithmetic
   ordinal, floating scalar or carry. Distinguish formal mapping, unique read count and first write
   after deduplication; lengths are INT64 <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [scalar glossary](../../../../glossary.md#term-scalar); [carry glossary](../../../../glossary.md#term-carry); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>`[B]`, eight-byte span/alignment, not body width. Retain
   canonical/dense-view/scalar floating reads and canonical writes.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [view glossary](../../../../glossary.md#term-view-meanings))</span>
3. After length producers, in scan invocation execution before generated entry or any scan write,
   recheck typed accessibility/span/alignment/overlap and all B values in <span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>`[0,T]`. Invalid last length
   leaves scan output/padding/history/scratch sentinels unchanged; no upstream rollback. Test same-partition
   and separate-schedule producers, invalid→valid retry, T=0/B=0, heap/segment/mixed, aliases/write overlap.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [rollback glossary](../../../../glossary.md#term-rollback); [partition glossary](../../../../glossary.md#partition); [heap glossary](../../../../glossary.md#term-heap); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
4. `active(t,b)=t<L[b]` dominates padded sequence/seed/row-history loads and dependent arithmetic,
   including fused SUM producers/both MATMUL operands. No compute-then-mask/0*x. Inactive carry is
   raw identity; emission/dSequence +0. Empty-row final/dInitialCarry preserve initial/final seed bits.
   Both traversals retain original indices/increasing-b contributors.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [fused glossary](../../../../glossary.md#specialization-and-fusion); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [mask glossary](../../../../glossary.md#term-boolean-mask); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
5. Previous carry saves precede active work; produced primals save at exact represented sites, no
   replay. ROW(0)/ROW(1)/SHARED histories retain `[T]+localShape`, inactive cells +0, no added B for
   SHARED. Test F=B, shared transpose, ROW(1) pruning, tied roots/positions, joint/missing seeds,
   false-flag carry propagation and cross-occurrence sums.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
6. Shared/hoisted work has any-active/active-time guards: all-zero L, T=0 and B=0 do no body work.
   Contributions start at typed zero, reduce active b in order, then ONE distinct shared carry ADD
   per time; inactive time is raw identity, not old+0. O(P) carry/contribution scratch, no B×P bank;
   demanded O(T×P) shared history explicit. Empty feature/K/first-carry domains cannot suppress required
   nonempty results. Test poisoned padding/seeds, raw signed-zero/NaN/subnormal state, instrumented
   inactive no-load/no-arithmetic and ordinary active specials.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-software-feature); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [subnormal glossary](../../../../glossary.md#term-subnormal))</span>
7. One fixed-geometry preparation reuses code/slots for changing L; fresh runs/results/concurrency
   remain isolated. No per-L specialization/reprepare or T-fold code expansion. Planned example:
   T=5, L=[5,3,0] executes 8 row transitions; reverse visits each valid prefix only, never padded tails.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [specialization glossary](../../../../glossary.md#term-specialization))</span>
8. Add length-only [canonical](../../../../glossary.md#term-canonical-form) `:length-aware-v1` and baked `|scan-length-aware-v1=` branches, before
   old static/symbolic dispatch. Bind ordered dtype/control/formal/actual/row/terminal/demand/storage
   facts and every baked geometry fact in BOTH existing specialization streams; exclude L values,
   assigned slots/addresses; keep canonical geometry-free. No ScanAttrs diagnostic text identity.
   Preserve exact legacy Stage/View/Stages serialization/projections: <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [dtype glossary](../../../../glossary.md#data-type--datatype); [serialization glossary](../../../../glossary.md#term-serialization))</span>`bakedSpecializationIdentity`
   currently includes Stages.toString; new role/components must not leak into old static v4/symbolic v3
   streams. Retain schema/family71, old all-valid/pointwise/fixed/ordinary keys AND generated bytes;
   regression witnesses compare both streams/bytes and reject cross-mode/stale length artifacts.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [pointwise glossary](../../../../glossary.md#term-pointwise); [regression glossary](../../../../glossary.md#term-regression); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
9. Add representative [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) length provider/v2-ledger witnesses and independent [BF16](../../../../glossary.md#floating-point-precision-and-rounding)/[FP16](../../../../glossary.md#term-fp16) negatives;
   `withType` preserves INT64 lengths. Preserve v1/old v2 rows; adapt projection/count checks only for
   additions, no broad regeneration. Low negatives reject body dtype, not a mistakenly converted length.

## [Validation](../../../../glossary.md#term-validation)

Recorded worker commands: focused, then final [CPU](../../../../glossary.md#term-cpu)/[Javadoc](../../../../glossary.md#term-javadoc) once. The worker also supplied
`-Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0042-implementation.W3lMdM`:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScan*Test' --tests '*CpuKernelSpecializationTest' -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false
./gradlew :backends:cpu:test :backends:cpu:javadoc -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false
```

Reuse successful evidence in review. Automated Class-File checks compare optimal specialized clean Java
algorithm/hot-loop/[dataflow](../../../../glossary.md#term-dataflow), branch dominance, saved stores and hidden helpers/allocation/boxing/reflection/
[dispatch](../../../../glossary.md#term-dispatch) for both [dtypes](../../../../glossary.md#data-type--datatype)/actual reverse. No formal numerical-proof/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit)-tier gate. MAIN's one named join:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest' --tests '*CpuSymbolicScanConformanceTest' --tests '*CpuRecurrentScanConformanceTest' :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuSymbolicScanIntegrationTest' --tests '*EngineCpuRecurrentScanIntegrationTest' -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false
```

MAIN owns ONE fresh-[JVM](../../../../glossary.md#term-java-virtual-machine-jvm) observational probe, new method in existing CpuGeneralScanTimingTest:

```bash
./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanTimingTest.lengthAwareForwardReverseCpu0042' -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0042-probe
```

Fixed T5/B3/F8/H8, L=[5,3,0], [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) forward/reverse; matched segments/[layout](../../../../glossary.md#layout)/[ABI](../../../../glossary.md#term-application-binary-interface-abi)/demand/scratch/sites
versus specialized clean Java. [Warmup](../../../../glossary.md#term-warmup): 20 pairs ×100 =2000 calls/side/case; measured: 10×100 =1000.
Alternate sides/check consumed outputs outside timing. System.nanoTime [admission](../../../../glossary.md#term-admission) deadline: 30s for the
entire [JUnit](../../../../glossary.md#term-junit) body including setup/cold/timing/checks; calls are nonpreemptive, report overshoot/censorship
honestly, no replacement/[fork](../../../../glossary.md#term-fork-meanings) expansion. Separate generation/definition/first-call, [binding](../../../../glossary.md#term-binding)/[preflight](../../../../glossary.md#term-preflight)
and [kernel](../../../../glossary.md#kernel) costs. No parity threshold/promise, forced tiers, five-fork [matrix](../../../../glossary.md#term-matrix-meanings) or further campaign.

## Documentation and review impact

Executor drafted affected [Javadoc](../../../../glossary.md#term-javadoc)/[CPU](../../../../glossary.md#term-cpu) guide. MAIN's independent Class C reviewer finalized
explanations, factual root/scoped support status, glossary, links and Javadoc, without changing
semantics. Applied documentation rules and general/architecture/[backend](../../../../glossary.md#backend)/[API](../../../../glossary.md#term-application-programming-interface-api)/[planning](../../../../glossary.md#term-planning) profiles.
MAIN owns shared integration and delegated this final docs/status pass exclusively after source freeze.
Size exception: retain readable [atomic](../../../../glossary.md#resource-transaction-and-rollback) [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) forward/actual-reverse [qualification](../../../../glossary.md#term-qualification) and preservation
criteria in one brief (>200 lines/15 KB, below 25 KB), with a compact Result rather than split gates.

## Result

- MAIN accepted 2026-10-26 after Arendt the 4th's final independent Class C source/docs/evidence APPROVE; zero open P1/P2.
- Delivered fixed-geometry F32/F64 forward/actual reverse, mixed INT64 preflight, activity/saved sites and length-only identity.
  R1 fusion-domain and R2 SUM-epilogue availability fixes have both-dtype/traversal regressions; legacy proofs retained.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [INT64 glossary](../../../../glossary.md#data-type--datatype); [preflight glossary](../../../../glossary.md#term-preflight); [identity glossary](../../../../glossary.md#term-identity-meanings); [fusion glossary](../../../../glossary.md#term-fusion); [epilogue glossary](../../../../glossary.md#term-epilogue); [dtype glossary](../../../../glossary.md#data-type--datatype))</span>
- Implementation: 9 CPU production, 7 CPU test/oracle and 4 conformance/ledger/Engine paths, plus final docs/Javadoc/status.
  No new type/API/Runtime handoff, dependency/build or upstream executable change; upstream Scan Javadoc is factual only.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [oracle glossary](../../../../glossary.md#term-test-oracle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Runtime glossary](../../../../glossary.md#term-runtime); [handoff glossary](../../../../glossary.md#term-handoff); [dependency glossary](../../../../glossary.md#term-dependency); [Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Focused command above: 174 = 166 passed + 8 timing skips. Final CPU: 222 suites, 1308 = 1266 passed + 42 skips; zero failures/errors.
  Final `:modules:model:javadoc :backends:cpu:javadoc` PASS with the same 94 pre-existing warnings; tests reused after comments only.
- MAIN's exact named join above PASS: 4 [conformance](../../../../glossary.md#term-conformance) suites/23 tests + 3 [Engine](../../../../glossary.md#term-engine) suites/27 tests; zero skips/failures/errors.
- Twenty representative compiled-baseline witnesses retain canonical identity, BOTH specialization streams and class bytes exactly; not exhaustive.
  Class-File/clean-Java checks cover both dtypes/actual reverse; all-zero activity does no body work, L=[5,3,0] executes 8 row transitions.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- MAIN's SOLE probe above PASS: four T5/B3/F8/H8 cases, 2000 warmup + 1000 measured calls per side/case; cold/binding costs separate.
  Body 600105250 ns, XML 0.618 s; censored=false, overshoot=0. Both sides include deadline-clock overhead.
  Pairs are nonstationary: no stable ratio, steady-state parity, universal speed or confirmed JIT-tier cause; no extra measurement.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [binding glossary](../../../../glossary.md#term-binding); [censored glossary](../../../../glossary.md#term-censored); [nonstationary glossary](../../../../glossary.md#term-nonstationary); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>
- [Worker evidence](/private/tmp/synaptik-cpu0042-implementation.W3lMdM/completion.md): baseline-relative changes, freeze, archived XML and structure.
  [MAIN evidence](/private/tmp/synaptik-cpu0042-main.Rlim1c/result.txt): joined/probe [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) and archived XML; [raw pairs](/private/tmp/synaptik-cpu0042-probe/length-aware-report.txt).
  [Final docs review](/private/tmp/synaptik-cpu0042-main.Rlim1c/final-docs-review.txt): [Javadoc](../../../../glossary.md#term-javadoc), links/anchors/whitespace, glossary and executable-neutral proof.
- Root/scoped status, explanations/API, CPU guide and trackers agree; existing glossary terms retained, no semantic/ADR change.
  Ordinary worker runs kept all performance flags false; only MAIN's one named probe enabled its explicit flag. No root/native rebuild.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Symbolic lengths, cold/v5, owned [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), higher-order and low/native/[vector](../../../../glossary.md#term-vector)/parallel routes stay excluded; no unresolved issue or follow-up.

Status: Complete
Source/test frontier: None; no automatic successor, campaign, [commit](../../../../glossary.md#term-commit) or push.
