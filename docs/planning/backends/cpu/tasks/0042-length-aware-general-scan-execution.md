# Task CPU0042: Length-aware general Scan execution

## Status

Complete — MAIN accepted 2026-10-26 after independent Class C code/docs APPROVE and recorded gates.
User “ok, pust se do toho” authorized this bounded implementation after approved planning/admission.
PlanningAuthorized=true; ImplementationAuthorized=true records CPU0042 only; source/test frontier None.
No successor, further measurement, commit or push is authorized.

## Change class

Class C — generated hot loops, mixed typed binding, producer-ordered validation and saved-state boundaries.

## Goal

Execute fixed T/B/features length-aware Scan forward and actual Compiler-generated first-order
reverse in FLOAT32 AND FLOAT64: compile → prepare → generated CPU loop. Runtime lengths change activity.

## Scope

Qualify closed tensor/scalar ADD/SUB/MUL/DIV, NEG/TANH/SIGMOID, rank-two MATMUL/transpose,
SUM/sumToShape, feature-only edits/CONTIGUOUS/same-dtype CAST and actual reverse closure.
Preserve saved sites/demand, ROW(0)/rank-two ROW(1)/SHARED bindings and terminal SHARED suffixes.
ROW identifies batch; SHARED has none even at F=B. PURE validity is not CPU support.
Scalar single-range route; declare resources before slots.
No new production type/package, supported/public API or Runtime handoff. Listed CPU-internal record
fields/constructors/methods may extend; retain legacy overloads/default keys and the segment + packed
long geometry/start/end entry ABI with checked argument/layout budget.

## Non-goals

No dynamic-B + lengths, variable T/features, higher-order/createGraph, fixed BPTT, NN/fluent migration,
cold copy/reconstruction, v5/Checkpoint, memo, control flow, new executor/Runtime API, packing/sorting/
compaction, low types, vector/parallel/OpenBLAS/Metal, dependencies, numerics or benchmark expansion.
NN0032's second-order/CONCAT/mixed-engine gates are neither dependencies nor waived.

## Contracts

- [Root](../../../../../ARCHITECTURE.md): `Scope-indexed normative contracts`, `Core lifecycle`, `Core invariants`.
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  `Source body, attachments and scoped identities`, `Canonical residual ports and demand`,
  `Reverse scan and exact primal bindings`, `Traversal, empty domains and declared qualification`,
  `Qualification, performance and migration gates`; preserve the separately qualified symbolic branch.
- [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan),
  **all five subheadings**: `Header, bindings and source admission`, `Bounded row-role analysis`,
  `Activity, raw identity and retained histories`, `First-order length-mode reverse`,
  `Runtime preflight and implementation gates` — already accepted semantics.
- [Backend execution](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes):
  `Concrete backend modules`, `CPU backend routes`, `Performance evidence and optimization tooling`.
- [Runtime/Prepare/Engine](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime):
  `modules/runtime`, `modules/prepare`, `Prepare lifecycle`, `Run lifecycle`, `Public reusable prepared execution`.
No authority redesign. Update factual support status only at future completion. Stop for MAIN on
contract/dtype ambiguity, required new type/supported API, Runtime handoff or global schema bump.

## Dependencies and integration

- Depends on: Complete [Model0037](../../../modules/model/tasks/0037-length-aware-general-scan-authority.md#result),
  [Model0038](../../../modules/model/tasks/0038-length-aware-scan-source-foundation.md#result),
  [Compiler0013](../../../modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result),
  [Prepare0011](../../../modules/prepare/tasks/0011-length-aware-scan-projection-and-resource-handoff.md#result),
  [CPU0033](0033-general-scan-f64-domain-closure.md#result), [CPU0037](0037-general-scan-dense-read-views.md#result),
  [CPU0041](0041-symbolic-scan-reverse-carrier-specialization.md#result),
  [Prepare0014](../../../modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result).
- Conflicts with: scan admission/binding/resources/IR/emission/identity, ledger and shared docs; serialize.
- Parallel group: None; serial implementation then independent review.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` + accepted dirty work;
  2477-file `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/cpu0042-implementation-baseline.json` and `-files`.
- Integration order: approved plan + user implementation approval + MAIN admission → clean CPU
  implementation → independent Class C code/docs review → MAIN checkpoint.
- Integration validation: named checkpoint below once; CI at next join, no extra root/native campaign.
- Shared-document integration owner: MAIN; final docs/status pass exclusively delegated to the
  independent reviewer after source freeze and MAIN's named checkpoint/sole probe.
MAIN launch verification 2026-10-26: Ready at CPU0042-only frontier; all eight dependencies Complete
with final reviewed/gated Results; contracts/source unchanged. All earlier agents closed; no active
conflicting writer/agent build per MAIN; global process inventory unavailable in sandbox.
Current 2477-file snapshot matches; versus 2476-file planning baseline only four approved docs changed.
HEAD=origin/main unchanged, index empty. Other frontiers unchanged; no automatic successor or commit/push.

## Files and symbols

CPU paths below are relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.

- `CpuCapabilityProvider.java` dispatch; `internal/lowering/CpuGeneralScanLowering.java` `analyze/lower`,
  `Facts/Geometry`, `shapedResources`: row/terminal proof, aliases, typed spans/demand/storage/resources.
- `internal/lowering/CpuPartitionLowering.java` `LoweredPartition(Facts, boundaries)` currently uses
  homogeneous `nCopies`; preserve INT64 through selection/prepare transport.
- `internal/ir/CpuGeneralScanIr.java` constructor/cardinality checks, `encodedKernelIr`, `Stage/View/Stages`,
  `canonicalStages`, `bakedSpecializationIdentity`: encode INT64 control separately from homogeneous
  arithmetic; formal-only mappings plus explicit length read position and row/terminal/demand facts.
- `internal/prepare/{CpuPartitionPreparer,CpuPartitionPreparationPlan,CpuPartitionFinalizer}.java`:
  `analyzeGeneralScan` one-width declarations, unit/runtime checks, child dtype/size/alignment/write offsets.
- `internal/cache/CpuKernelSpecialization.java` generalScanIr constructor guards, `compatibilityBytes`,
  `classIdentityBytes`; `internal/codegen/emit/CpuClassFileKernelGenerator.java` scan path and
  `CpuCarrierEmitter.prepareSegmentLayouts`: checked all-segment layout/argument-slot budget.
- `internal/executable/CpuGeneralScanPreparedExecutable.java` constructor, accepts/bindCompatible,
  `Invocation.executeBound`: separate typed spans/slicing/read count; after-producer value preflight.
  `CpuPreparedPartitionExecutable.bindCompatible/executeBound`: bind-all then ordered children, no new callbacks.
- `internal/codegen/emit/CpuGeneralScanEmitter.java` entry/`Shaped.emit`, bind/publish/load/store;
  `CpuGeneralScan{Reduction,Matmul}Emitter.emit`: typed activity/hoist/contributor/history loops.
- Read-only Compiler `ScanBodyInference.validateRows`, `ScanGradientRules.preflight/apply`; Prepare
  `PrepareContext.validateScan`, `PrepareContextTest` length and `GraphPreparationTest` actual-length-reverse
  fixtures. No upstream production change; fixed recurrent Invocation is ordering evidence only.
- Extend existing CPU tests `CpuGeneralScan{Capability,Lowering,Preparation,Execution,GeneratedKernel}Test`,
  `CpuGeneralScanCleanJavaOracle` and one opt-in method in `CpuGeneralScanTimingTest`; no new test framework.
  Extend named conformance/Engine tests below and `LowPrecisionCapabilityLedgerTest`, its conformance
  `withType` helper and `testing/backend-conformance/src/test/resources/low-precision-capability-ledger-v2.tsv`.

## Acceptance criteria

1. Qualify F32/F64 forward AND actual Compiler siblings independently. Original AD requires original
   carries/sequences ROW(0), not generated mixed-role reverse. Unsupported bodies reject even dead/empty.
2. Outer actuals are `c+s+i+1`, local formals `c+s+i`: no fake lengths formal/Tensor, arithmetic
   ordinal, floating scalar or carry. Distinguish formal mapping, unique read count and first write
   after deduplication; lengths are INT64 `[B]`, eight-byte span/alignment, not body width. Retain
   canonical/dense-view/scalar floating reads and canonical writes.
3. After length producers, in scan invocation execution before generated entry or any scan write,
   recheck typed accessibility/span/alignment/overlap and all B values in `[0,T]`. Invalid last length
   leaves scan output/padding/history/scratch sentinels unchanged; no upstream rollback. Test same-partition
   and separate-schedule producers, invalid→valid retry, T=0/B=0, heap/segment/mixed, aliases/write overlap.
4. `active(t,b)=t<L[b]` dominates padded sequence/seed/row-history loads and dependent arithmetic,
   including fused SUM producers/both MATMUL operands. No compute-then-mask/0*x. Inactive carry is
   raw identity; emission/dSequence +0. Empty-row final/dInitialCarry preserve initial/final seed bits.
   Both traversals retain original indices/increasing-b contributors.
5. Previous carry saves precede active work; produced primals save at exact represented sites, no
   replay. ROW(0)/ROW(1)/SHARED histories retain `[T]+localShape`, inactive cells +0, no added B for
   SHARED. Test F=B, shared transpose, ROW(1) pruning, tied roots/positions, joint/missing seeds,
   false-flag carry propagation and cross-occurrence sums.
6. Shared/hoisted work has any-active/active-time guards: all-zero L, T=0 and B=0 do no body work.
   Contributions start at typed zero, reduce active b in order, then ONE distinct shared carry ADD
   per time; inactive time is raw identity, not old+0. O(P) carry/contribution scratch, no B×P bank;
   demanded O(T×P) shared history explicit. Empty feature/K/first-carry domains cannot suppress required
   nonempty results. Test poisoned padding/seeds, raw signed-zero/NaN/subnormal state, instrumented
   inactive no-load/no-arithmetic and ordinary active specials.
7. One fixed-geometry preparation reuses code/slots for changing L; fresh runs/results/concurrency
   remain isolated. No per-L specialization/reprepare or T-fold code expansion. Planned example:
   T=5, L=[5,3,0] executes 8 row transitions; reverse visits each valid prefix only, never padded tails.
8. Add length-only canonical `:length-aware-v1` and baked `|scan-length-aware-v1=` branches, before
   old static/symbolic dispatch. Bind ordered dtype/control/formal/actual/row/terminal/demand/storage
   facts and every baked geometry fact in BOTH existing specialization streams; exclude L values,
   assigned slots/addresses; keep canonical geometry-free. No ScanAttrs diagnostic text identity.
   Preserve exact legacy Stage/View/Stages serialization/projections: `bakedSpecializationIdentity`
   currently includes Stages.toString; new role/components must not leak into old static v4/symbolic v3
   streams. Retain schema/family71, old all-valid/pointwise/fixed/ordinary keys AND generated bytes;
   regression witnesses compare both streams/bytes and reject cross-mode/stale length artifacts.
9. Add representative F32/F64 length provider/v2-ledger witnesses and independent BF16/FP16 negatives;
   `withType` preserves INT64 lengths. Preserve v1/old v2 rows; adapt projection/count checks only for
   additions, no broad regeneration. Low negatives reject body dtype, not a mistakenly converted length.

## Validation

Recorded worker commands: focused, then final CPU/Javadoc once. The worker also supplied
`-Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0042-implementation.W3lMdM`:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScan*Test' --tests '*CpuKernelSpecializationTest' -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false
./gradlew :backends:cpu:test :backends:cpu:javadoc -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false
```

Reuse successful evidence in review. Automated Class-File checks compare optimal specialized clean Java
algorithm/hot-loop/dataflow, branch dominance, saved stores and hidden helpers/allocation/boxing/reflection/
dispatch for both dtypes/actual reverse. No formal numerical-proof/JIT-tier gate. MAIN's one named join:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest' --tests '*CpuSymbolicScanConformanceTest' --tests '*CpuRecurrentScanConformanceTest' :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuSymbolicScanIntegrationTest' --tests '*EngineCpuRecurrentScanIntegrationTest' -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false
```

MAIN owns ONE fresh-JVM observational probe, new method in existing CpuGeneralScanTimingTest:

```bash
./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanTimingTest.lengthAwareForwardReverseCpu0042' -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0042-probe
```

Fixed T5/B3/F8/H8, L=[5,3,0], F32/F64 forward/reverse; matched segments/layout/ABI/demand/scratch/sites
versus specialized clean Java. Warmup: 20 pairs ×100 =2000 calls/side/case; measured: 10×100 =1000.
Alternate sides/check consumed outputs outside timing. System.nanoTime admission deadline: 30s for the
entire JUnit body including setup/cold/timing/checks; calls are nonpreemptive, report overshoot/censorship
honestly, no replacement/fork expansion. Separate generation/definition/first-call, binding/preflight
and kernel costs. No parity threshold/promise, forced tiers, five-fork matrix or further campaign.

## Documentation and review impact

Executor drafted affected Javadoc/CPU guide. MAIN's independent Class C reviewer finalized
explanations, factual root/scoped support status, glossary, links and Javadoc, without changing
semantics. Applied documentation rules and general/architecture/backend/API/planning profiles.
MAIN owns shared integration and delegated this final docs/status pass exclusively after source freeze.
Size exception: retain readable atomic F32/F64 forward/actual-reverse qualification and preservation
criteria in one brief (>200 lines/15 KB, below 25 KB), with a compact Result rather than split gates.

## Result

- MAIN accepted 2026-10-26 after Arendt the 4th's final independent Class C source/docs/evidence APPROVE; zero open P1/P2.
- Delivered fixed-geometry F32/F64 forward/actual reverse, mixed INT64 preflight, activity/saved sites and length-only identity.
  R1 fusion-domain and R2 SUM-epilogue availability fixes have both-dtype/traversal regressions; legacy proofs retained.
- Implementation: 9 CPU production, 7 CPU test/oracle and 4 conformance/ledger/Engine paths, plus final docs/Javadoc/status.
  No new type/API/Runtime handoff, dependency/build or upstream executable change; upstream Scan Javadoc is factual only.
- Focused command above: 174 = 166 passed + 8 timing skips. Final CPU: 222 suites, 1308 = 1266 passed + 42 skips; zero failures/errors.
  Final `:modules:model:javadoc :backends:cpu:javadoc` PASS with the same 94 pre-existing warnings; tests reused after comments only.
- MAIN's exact named join above PASS: 4 conformance suites/23 tests + 3 Engine suites/27 tests; zero skips/failures/errors.
- Twenty representative compiled-baseline witnesses retain canonical identity, BOTH specialization streams and class bytes exactly; not exhaustive.
  Class-File/clean-Java checks cover both dtypes/actual reverse; all-zero activity does no body work, L=[5,3,0] executes 8 row transitions.
- MAIN's SOLE probe above PASS: four T5/B3/F8/H8 cases, 2000 warmup + 1000 measured calls per side/case; cold/binding costs separate.
  Body 600105250 ns, XML 0.618 s; censored=false, overshoot=0. Both sides include deadline-clock overhead.
  Pairs are nonstationary: no stable ratio, steady-state parity, universal speed or confirmed JIT-tier cause; no extra measurement.
- [Worker evidence](/private/tmp/synaptik-cpu0042-implementation.W3lMdM/completion.md): baseline-relative changes, freeze, archived XML and structure.
  [MAIN evidence](/private/tmp/synaptik-cpu0042-main.Rlim1c/result.txt): joined/probe logs and archived XML; [raw pairs](/private/tmp/synaptik-cpu0042-probe/length-aware-report.txt).
  [Final docs review](/private/tmp/synaptik-cpu0042-main.Rlim1c/final-docs-review.txt): Javadoc, links/anchors/whitespace, glossary and executable-neutral proof.
- Root/scoped status, explanations/API, CPU guide and trackers agree; existing glossary terms retained, no semantic/ADR change.
  Ordinary worker runs kept all performance flags false; only MAIN's one named probe enabled its explicit flag. No root/native rebuild.
- Symbolic lengths, cold/v5, owned NN, higher-order and low/native/vector/parallel routes stay excluded; no unresolved issue or follow-up.

Status: Complete
Source/test frontier: None; no automatic successor, campaign, commit or push.
