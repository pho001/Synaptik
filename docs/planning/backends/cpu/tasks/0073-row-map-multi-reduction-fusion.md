# Task CPU0073: Row-map and multiple-terminal reduction fusion

## Status

Complete — MAIN accepted implementation, Hooke's final code/tests/Javadoc/evidence-transfer
APPROVE (zero open findings) and the named join on 2026-10-30; independent documentation closure
is recorded below. User approval and Boole's authority APPROVE preceded the verified Ready receipt.
The approved [proposal](../row-map-multi-reduction-fusion-plan.md) pre-edit SHA-256 is
`a1f510c71aa14eef07b56fc48bde736c8dbf1079467c5142f8c69a457a37132b`.
Source/measurement frontiers: None; old budgets closed, no commit/push or automatic successor.

## Change class

Class C — generated hot-loop/dataflow, materialization/lifetime and artifact identity changes.
Authority prerequisite, clean implementation and independent code/documentation contexts are complete.

## Goal

Deliver one general CPU lowering capability: share a pure ordered row map, including bounded
feature contractions, among row outputs and multiple terminal contractions/represented sums.
Ordinary graphs get one lowered execution unit and fusion emitter; Scan gets a generation-time
region in its existing unit/time loop, never a per-step Runtime unit, callback, worker or Job.
Reuse existing pointwise fusion and operation math; introduce no Tensor semantic operation.

## Scope

Use the proposal's [admission matrix](../row-map-multi-reduction-fusion-plan.md#v1-admission-and-exclusions),
[safety/range proof](../row-map-multi-reduction-fusion-plan.md#legality-resources-and-range-proof),
[exact type/seam map](../row-map-multi-reduction-fusion-plan.md#minimal-proposed-production-map-and-signatures)
and [identity recommendation](../row-map-multi-reduction-fusion-plan.md#compatibility-and-retirement-recommendation).
One cohesive capability covers ordinary static F32/F64, already-admitted symbolic ordinary F32,
and existing separately admitted static/symbolic/length/combined Scan domains, through a common core.
Keep ordinary exact SUM outside, retaining required D materialization/workspaces. Prove full
partition/body uses, dependency convexity, physical aliases, named row-axis provenance and lifetimes
before cuts/resources. Retain saved/public/external stores or split the group.

## Non-goals

No new numerics/FMA/reassociation, pointwise-fusion rewrite, ordinary symbolic F64 MATMUL,
all-valid symbolic F64 Scan, broader operations/layouts/carriers, low types, SIMD/threads/OpenBLAS,
Runtime protocol/executor/manager/interpreter, B/T unrolling, per-B cache or Job reuse.
Across-time carry promotion is separate, not mandatory v1 work. CPU0049's simple producer/SUM
and reduction-epilogue work is neither duplicated nor silently closed.

## Contracts

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
- Backend [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes),
  [CPU0073 accepted row-map and multiple-terminal reduction fusion](../../../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion),
  [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract),
  [CPU0071 approved static MATMUL compatibility cutover](../../../../architecture/contracts/backend-execution.md#cpu0071-approved-static-matmul-compatibility-cutover).
- Scan [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification),
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates),
  [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
  [Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories),
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates).
- Foundational [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives)
  and [Aggregate floor](../../../../architecture/contracts/foundational-modules.md#aggregate-floor).
- Runtime [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).

Only the exact MAIN-accepted CPU0073 packet qualifies new-named-type/identity gates.
Ambiguous authority, extra production types/seams or cross-owner edges stop work.
Private helpers within existing owners and the approved capability need no fresh approval.

## Dependencies and integration

- Depends on: CPU0070/0071/0072 Complete; retained CPU0064/0065 and Compiler0015/0016/Prepare0015;
  user approval, independent authority APPROVE and MAIN acceptance recorded above.
- Conflicts with: CPU0048–0052/common fusion, Scan stages/resources/cache/identity and shared contracts.
- Parallel group: None; one serial writer. No concurrent implementation writers authorized.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty base
  (456 existing entries; MAIN's 2536-path hash inventory); preserve all user-owned changes.
- Integration order: independent authority review → MAIN authority acceptance/Ready receipt →
  clean implementation → independent targeted
  code/docs review and corrections → MAIN named checkpoint and shared-document closure.
- Integration validation: CPU0073 stabilized CPU module/Javadoc and named conformance/Engine
  checkpoint below, once; MAIN decides a root gate only for a concrete cross-module/authority risk.
- Shared-document integration owner: MAIN; one exclusively delegated planning/docs writer at a time.

MAIN's Ready verification confirmed dependencies, contracts/base, inactive conflicts and the
456-entry/2536-path dirty base. Baseline compilation passed; 585 originals were frozen.
Implementation/review/integration are accepted; HEAD/index unchanged, no automatic successor.

## Files and symbols

Under `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`, implemented additions:

- `internal.ir.CpuRowMapReductionIr`, nested `Contraction` and `Sum` only.
- `internal.lowering.CpuRowMapReductionLowering.prove` and
  `internal.codegen.emit.CpuRowMapReductionEmitter.emitBound` only.

No additional production type/package/move/public facade is approved.
Existing owners affected: `CpuPortableKernelIr`, `CpuPortableRoutePlan.encoded` (explicit region
envelope branch before aggregate cast), `CpuPartitionLowering`,
`CpuPartitionDagDecomposer`, `CpuSpecializedSubgraphRecognizer`, `CpuPartitionPreparer`,
`CpuPartitionPreparationPlan`, `CpuGeneralScanLowering`, `CpuGeneralScanIr.Stages`,
`CpuGeneralScanEmitter`, `CpuGeneralScanStageSchedule`, `CpuClassFileKernelGenerator`,
`CpuMatmulEmitter`, `CpuAggregateEmitter`, `CpuKernelSpecialization`, and existing
`CpuPreparedExecutable.bindOrdinaryBatch` (full-body work=1). Exact SUM stays unchanged.
New durable test-only classes: `internal.codegen.emit.CpuRowMapReductionJavaOracle`,
`CpuRowMapReductionGeneratedKernelTest`, `internal.lowering.CpuRowMapReductionLoweringTest`.
Extend existing Scan/dynamic-batch/generated-artifact tests and the named checkpoint classes below.

## Acceptance criteria

1. Demonstrate non-RNN dense actual AD (delta, dX/dZ, dW/dU, exact db boundary) and complete real
   Compiler-generated Scan reverse; handwritten reverse alone is insufficient. Durable repository
   test oracles contain self-contained typed equations/cases and saved primals; no temporary files.
2. Recognize multiple terminal sinks, preserve contributor/operand order and typed producing sites,
   and reject terminal-feedback, leave/reentry paths, ambiguous F=B axes, unknown aliases,
   unsupported dtype/layout/family, and over-budget candidates without changing fallback bytes.
3. Mask dominates both contraction loads/math. Test B=0/1, T=0, all inactive, poisoned padding,
   signed-zero raw identity, active specials/cancellation, demanded/dead histories and tied uses.
   Each shared-carry ADD occurs once after row reduction at active(t); no per-row carry update.
4. Ordinary work count is one, range [0,1) executes the full body even at B=0 and publishes
   nonempty shared zeros; [0,0) is empty, other ranges reject before writes. No pointwise row-range
   or parallel splitting. Confirm existing binder convention before approval, without Runtime change.
5. Cold limits/liveness/alias proofs precede resources and slots. No requirement/topology changes
   at finalization, extra RunState, B×parameter banks, repeated pointwise math or hidden fallback costs.
   Validate total viewOrigins and mixed pointwise/contraction rowSites; shared caller-allocated
   row/value/terminal local handles obey the proposal's generation-only lifetime contract.
6. Selected ordinary projection 72 and exact proposed suffixes key all baked facts in both streams;
   envelope 71/Scan projection 71 and noneligible identities/ClassFiles/resources remain unchanged.
   Key fixed R/T/stride/resource bytes if baked; exclude only invocation symbolic B/L/resolved bytes.
   Prove stale-artifact rejection, current reuse and invocation-B/L-independent identities.
   Add exclusive Optional rowMapReductionIr iff projection 72; preserve old constructor delegates
   with empty region. Test region encodedKernelIr routing, all factory/constructor guards and reuse.
7. Generated classes match optimal specialized Java's loop/dataflow and separate math sites;
   no avoidable row-intermediate traffic where unobserved, allocation/boxing/reflection/dispatch.
   Preserve D stores for ordinary exact SUM, all live saved sites and old fallback/oracle coverage.

## Validation

Executed under MAIN's receipt; this documentation pass reruns no Java commands.
Focused development and the single full module/Javadoc command (see Result):

```bash
./gradlew :backends:cpu:test --tests '*CpuRowMapReductionLoweringTest' --tests '*CpuRowMapReductionGeneratedKernelTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc --offline
```

MAIN's named checkpoint ran once in this combined offline invocation:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*CpuSymbolicScanConformanceTest' --tests '*CpuOrdinaryDynamicBatchConformanceTest' :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuSymbolicScanIntegrationTest' --tests '*EngineCpuOrdinaryDynamicBatchIntegrationTest' --offline
```

Retain static F64, symbolic 64→40→1→64/B=0, concurrent mixed-B, cleanup/retry/overlap,
length preflight-before-writes and actual ClassFile/matched-entry coverage in durable fixtures.
Independent review reuses successful source-matched evidence; no duplicate suite just for review.
Architecture tests/root validation only if an actual dependency or cross-module contract changes.
Check Markdown links/anchors/whitespace and final incremental diff; no full docs build.
No timing is authorized. The proposal's finite six-row/one-JVM/watchdog protocol is a separate
future approval decision, not a test prerequisite silently permitting measurement.

## Documentation and review impact

Independent final documentation updates the root/backend mechanism, Scan mechanism/status only,
Scan explanation, CPU guide, [ADR 0052](../../../../design/decisions/0052-row-map-multi-reduction-fusion.md)
and five planning files. General/architecture/backend/ADR/planning/example profiles apply;
API/Javadoc profile guides read-only review. Source Javadocs are finalized by the code author.
Targeted glossary review: no meaning change; row map/terminal remain local optimization labels.
Protected Scan semantics and Runtime/numerics contracts are unchanged. MAIN owns integration;
no measurement permission follows.

## Result

- Delivered the common IR/proof/generation core: three production types plus two nested records;
  ordinary static F32/F64, admitted symbolic scalar F32 and separately admitted scalar Scan cases.
- Implementation scope: 57 paths (19 production Java, 16 test Java, 22 resources); final docs: 11 paths.
  Computed shared producer required; exact ordinary SUM/D stores remain separate, Scan carry ADDs per time.
- Hooke final source/tests/Javadoc/evidence-transfer APPROVE: zero open findings; MAIN accepted 2026-10-30.
- `./gradlew :backends:cpu:test :backends:cpu:javadoc --offline` FAILED: 1409 tests,
  1353 passed, 54 skipped, two failed, zero errors. This is not a fresh full-suite pass.
- `./gradlew :backends:cpu:test --tests '*CpuMatmulBindingTest' --tests '*CpuSymbolicScanTimingMethodologyTest' :backends:cpu:javadoc --offline`: nine PASS; Javadoc PASS, 94 existing warnings.
- `./gradlew :backends:cpu:test --tests '*CpuRowMapReductionGeneratedKernelTest.fusedScanOracleCoversDirectionsSeedsMasksAndEmptyTime' --tests '*CpuSymbolicScanGeneratedKernelTest.selectedComputedRowMapPreservesRawSingletonBeforeDistinctCarryAdd' --offline`: two PASS.
  Hooke approved unchanged pass/skip transfer and exact affected-consumer closure, not a second full run.
- MAIN's combined command above PASS in 16s: 16 conformance + 26 Engine tests, no failures/errors/skips;
  public dense AD reuses one prepared owner across sequential/concurrent B. Sandbox lock denial ran no tests.
- Optimal typed Java/actual ClassFile checks passed; 585 originals unchanged, 356 pre-edit payloads
  authenticated, seven eligible reverse cases retire; noneligible controls retain literal artifacts/resources.
- Exact initial retention-failure attribution is unavailable (original XML not archived); current causal
  producer witnesses and complete corpus are separate accepted evidence, not reconstructed history.
- Source evidence: `/private/tmp/synaptik-cpu0073-run.DOIRA8/implementation-evidence/source.DEspCX/final-handoff.3E6SRI/COMPLETION.md`;
  join log/XML: `/private/tmp/synaptik-cpu0073-joins.sR9d6U/`. Independent docs checks cover links/anchors,
  whitespace, protected semantics, exact incremental scope/hashes and brief size; no Java edits/test reruns.
- No known defects. Performance unmeasured; across-time carry promotion separate, CPU0049 queued.
  No broader capability, Runtime protocol, timing, commit/push or automatic successor.
- Status: Complete. Required follow-up: none within CPU0073.
