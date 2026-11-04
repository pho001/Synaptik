# Task CPU0073: Row-map and multiple-terminal reduction [fusion](../../../../glossary.md#term-fusion)

## Reader context

A **row map** computes the ordered intermediate values for one logical row. A **terminal**
combines contributions from rows into a final result, such as a weight gradient. Fusion here
lets several consumers reuse the row's computed value rather than storing and rereading every
private intermediate. It does not permit deleting a value that an output, saved history or
separate exact reduction still needs.

`CpuRowMapReductionLowering` proves that sharing is legal before Prepare assigns memory slots.
Ordinary CPU generation emits one whole-body work item; Scan generation embeds the region
inside its existing time loop. Runtime later invokes those prepared executables, not the
lowerer or an application callback for each row. The
[focused row-fusion example](../../../../backend-guide/cpu-backend.md#implemented-row-map-and-multiple-terminal-fusion)
shows why ordinary exact bias SUM still needs its stored input. For the broader present-day body
domain, use [current Scan composition](../../../../backend-guide/cpu-backend.md#current-general-scan-composition)
rather than treating this task's bounded admission matrix as a current operation whitelist.
The recorded failed full run and evidence-transfer limits below remain unchanged.

## Status

Complete — MAIN accepted implementation, Hooke's final code/tests/[Javadoc](../../../../glossary.md#term-javadoc)/evidence-transfer
APPROVE (zero open findings) and the named join on 2026-10-30; independent documentation closure
is recorded below. User approval and Boole's authority APPROVE preceded the verified [Ready](../../../../glossary.md#term-ready) receipt.
The approved [proposal](../row-map-multi-reduction-fusion-plan.md) pre-edit [SHA-256](../../../../glossary.md#term-sha-256) is
`a1f510c71aa14eef07b56fc48bde736c8dbf1079467c5142f8c69a457a37132b`.
Source/measurement frontiers: None; old budgets closed, no [commit](../../../../glossary.md#term-commit)/push or automatic successor.

## Change class

Class C — generated hot-loop/[dataflow](../../../../glossary.md#term-dataflow), [materialization](../../../../glossary.md#materialization)/[lifetime](../../../../glossary.md#term-lifetime) and [artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings) changes.
Authority prerequisite, clean implementation and independent code/documentation contexts are complete.

## Goal

Deliver one general [CPU](../../../../glossary.md#term-cpu) [lowering](../../../../glossary.md#lowering) capability: share a pure ordered row map, including bounded
[feature](../../../../glossary.md#term-feature) contractions, among row outputs and multiple terminal contractions/represented sums.
Ordinary [graphs](../../../../glossary.md#graph) get one lowered execution unit and [fusion](../../../../glossary.md#term-fusion) emitter; [Scan](../../../../glossary.md#term-cumulative-scan) gets a generation-time
region in its existing unit/time loop, never a per-step [Runtime](../../../../glossary.md#term-runtime) unit, [callback](../../../../glossary.md#term-callback), worker or Job.
Reuse existing [pointwise](../../../../glossary.md#term-pointwise) fusion and [operation](../../../../glossary.md#operation) math; introduce no [Tensor](../../../../glossary.md#tensor) semantic operation.

## [Scope](../../../../glossary.md#term-scope)

Use the proposal's [admission matrix](../row-map-multi-reduction-fusion-plan.md#v1-admission-and-exclusions)<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>,
[safety/range proof](../row-map-multi-reduction-fusion-plan.md#legality-resources-and-range-proof),
[exact type/seam map](../row-map-multi-reduction-fusion-plan.md#minimal-proposed-production-map-and-signatures)
and [identity recommendation](../row-map-multi-reduction-fusion-plan.md#compatibility-and-retirement-recommendation)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>.
One cohesive capability covers ordinary static [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding), already-admitted symbolic ordinary F32,
and existing separately admitted static/symbolic/length/combined [Scan](../../../../glossary.md#term-cumulative-scan) domains, through a common core.
Keep ordinary exact SUM outside, retaining required D [materialization](../../../../glossary.md#materialization)/workspaces. Prove full
[partition](../../../../glossary.md#partition)/body uses, [dependency](../../../../glossary.md#term-dependency) convexity, physical [aliases](../../../../glossary.md#aliasing-and-arena), named row-[axis](../../../../glossary.md#term-axis) [provenance](../../../../glossary.md#provenance) and lifetimes
before cuts/resources. Retain saved/public/external stores or split the group.

## Non-goals

No new numerics/[FMA](../../../../glossary.md#term-fused-multiply-add-fma)/[reassociation](../../../../glossary.md#term-reassociation), [pointwise](../../../../glossary.md#term-pointwise)-[fusion](../../../../glossary.md#term-fusion) rewrite, ordinary symbolic [F64](../../../../glossary.md#floating-point-precision-and-rounding) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul),
all-valid symbolic F64 [Scan](../../../../glossary.md#term-cumulative-scan), broader operations/[layouts](../../../../glossary.md#layout)/carriers, low types, [SIMD](../../../../glossary.md#simd)/threads/[OpenBLAS](../../../../glossary.md#term-openblas),
[Runtime](../../../../glossary.md#term-runtime) protocol/executor/manager/interpreter, B/T [unrolling](../../../../glossary.md#term-unrolling), per-B [cache](../../../../glossary.md#term-cache) or Job reuse.
Across-time [carry](../../../../glossary.md#term-carry) promotion is separate, not mandatory v1 work. CPU0049's simple [producer](../../../../glossary.md#term-producer)/SUM
and reduction-[epilogue](../../../../glossary.md#term-epilogue) work is neither duplicated nor silently closed.

## Contracts

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>.
- [Backend](../../../../glossary.md#backend) [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>,
  [CPU0073 accepted row-map and multiple-terminal reduction fusion](../../../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>,
  [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>,
  [CPU0071 approved static MATMUL compatibility cutover](../../../../architecture/contracts/backend-execution.md#cpu0071-approved-static-matmul-compatibility-cutover)<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>.
- [Scan](../../../../glossary.md#term-cumulative-scan) [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>,
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates),
  [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
  [Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories),
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>.
- Foundational [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives)
  and [Aggregate floor](../../../../architecture/contracts/foundational-modules.md#aggregate-floor)<span data-glossary-navigation="true"> ([Aggregate floor glossary](../../../../glossary.md#term-aggregate-floor))</span>.
- [Runtime](../../../../glossary.md#term-runtime) [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.

Only the exact MAIN-accepted CPU0073 packet qualifies new-named-type/[identity](../../../../glossary.md#term-identity-meanings) gates.
Ambiguous authority, extra production types/seams or cross-[owner](../../../../glossary.md#term-owner) edges stop work.
Private helpers within existing owners and the approved capability need no fresh approval.

## Dependencies and integration

- Depends on: CPU0070/0071/0072 Complete; retained CPU0064/0065 and Compiler0015/0016/Prepare0015;
  user approval, independent authority APPROVE and MAIN acceptance recorded above.
- Conflicts with: CPU0048–0052/common [fusion](../../../../glossary.md#term-fusion), [Scan](../../../../glossary.md#term-cumulative-scan) stages/resources/[cache](../../../../glossary.md#term-cache)/[identity](../../../../glossary.md#term-identity-meanings) and shared contracts.
- Parallel group: None; one serial writer. No concurrent implementation writers authorized.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty base
  (456 existing entries; MAIN's 2536-path hash inventory); preserve all user-owned changes.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>
- Integration order: independent authority review → MAIN authority acceptance/Ready receipt →
  clean implementation → independent targeted
  code/docs review and corrections → MAIN named checkpoint and shared-document closure.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration validation: CPU0073 stabilized CPU module/Javadoc and named conformance/Engine
  checkpoint below, once; MAIN decides a root gate only for a concrete cross-module/authority risk.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; one exclusively delegated [planning](../../../../glossary.md#term-planning)/docs writer at a time.

MAIN's [Ready](../../../../glossary.md#term-ready) verification confirmed dependencies, contracts/base, inactive conflicts and the
456-entry/2536-path dirty base. Baseline [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) passed; 585 originals were frozen.
Implementation/review/integration are accepted; HEAD/index unchanged, no automatic successor.

## Files and symbols

Under `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`, implemented additions:

- `internal.ir.CpuRowMapReductionIr`, nested [`Contraction`](../../../../glossary.md#contraction-dimension) and `Sum` only.
- `internal.lowering.CpuRowMapReductionLowering.prove` and
  `internal.codegen.emit.CpuRowMapReductionEmitter.emitBound` only.

No additional production type/package/move/public [facade](../../../../glossary.md#term-facade) is approved.
Existing [owners](../../../../glossary.md#term-owner) affected: `CpuPortableKernelIr`, `CpuPortableRoutePlan.encoded` (explicit region
envelope branch before aggregate cast), `CpuPartitionLowering`,
`CpuPartitionDagDecomposer`, `CpuSpecializedSubgraphRecognizer`, `CpuPartitionPreparer`,
`CpuPartitionPreparationPlan`, `CpuGeneralScanLowering`, `CpuGeneralScanIr.Stages`,
`CpuGeneralScanEmitter`, `CpuGeneralScanStageSchedule`, `CpuClassFileKernelGenerator`,
`CpuMatmulEmitter`, `CpuAggregateEmitter`, `CpuKernelSpecialization`, and existing
`CpuPreparedExecutable.bindOrdinaryBatch` (full-body work=1). Exact SUM stays unchanged.
New durable test-only classes: `internal.codegen.emit.CpuRowMapReductionJavaOracle`,
`CpuRowMapReductionGeneratedKernelTest`, `internal.lowering.CpuRowMapReductionLoweringTest`.
Extend existing [Scan](../../../../glossary.md#term-cumulative-scan)/dynamic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/generated-[artifact](../../../../glossary.md#term-artifact) tests and the named [checkpoint](../../../../glossary.md#term-checkpoint) classes below.

## Acceptance criteria

1. Demonstrate non-RNN dense actual AD (delta, dX/dZ, dW/dU, exact db boundary) and complete real
   Compiler-generated Scan reverse; handwritten reverse alone is insufficient. Durable repository
   test oracles contain self-contained typed equations/cases and saved primals; no temporary files.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [AD glossary](../../../../glossary.md#autograd); [Compiler glossary](../../../../glossary.md#term-compiler); [Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
2. Recognize multiple terminal sinks, preserve contributor/operand order and typed producing sites,
   and reject terminal-feedback, leave/reentry paths, ambiguous F=B axes, unknown aliases,
   unsupported dtype/layout/family, and over-budget candidates without changing fallback bytes.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [dtype glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout))</span>
3. Mask dominates both contraction loads/math. Test B=0/1, T=0, all inactive, poisoned padding,
   signed-zero raw identity, active specials/cancellation, demanded/dead histories and tied uses.
   Each shared-carry ADD occurs once after row reduction at active(t); no per-row carry update.<span data-glossary-navigation="true"> ([Mask glossary](../../../../glossary.md#term-boolean-mask); [contraction glossary](../../../../glossary.md#contraction-dimension); [padding glossary](../../../../glossary.md#padding); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [identity glossary](../../../../glossary.md#term-identity-meanings); [cancellation glossary](../../../../glossary.md#term-cancellation); [carry glossary](../../../../glossary.md#term-carry))</span>
4. Ordinary work count is one, range [0,1) executes the full body even at B=0 and publishes
   nonempty shared zeros; [0,0) is empty, other ranges reject before writes. No pointwise row-range
   or parallel splitting. Confirm existing binder convention before approval, without Runtime change.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
5. Cold limits/liveness/alias proofs precede resources and slots. No requirement/topology changes
   at finalization, extra RunState, B×parameter banks, repeated pointwise math or hidden fallback costs.
   Validate total viewOrigins and mixed pointwise/contraction rowSites; shared caller-allocated
   row/value/terminal local handles obey the proposal's generation-only lifetime contract.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [topology glossary](../../../../glossary.md#term-topology); [RunState glossary](../../../../glossary.md#run-state--runstate); [parameter glossary](../../../../glossary.md#term-parameter); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
6. Selected ordinary projection 72 and exact proposed suffixes key all baked facts in both streams;
   envelope 71/Scan projection 71 and noneligible identities/ClassFiles/resources remain unchanged.
   Key fixed R/T/stride/resource bytes if baked; exclude only invocation symbolic B/L/resolved bytes.
   Prove stale-artifact rejection, current reuse and invocation-B/L-independent identities.
   Add exclusive Optional rowMapReductionIr iff projection 72; preserve old constructor delegates
   with empty region. Test region encodedKernelIr routing, all factory/constructor guards and reuse.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [artifact glossary](../../../../glossary.md#term-artifact))</span>
7. Generated classes match optimal specialized Java's loop/dataflow and separate math sites;
   no avoidable row-intermediate traffic where unobserved, allocation/boxing/reflection/dispatch.
   Preserve D stores for ordinary exact SUM, all live saved sites and old fallback/oracle coverage.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [dispatch glossary](../../../../glossary.md#term-dispatch); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>

## [Validation](../../../../glossary.md#term-validation)

Executed under MAIN's receipt; this documentation pass reruns no Java commands.
Focused development and the single full [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) command (see Result):

```bash
./gradlew :backends:cpu:test --tests '*CpuRowMapReductionLoweringTest' --tests '*CpuRowMapReductionGeneratedKernelTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc --offline
```

MAIN's named [checkpoint](../../../../glossary.md#term-checkpoint) ran once in this combined offline invocation:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*CpuSymbolicScanConformanceTest' --tests '*CpuOrdinaryDynamicBatchConformanceTest' :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuSymbolicScanIntegrationTest' --tests '*EngineCpuOrdinaryDynamicBatchIntegrationTest' --offline
```

Retain static [F64](../../../../glossary.md#floating-point-precision-and-rounding), symbolic 64→40→1→64/B=0, concurrent mixed-B, cleanup/retry/overlap,
length [preflight](../../../../glossary.md#term-preflight)-before-writes and actual ClassFile/matched-entry coverage in durable fixtures.
Independent review reuses successful source-matched evidence; no duplicate suite just for review.
Architecture tests/root validation only if an actual [dependency](../../../../glossary.md#term-dependency) or cross-module contract changes.
Check Markdown links/anchors/whitespace and final incremental diff; no full docs build.
No timing is authorized. The proposal's [finite](../../../../glossary.md#term-finite) six-row/one-[JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/watchdog protocol is a separate
future approval decision, not a test prerequisite silently permitting measurement.

## Documentation and review impact

Independent final documentation updates the root/[backend](../../../../glossary.md#backend) mechanism, [Scan](../../../../glossary.md#term-cumulative-scan) mechanism/status only,
Scan explanation, [CPU](../../../../glossary.md#term-cpu) guide, [ADR 0052](../../../../design/decisions/0052-row-map-multi-reduction-fusion.md)
and five [planning](../../../../glossary.md#term-planning) files. General/architecture/backend/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/planning/example profiles apply;
[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) profile guides read-only review. Source Javadocs are finalized by the code author.
Targeted glossary review: no meaning change; row map/terminal remain local optimization labels.
Protected Scan semantics and [Runtime](../../../../glossary.md#term-runtime)/numerics contracts are unchanged. MAIN owns integration;
no measurement permission follows.

## Result

- Delivered the common IR/proof/generation core: three production types plus two nested records;
  ordinary static F32/F64, admitted symbolic scalar F32 and separately admitted scalar Scan cases.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [scalar glossary](../../../../glossary.md#term-scalar); [Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Implementation scope: 57 paths (19 production Java, 16 test Java, 22 resources); final docs: 11 paths.
  Computed shared producer required; exact ordinary SUM/D stores remain separate, Scan carry ADDs per time.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [producer glossary](../../../../glossary.md#term-producer); [carry glossary](../../../../glossary.md#term-carry))</span>
- Hooke final source/tests/[Javadoc](../../../../glossary.md#term-javadoc)/evidence-transfer APPROVE: zero open findings; MAIN accepted 2026-10-30.
- `./gradlew :backends:cpu:test :backends:cpu:javadoc --offline` FAILED: 1409 tests,
  1353 passed, 54 skipped, two failed, zero errors. This is not a fresh full-suite pass.
- `./gradlew :backends:cpu:test --tests '*CpuMatmulBindingTest' --tests '*CpuSymbolicScanTimingMethodologyTest' :backends:cpu:javadoc --offline`: nine PASS; Javadoc PASS, 94 existing warnings.
- `./gradlew :backends:cpu:test --tests '*CpuRowMapReductionGeneratedKernelTest.fusedScanOracleCoversDirectionsSeedsMasksAndEmptyTime' --tests '*CpuSymbolicScanGeneratedKernelTest.selectedComputedRowMapPreservesRawSingletonBeforeDistinctCarryAdd' --offline`: two PASS.
  Hooke approved unchanged pass/skip transfer and exact affected-consumer closure, not a second full run.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- MAIN's combined command above PASS in 16s: 16 conformance + 26 Engine tests, no failures/errors/skips;
  public dense AD reuses one prepared owner across sequential/concurrent B. Sandbox lock denial ran no tests.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [AD glossary](../../../../glossary.md#autograd); [owner glossary](../../../../glossary.md#term-owner))</span>
- Optimal typed Java/actual ClassFile checks passed; 585 originals unchanged, 356 pre-edit payloads
  authenticated, seven eligible reverse cases retire; noneligible controls retain literal artifacts/resources.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Exact initial retention-failure attribution is unavailable (original XML not archived); current causal
  producer witnesses and complete corpus are separate accepted evidence, not reconstructed history.
- Source evidence: `/private/tmp/synaptik-cpu0073-run.DOIRA8/implementation-evidence/source.DEspCX/final-handoff.3E6SRI/COMPLETION.md`;
  join log/XML: <span data-glossary-navigation="true"> ([log glossary](../../../../glossary.md#term-log))</span>`/private/tmp/synaptik-cpu0073-joins.sR9d6U/`. Independent docs checks cover links/anchors,
  whitespace, protected semantics, exact incremental scope/hashes and brief size; no Java edits/test reruns.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span>
- No known defects. Performance unmeasured; across-time carry promotion separate, CPU0049 queued.
  No broader capability, Runtime protocol, timing, commit/push or automatic successor.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [commit glossary](../../../../glossary.md#term-commit))</span>
- Status: Complete. Required follow-up: none within CPU0073.
