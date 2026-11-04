# Task Prepare0014: [Symbolic Scan](../../../../glossary.md#symbolic-scan) [CPU](../../../../glossary.md#term-cpu) vertical

## Status

Complete — MAIN accepted 2026-10-25 after independent code/docs APPROVE and the configured root gate.
PlanningAuthorized=true; ImplementationAuthorized=true record historical [admission](../../../../glossary.md#term-admission); no successor authorization.
Open P1/P2: zero. Source frontier: None. MAIN [Ready](../../../../glossary.md#term-ready) verification 2026-10-25 17:00 UTC.
Size exception: the [atomic](../../../../glossary.md#resource-transaction-and-rollback) cross-[owner](../../../../glossary.md#term-owner) mechanism, falsifiable acceptance, bounded [validation](../../../../glossary.md#term-validation) and
compact completion record need over 15 KB/200 lines; MAIN accepts this readable ≤240-line packet,
below the 25 KB split gate rather than separating one cross-owner [qualification](../../../../glossary.md#term-qualification).

## Change class

Class C — generated loops, per-[run](../../../../glossary.md#term-run) resources, scoped projection and public [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle).
[Prepare](../../../../glossary.md#prepare) transport, [CPU](../../../../glossary.md#term-cpu) recipes/emission and Engine [admission](../../../../glossary.md#term-admission)/[publication](../../../../glossary.md#term-publication) must qualify atomically,
not as separate [guard](../../../../glossary.md#term-guard) removals. One clean serial implementer, then independent code/docs review.

## Goal

All-valid [symbolic Scan](../../../../glossary.md#symbolic-scan) forward AND actual [Compiler](../../../../glossary.md#term-compiler)-generated backward via source [graph](../../../../glossary.md#graph)/public
[Engine](../../../../glossary.md#term-engine).[run](../../../../glossary.md#term-run)/runValues: one [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare) for 64→40→1→64; varying B, fixed time T/[features](../../../../glossary.md#term-software-feature) F/state H.

## [Scope](../../../../glossary.md#term-scope)

[Homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32) ([F32](../../../../glossary.md#floating-point-precision-and-rounding)), generated [scalar](../../../../glossary.md#term-scalar)/single-range [CPU](../../../../glossary.md#term-cpu): [tensor](../../../../glossary.md#tensor)/scalar ADD/SUB/MUL/DIV,
NEG/[TANH](../../../../glossary.md#term-tanh)/[SIGMOID](../../../../glossary.md#term-sigmoid), [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[transpose](../../../../glossary.md#term-transpose), SUM/sumToShape, explicit [constants](../../../../glossary.md#term-constant) and necessary
expand/[rank](../../../../glossary.md#term-rank) edits/[contiguous](../../../../glossary.md#term-contiguous)/same-[dtype](../../../../glossary.md#data-type--datatype) casts preserving B's named [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). Include bounded
rank-three outer [T,B,*] [seed](../../../../glossary.md#term-seed)/zero/constant/[pointwise](../../../../glossary.md#term-pointwise)/[layout](../../../../glossary.md#layout) closure, not arbitrary-rank [admission](../../../../glossary.md#term-admission).
[Model](../../../../glossary.md#term-model) [PURE](../../../../glossary.md#term-pure)/dtypes and completed [Compiler](../../../../glossary.md#term-compiler) F32/[F64](../../../../glossary.md#floating-point-precision-and-rounding) [automatic differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)) remain separate
from CPU F32 admission. Retain static F32/F64, ordinary dynamic-B and length [metadata](../../../../glossary.md#term-metadata)/rejection.

## Non-goals

No second [graph](../../../../glossary.md#graph)/interpreter/runner/session, [memo](../../../../glossary.md#cache-and-memoization), per-B [cache](../../../../glossary.md#term-cache)/relowering, [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) B/B_max,
variable T/[features](../../../../glossary.md#term-feature), lengths/skipping, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training, cold/copy/[persistence](../../../../glossary.md#term-persistence), native/mixed,
low/[F64](../../../../glossary.md#floating-point-precision-and-rounding) symbolic [CPU](../../../../glossary.md#term-cpu), [vector](../../../../glossary.md#term-vector)/parallel/tuning/higher-order or new numerical permission.
[Compiler](../../../../glossary.md#term-compiler) production/[Runtime](../../../../glossary.md#term-runtime) protocols stay read-only.

## Contracts

Start at [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read these headings and affected source/tests; contracts win over plans.

- Root headings: Core [lifecycle](../../../../glossary.md#lifecycle); Core invariants; [Dependency](../../../../glossary.md#term-dependency) rules; Testing requirements.
- [recurrent-scan.md](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan): Accepted symbolic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [scan](../../../../glossary.md#term-cumulative-scan), all four subsections (Symbolic templates, roles and ports; Actual first-order closure and arithmetic; Empty domains and numeric resources; [Qualification](../../../../glossary.md#term-qualification) and mechanism gates); Source body, attachments and scoped [identities](../../../../glossary.md#term-identity-meanings); [Canonical](../../../../glossary.md#term-canonical-form) [residual](../../../../glossary.md#scan-residual-port-and-history-demand) ports and demand.
- [runtime-prepare-engine.md](../../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract)<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine))</span>: Accepted symbolic-batch resource contract; `modules/prepare`; [Prepare](../../../../glossary.md#prepare) lifecycle; [Run](../../../../glossary.md#term-run) lifecycle; Public reusable [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution).
- [backend-execution.md](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes): [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) routes; Accepted ordinary symbolic-batch CPU contract.
- [Compiler](../../../../glossary.md#term-compiler) (complete/read-only): [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [binding glossary](../../../../glossary.md#term-binding))</span>.

Stop on ambiguity, authority conflict or [scope](../../../../glossary.md#term-scope) expansion; do not invent architecture.

## Dependencies and integration

- Depends on: Compiler0014; Prepare0013 and Prepare0011; CPU0037; [Runtime0016](../../runtime/tasks/0016-persistent-prepared-resource-lifecycle.md); Engine0021; Model0041/0042; Planning0008 — all Complete.
- Conflicts with: [Scan](../../../../glossary.md#term-cumulative-scan) projection/resources/[IR](../../../../glossary.md#term-intermediate-representation-ir)/emission/[identity](../../../../glossary.md#term-identity-meanings), ordinary dynamic-B [binding](../../../../glossary.md#term-binding), [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle)/[publication](../../../../glossary.md#term-publication)/shared docs; serialize (inactive).
- Parallel group: None; sole repository writer, no concurrent implementation.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted uncommitted Prepare0014 [planning](../../../../glossary.md#term-planning) [snapshot](../../../../glossary.md#term-snapshot), not merged predecessors.
- Integration order: approved [Ready](../../../../glossary.md#term-ready) verification → [atomic](../../../../glossary.md#resource-transaction-and-rollback) vertical implementation → independent code/docs review → MAIN's source-matched vertical join; no successor launch.
- Integration [validation](../../../../glossary.md#term-validation): `symbolic-scan-cpu-vertical-join`, configured root test/[Javadoc](../../../../glossary.md#term-javadoc) below ONCE after worker evidence/review.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; exclusive final docs delegation to independent reviewer, freeze before join.

MAIN source [admission](../../../../glossary.md#term-admission) 2026-10-25 17:00 UTC: user “jdi na to” approves this implementation.
All declared predecessors' Status/Result (Runtime0016 Completion summary), reviews/validation
Complete; exact contracts/base stable, conflicts inactive, sole writer, index empty. All 2461 files
match accepted planning. Baseline: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/pre-prepare0014-implementation.json`
and its `-files/`; evidence `/private/tmp/synaptik-prepare0014-impl.a04A4j`. No [commit](../../../../glossary.md#term-commit)/push/successor.

## Files, symbols and selected mechanism

Roots: `modules/<owner>/src/main/java/io/github/pho001/synaptik/<owner>/`; [CPU](../../../../glossary.md#term-cpu) uses `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.
No new named production type or supported public/[Runtime](../../../../glossary.md#term-runtime) [signature](../../../../glossary.md#term-signature) selected. Do not force this
via god records, anonymous infrastructure or fake affine bounds. If a named recipe/type/signature
is necessary, STOP for user approval with exact package/name/purpose/declaration and tradeoff.

- [Prepare](../../../../glossary.md#prepare) `analysis/PrepareContext` constructor/ordinaryBatchShape/validateScan/canonical currently
  impose rank-two outer/static-local guards. Extend bounded all-valid symbolic projection and
  explicit requiresBatchSize, unresolved layouts and one B; retain static/length branches.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [layouts glossary](../../../../glossary.md#layout))</span>
  `GraphPreparation.Projection` keeps complete-graph demand and validates ALL contexts before
  analyzers. Local IDs remain scoped, never outer slots; no CPU geometry interpretation.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- CPU `CpuCapabilityProvider.supports` and `internal/prepare/CpuPartitionPreparer.analyze` dispatch
  symbolic inputs to ordinary batch BEFORE Scan. Add Scan qualification and ordinary/Scan unit
  composition in existing analysis. <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [qualification glossary](../../../../glossary.md#term-qualification))</span>`internal/lowering/CpuPartitionLowering` batchLayouts/
  batchByteSizes/lowerOrdinaryBatch/supportsOrdinaryBatch and `internal/codegen/emit/CpuAggregateEmitter.emitOrdinaryBatch` own bounded outer [AD](../../../../glossary.md#autograd) closure.
- CPU `internal/lowering/CpuGeneralScanLowering` checkTypeShape/analyzePointwise/analyzeShaped,
  shapedResources/Facts/Geometry currently bind static spans/cuts/banks. Prepare named-axis
  provenance (including transpose/shared accumulators), not numeric-size roles. Evolve existing<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [provenance glossary](../../../../glossary.md#provenance); [transpose glossary](../../../../glossary.md#term-transpose))</span>
  `internal/ir/CpuGeneralScanIr.View/Cut/Stages`: symbolic recipes versus baked facts.
  `CpuAccessPlan.BatchBinding` supplies constant/factor extents/strides. shapedResources uses checked
  width products/sums: four-byte internal alignment + fixed disjoint regions + one B/frame give
  EXACT constant+factor*B frame/offset/total recipes via <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [strides glossary](../../../../glossary.md#term-stride); [alignment glossary](../../../../glossary.md#term-alignment))</span>`CpuRepresentationRecipes.BatchByteSize`,
  including zero slope, with no rounding term. Scan producing-site SUM needs no five-limb scratch;
  outer <span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs))</span>`CpuAggregateLowering.BatchGeometry` keeps conditional 0/48 bytes. Reuse requires all-B
  dominance. Stronger alignment/conditional elision/non-affine Scan scratch needs the decision
  gate before Ready; no fake affine bound, god recipe tree or Runtime change.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- CPU `internal/prepare/CpuPartitionPreparer.analyzeGeneralScan`, `CpuPartitionPreparationPlan`
  UnitRuntimeFacts/unit validation and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`CpuPartitionFinalizer`: exact fixed/run-size declarations
  before slots; finalize afterward. Explicit symbolic mode even for [B,0]; structural zero-byte
  workspaces retain slots/creators. Existing creators take actual bytes/alignment; no B topology.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [topology glossary](../../../../glossary.md#term-topology))</span>
- CPU `internal/codegen/emit/CpuGeneralScanEmitter.Shaped` emit/loop/load/store/denseAddress,
  `CpuGeneralScanMatmulEmitter.emit/stride` and `CpuGeneralScanReductionEmitter.emit` bake extents,
  singleton/empty decisions and strides. Separate symbolic emission loads extents/strides/frame/
  cut/carry offsets from existing long[] entry packing at invocation/necessary row setup, then
  direct time/row/feature/contributor loops. No per-step/element allocation/maps/boxing/dispatch.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [carry glossary](../../../../glossary.md#term-carry); [feature glossary](../../../../glossary.md#term-software-feature); [element glossary](../../../../glossary.md#term-element))</span>
- CPU `internal/executable/CpuGeneralScanPreparedExecutable` fixed spans/hooks/bindCompatible/
  Invocation pack {time,elements}. Retain recipes; use existing state-aware hooks with actual
  state.geometry() bytes, bind B once, fresh numeric/access/address/carry arrays and unchanged
  long[] entry ABI. Keep typed carriers, full-capacity overlap checks and direct segment entry.
  Existing sibling <span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>`CpuPreparedExecutable.bindOrdinaryBatch` binds ordinary numeric recipes;
  `CpuPreparedPartitionExecutable` state-aware hooks/children compose them with [Scan](../../../../glossary.md#term-cumulative-scan), not a new runner.
- [Engine](../../../../glossary.md#term-engine) [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) preindexes only outer numeric checks: include scoped unused/sibling
  locals. `AdvancedEngine.runOrdinaryBatchOpen` keeps fresh full BatchBindingPlan.bind/private
  obligations → numeric/RunMemoryGeometry.resolve/aggregate bounds → ALL current-storage validation
  → exact-prefix borrow → sole Runtime lease/run → actual RunResult.Publication descriptors.
  No memo, late outward-close gate or Compiler changes.<span data-glossary-navigation="true"> ([RunMemoryGeometry glossary](../../../../glossary.md#run-memory-geometry--runmemorygeometry); [lease glossary](../../../../glossary.md#term-lease); [RunResult glossary](../../../../glossary.md#run-result--runresult); [Publication glossary](../../../../glossary.md#term-publication); [descriptors glossary](../../../../glossary.md#term-descriptor); [memo glossary](../../../../glossary.md#cache-and-memoization); [Compiler glossary](../../../../glossary.md#term-compiler))</span>

[Identity](../../../../glossary.md#term-identity-meanings): envelope/family 71, static [canonical](../../../../glossary.md#term-canonical-form) v1, `scan-shaped-v4`, `scan-matmul-v4` and
[pointwise](../../../../glossary.md#term-pointwise)/fixed/ordinary keys AND generated bytes stay unchanged. Symbolic mode alone
adds canonical `:symbolic-batch-v1` and baked `|scan-symbolic-batch-v1=` in
`CpuGeneralScanIr.encodedKernelIr/bakedSpecializationIdentity`, consumed by BOTH
`CpuKernelSpecialization.classIdentityBytes/compatibilityBytes` (ordinaryBatch stays false for Scan).
Key all baked [dtype](../../../../glossary.md#data-type--datatype)/body/role/demand/mapping/recipe/packing/control facts, not instance B/resolved
bytes/slots/addresses. `CpuClassFileKernelGenerator.validate/verify/defineClassBytes` and
`CpuGeneratedKernelArtifactStore.readEnvelope` keep exact [fingerprint](../../../../glossary.md#term-fingerprint)/[metadata](../../../../glossary.md#term-metadata)/name/entry checks;
no decoder/[schema](../../../../glossary.md#term-schema) change. Test static↔symbolic/changed-recipe rejection; no diagnostic-text identity.

## Acceptance criteria

- Actual Compiler-generated reverse, independently qualified forward/reverse bodies; both
  traversals, joint final/emission and emission-only nonuniform seeds, ties/reused occurrences,
  disconnected invariants and exact saved TANH/SIGMOID outputs. Compare all outputs/gradients
  with independent ordinary unrolling and specialized clean Java.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [TANH glossary](../../../../glossary.md#term-tanh); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [gradients glossary](../../../../glossary.md#term-gradient); [unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- 64→40→1→64 plus B=0,1,F,H and B=F=H: stable compile/prepare/artifact/code/routes/slots/positions;
  actual time/batch/features stride B<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [artifact glossary](../../../../glossary.md#term-artifact); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [features glossary](../../../../glossary.md#term-feature); [stride glossary](../../../../glossary.md#term-stride))</span>*F, frame B*H, ALL port layouts/flags/order/identities and current
  input-view descriptors. Parameter-shaped accumulation, never B×parameter. Assert exact demanded
  row/shared bytes and zero dead history bytes/stores, including reverse residuals; shared [F,H]
  history is [T,F,H], not row-banked.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [Parameter glossary](../../../../glossary.md#term-parameter))</span>
- Inclusive cut lifetimes/size dominance hold for EVERY B or regions stay disjoint. All count/
  stride/span/offset/padding/aggregate overflow rejects before borrow/allocation, including unused
  trailing local/outer domains behind zero; no per-B generation.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Separate T=0 plan: no body/hoisted work, raw distinct final-carry copy, dInitial=final seed,
  empty stacks/dX and typed-zero invariant gradients. B=0 separately/with T=0: nonempty shared
  work/dW executes; K=0 MATMUL/SUM typed zero, mean(empty) NaN, actual mean divisor B*H.
  Retain singleton/raw-copy, special-value, signed-zero and represented saved-site semantics.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [seed glossary](../../../../glossary.md#term-seed); [invariant glossary](../../../../glossary.md#term-invariant); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [mean glossary](../../../../glossary.md#term-mean); [NaN glossary](../../../../glossary.md#term-nan); [singleton glossary](../../../../glossary.md#term-singleton); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Fresh arrays/resources; legal read aliases, rejected writable overlap/capacity tails; reverse
  attempt-all cleanup/suppression, failed run/no publication, retry, concurrent mixed-B/open-result
  isolation and close/publication races. Numeric-only/model-free Runtime, one RunState/lease,
  borrowed caller storage, Engine prepared-before-input precedence.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication); [model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime); [RunState glossary](../../../../glossary.md#run-state--runstate); [lease glossary](../../../../glossary.md#term-lease); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Replace only valid symbolic rejection assertions with positives. PrepareContextTest's
  scanRejectsEveryDynamicLocalIncludingUnusedFormalAndUnusedSiblingEvenAtZeroTime and
  GraphPreparationTest's laterPartitionDynamicUnusedFormalFailsBeforeTheFirstAnalyzer keep malformed
  unused/mismatched-symbol/binding negatives. Retain static/ordinary/length metadata and CPU length,
  unsupported body/dtype/layout, cold/copy/wire, higher-order/residual-root rejection gates.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [binding glossary](../../../../glossary.md#term-binding); [metadata glossary](../../../../glossary.md#term-metadata); [CPU glossary](../../../../glossary.md#term-cpu); [dtype glossary](../../../../glossary.md#data-type--datatype); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>

## [Validation](../../../../glossary.md#term-validation)

Authorized worker commands. New fixtures: CpuSymbolicScanGeneratedKernelTest ([CPU](../../../../glossary.md#term-cpu) internal.codegen.emit),
EngineSymbolicScanTest ([Engine](../../../../glossary.md#term-engine)), CpuSymbolicScanConformanceTest ([conformance](../../../../glossary.md#term-conformance)),
EngineCpuSymbolicScanIntegrationTest (integration). Extend existing PrepareContextTest/
GraphPreparationTest and CPU [Scan](../../../../glossary.md#term-cumulative-scan) capability/[lowering](../../../../glossary.md#lowering)/preparation/execution tests. All performance
opt-ins stay unset/false except the probe. Shell commands use this task-local option array:

```bash
scan_checks=(--max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false)
./gradlew :modules:prepare:test --tests '*PrepareContextTest' --tests '*GraphPreparationTest' :backends:cpu:test --tests '*CpuGeneralScan*Test' --tests '*CpuSymbolicScanGeneratedKernelTest' :modules:engine:test --tests '*EngineSymbolicScanTest' --tests '*EngineOrdinaryDynamicBatchTest' "${scan_checks[@]}"
./gradlew :testing:backend-conformance:test --tests '*CpuSymbolicScanConformanceTest' --tests '*CpuGeneralScanConformanceTest' --tests '*CpuOrdinaryDynamicBatchConformanceTest' :testing:integration-tests:test --tests '*EngineCpuSymbolicScanIntegrationTest' --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuOrdinaryDynamicBatchIntegrationTest' "${scan_checks[@]}"
./gradlew :modules:prepare:test :backends:cpu:test :modules:engine:test :modules:prepare:javadoc :backends:cpu:javadoc :modules:engine:javadoc "${scan_checks[@]}"
```

Last command: stabilized worker [modules](../../../../glossary.md#term-module) ONCE; reviewer reuses evidence. Class-File tests check
clean-Java algorithm/loops/[dataflow](../../../../glossary.md#term-dataflow), saved loads, dynamic M/K/N, hidden calls/allocation/boxing/
reflection/[dispatch](../../../../glossary.md#term-dispatch) and static key/byte goldens. One probe: future CPU CpuSymbolicScanTimingTest +
Engine EngineSymbolicScanTimingTest; existing forwarded default-false dynamicBatch flag.
Separate cold [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare)/generate/define, BIND (full semantic/scoped/numeric/storage [admission](../../../../glossary.md#term-admission)),
[KERNEL](../../../../glossary.md#kernel) generated/direct forward+actual reverse with matched resources, and Engine fresh-[run](../../../../glossary.md#term-run)/result/
[materialization](../../../../glossary.md#materialization). Fixed cases: T3/F4/H3 at B1/40/64 and T4/[F32](../../../../glossary.md#floating-point-precision-and-rounding)/H32 at B64; 100 [warmup](../../../../glossary.md#term-warmup) iterations per
variant, three alternating paired [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) of 300 iterations each. At most 30s per test body,
60s aggregate including cold setup/warmup; censor/report unfinished cases at cap. One [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/module,
no adaptive loops, retries/[forks](../../../../glossary.md#term-fork-meanings)/campaign or parity promise; unsupported environment is reported.

```bash
./gradlew :backends:cpu:test --tests '*CpuSymbolicScanTimingTest' :modules:engine:test --tests '*EngineSymbolicScanTimingTest' --no-parallel --max-workers=1 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=true
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc "${scan_checks[@]}"
```

After the original Engine cold-setup failure, the user explicitly approved ONE Engine-only
environment follow-up, without source/build edits or CPU timing repetition:

```bash
JAVA_TOOL_OPTIONS='--add-modules=jdk.incubator.vector' ./gradlew --no-daemon :modules:engine:test --tests '*EngineSymbolicScanTimingTest' --no-parallel --max-workers=1 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=true
```

The root command above is MAIN's ONCE source-matched join after review: JDK26/existing [native library](../../../../glossary.md#term-native-library),
no rebuild. Unsupported/missing environment stops the gate; do not weaken skips. Record commands,
counts/skips, [identity](../../../../glossary.md#term-identity-meanings)/resource evidence and timing limits, not speed/negligible-B-overhead claims.

## Documentation and review impact

Independent Class C pass finalizes [Javadoc](../../../../glossary.md#term-javadoc), [partition](../../../../glossary.md#partition)-preparer/[CPU](../../../../glossary.md#term-cpu)/running-models guides, [runtime](../../../../glossary.md#term-runtime)-
[prepare](../../../../glossary.md#prepare)-[backend](../../../../glossary.md#backend) explanation, [API](../../../../glossary.md#term-application-programming-interface-api) support-status and glossary [Symbolic Scan](../../../../glossary.md#symbolic-scan)/[Symbolic batch binding](../../../../glossary.md#symbolic-batch-binding)/
[Run memory geometry](../../../../glossary.md#run-memory-geometry--runmemorygeometry). MAIN owns final status integration; new [normative](../../../../glossary.md#term-normative) decisions need approval.
Narrow [Model](../../../../glossary.md#term-model) exception: `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Scan.java`
class support-status Javadoc ONLY; correct execution status, retain cold/[persistence](../../../../glossary.md#term-persistence)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) gates. No executable/API/header edit.

## Result

Status: Complete. [Homogeneous](../../../../glossary.md#term-homogeneous) [F32](../../../../glossary.md#floating-point-precision-and-rounding) [symbolic Scan](../../../../glossary.md#symbolic-scan) forward AND actual [Compiler](../../../../glossary.md#term-compiler) reverse work through
one public [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare) for changing B, fixed T/F/H. Existing [owners](../../../../glossary.md#term-owner) only: 25 implementation
paths plus targeted documentation; no new production type/[API](../../../../glossary.md#term-application-programming-interface-api), Compiler/[Runtime](../../../../glossary.md#term-runtime) executable,
build/[dependency](../../../../glossary.md#term-dependency)/native change. Static six-[Scan](../../../../glossary.md#term-cumulative-scan) and 17 ordinary/static key/[identity](../../../../glossary.md#term-identity-meanings)/Class-File
witnesses remain identical. Independent code and documentation reviews APPROVE, zero open P1/P2.

MAIN's final source-matched configured `test javadoc` PASS in 5m42s, 119 tasks (40 executed,
79 up-to-date). Current included-project XML: 4944 tests, 0 failures/errors, 53 skips; unchanged
UP-TO-DATE results are reused, not claimed fresh. Post-fix FULL [CPU](../../../../glossary.md#term-cpu): 1253/40 skips; Prepare95/0,
Engine151/2, conformance66/0, integration214/1, architecture25/0. Skips:42 CPU/[Engine](../../../../glossary.md#term-engine) optional or
conditional cases,10 optional [Metal](../../../../glossary.md#term-metal) [observer](../../../../glossary.md#term-observer) cases,1 [host](../../../../glossary.md#term-host)-ineligible tuning case. Required native
Metal cases ran; existing library [hash](../../../../glossary.md#term-hash) unchanged. [Javadoc](../../../../glossary.md#term-javadoc) passed with94 CPU/46 Metal warnings
and2 incubator warnings, not warning-free. Documentation and affected Javadocs finalized.

Original bounded probe: CPU four cases completed (XML2.948s), Engine cold setup failed missing
the [Vector](../../../../glossary.md#term-vector) [module](../../../../glossary.md#term-module) (0.275s), preserved honestly. User-approved Engine-only environment follow-up
PASS: four cases, 100 warmups/three paired300-call [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), XML6.046s, no censorship or CPU rerun.
Aggregate bodies9.269s satisfy30s/body and60s total caps. BIND and fresh-[run](../../../../glossary.md#term-run)/[materialization](../../../../glossary.md#materialization)
samples are separate from [kernels](../../../../glossary.md#kernel); samples vary and do NOT prove parity, a causal explanation
or negligible B overhead. No further performance campaign is authorized.

Evidence root `/private/tmp/synaptik-prepare0014-impl.a04A4j`: `handoff.txt`, `acceptance-matrix.txt`,
`final-code-review.txt`, `docs-final.txt`, `engine-only-result.txt`, `main-root-join.log` and
`root-test-results.tgz`. Final [token](../../../../glossary.md#term-token)/[scope](../../../../glossary.md#term-scope)/link checks PASS; prior dirty work, HEAD/origin/index
unchanged. [Memo](../../../../glossary.md#cache-and-memoization), lengths/skipping, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training and other excluded workflows remain future gates.
No required implementation follow-up, [commit](../../../../glossary.md#term-commit)/push or successor launch.
