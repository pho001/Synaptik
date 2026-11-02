# Task Prepare0014: Symbolic Scan CPU vertical

## Status

Complete — MAIN accepted 2026-10-25 after independent code/docs APPROVE and the configured root gate.
PlanningAuthorized=true; ImplementationAuthorized=true record historical admission; no successor authorization.
Open P1/P2: zero. Source frontier: None. MAIN Ready verification 2026-10-25 17:00 UTC.
Size exception: the atomic cross-owner mechanism, falsifiable acceptance, bounded validation and
compact completion record need over 15 KB/200 lines; MAIN accepts this readable ≤240-line packet,
below the 25 KB split gate rather than separating one cross-owner qualification.

## Change class

Class C — generated loops, per-run resources, scoped projection and public Engine lifecycle.
Prepare transport, CPU recipes/emission and Engine admission/publication must qualify atomically,
not as separate guard removals. One clean serial implementer, then independent code/docs review.

## Goal

All-valid symbolic Scan forward AND actual Compiler-generated backward via source graph/public
Engine.run/runValues: one compile/prepare for 64→40→1→64; varying B, fixed time T/features F/state H.

## Scope

Homogeneous FLOAT32 (F32), generated scalar/single-range CPU: tensor/scalar ADD/SUB/MUL/DIV,
NEG/TANH/SIGMOID, rank-two MATMUL/transpose, SUM/sumToShape, explicit constants and necessary
expand/rank edits/contiguous/same-dtype casts preserving B's named coordinate. Include bounded
rank-three outer [T,B,*] seed/zero/constant/pointwise/layout closure, not arbitrary-rank admission.
Model PURE/dtypes and completed Compiler F32/F64 automatic differentiation (AD) remain separate
from CPU F32 admission. Retain static F32/F64, ordinary dynamic-B and length metadata/rejection.

## Non-goals

No second graph/interpreter/runner/session, memo, per-B cache/relowering, sample B/B_max,
variable T/features, lengths/skipping, NN/Training, cold/copy/persistence, native/mixed,
low/F64 symbolic CPU, vector/parallel/tuning/higher-order or new numerical permission.
Compiler production/Runtime protocols stay read-only.

## Contracts

Start at [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read these headings and affected source/tests; contracts win over plans.

- Root headings: Core lifecycle; Core invariants; Dependency rules; Testing requirements.
- [recurrent-scan.md](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan): Accepted symbolic-batch scan, all four subsections (Symbolic templates, roles and ports; Actual first-order closure and arithmetic; Empty domains and numeric resources; Qualification and mechanism gates); Source body, attachments and scoped identities; Canonical residual ports and demand.
- [runtime-prepare-engine.md](../../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract): Accepted symbolic-batch resource contract; `modules/prepare`; Prepare lifecycle; Run lifecycle; Public reusable prepared execution.
- [backend-execution.md](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes): CPU backend routes; Accepted ordinary symbolic-batch CPU contract.
- Compiler (complete/read-only): [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding).

Stop on ambiguity, authority conflict or scope expansion; do not invent architecture.

## Dependencies and integration

- Depends on: Compiler0014; Prepare0013 and Prepare0011; CPU0037; [Runtime0016](../../runtime/tasks/0016-persistent-prepared-resource-lifecycle.md); Engine0021; Model0041/0042; Planning0008 — all Complete.
- Conflicts with: Scan projection/resources/IR/emission/identity, ordinary dynamic-B binding, Engine lifecycle/publication/shared docs; serialize (inactive).
- Parallel group: None; sole repository writer, no concurrent implementation.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted uncommitted Prepare0014 planning snapshot, not merged predecessors.
- Integration order: approved Ready verification → atomic vertical implementation → independent code/docs review → MAIN's source-matched vertical join; no successor launch.
- Integration validation: `symbolic-scan-cpu-vertical-join`, configured root test/Javadoc below ONCE after worker evidence/review.
- Shared-document integration owner: MAIN; exclusive final docs delegation to independent reviewer, freeze before join.

MAIN source admission 2026-10-25 17:00 UTC: user “jdi na to” approves this implementation.
All declared predecessors' Status/Result (Runtime0016 Completion summary), reviews/validation
Complete; exact contracts/base stable, conflicts inactive, sole writer, index empty. All 2461 files
match accepted planning. Baseline: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/pre-prepare0014-implementation.json`
and its `-files/`; evidence `/private/tmp/synaptik-prepare0014-impl.a04A4j`. No commit/push/successor.

## Files, symbols and selected mechanism

Roots: `modules/<owner>/src/main/java/io/github/pho001/synaptik/<owner>/`; CPU uses `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.
No new named production type or supported public/Runtime signature selected. Do not force this
via god records, anonymous infrastructure or fake affine bounds. If a named recipe/type/signature
is necessary, STOP for user approval with exact package/name/purpose/declaration and tradeoff.

- Prepare `analysis/PrepareContext` constructor/ordinaryBatchShape/validateScan/canonical currently
  impose rank-two outer/static-local guards. Extend bounded all-valid symbolic projection and
  explicit requiresBatchSize, unresolved layouts and one B; retain static/length branches.
  `GraphPreparation.Projection` keeps complete-graph demand and validates ALL contexts before
  analyzers. Local IDs remain scoped, never outer slots; no CPU geometry interpretation.
- CPU `CpuCapabilityProvider.supports` and `internal/prepare/CpuPartitionPreparer.analyze` dispatch
  symbolic inputs to ordinary batch BEFORE Scan. Add Scan qualification and ordinary/Scan unit
  composition in existing analysis. `internal/lowering/CpuPartitionLowering` batchLayouts/
  batchByteSizes/lowerOrdinaryBatch/supportsOrdinaryBatch and `internal/codegen/emit/CpuAggregateEmitter.emitOrdinaryBatch` own bounded outer AD closure.
- CPU `internal/lowering/CpuGeneralScanLowering` checkTypeShape/analyzePointwise/analyzeShaped,
  shapedResources/Facts/Geometry currently bind static spans/cuts/banks. Prepare named-axis
  provenance (including transpose/shared accumulators), not numeric-size roles. Evolve existing
  `internal/ir/CpuGeneralScanIr.View/Cut/Stages`: symbolic recipes versus baked facts.
  `CpuAccessPlan.BatchBinding` supplies constant/factor extents/strides. shapedResources uses checked
  width products/sums: four-byte internal alignment + fixed disjoint regions + one B/frame give
  EXACT constant+factor*B frame/offset/total recipes via `CpuRepresentationRecipes.BatchByteSize`,
  including zero slope, with no rounding term. Scan producing-site SUM needs no five-limb scratch;
  outer `CpuAggregateLowering.BatchGeometry` keeps conditional 0/48 bytes. Reuse requires all-B
  dominance. Stronger alignment/conditional elision/non-affine Scan scratch needs the decision
  gate before Ready; no fake affine bound, god recipe tree or Runtime change.
- CPU `internal/prepare/CpuPartitionPreparer.analyzeGeneralScan`, `CpuPartitionPreparationPlan`
  UnitRuntimeFacts/unit validation and `CpuPartitionFinalizer`: exact fixed/run-size declarations
  before slots; finalize afterward. Explicit symbolic mode even for [B,0]; structural zero-byte
  workspaces retain slots/creators. Existing creators take actual bytes/alignment; no B topology.
- CPU `internal/codegen/emit/CpuGeneralScanEmitter.Shaped` emit/loop/load/store/denseAddress,
  `CpuGeneralScanMatmulEmitter.emit/stride` and `CpuGeneralScanReductionEmitter.emit` bake extents,
  singleton/empty decisions and strides. Separate symbolic emission loads extents/strides/frame/
  cut/carry offsets from existing long[] entry packing at invocation/necessary row setup, then
  direct time/row/feature/contributor loops. No per-step/element allocation/maps/boxing/dispatch.
- CPU `internal/executable/CpuGeneralScanPreparedExecutable` fixed spans/hooks/bindCompatible/
  Invocation pack {time,elements}. Retain recipes; use existing state-aware hooks with actual
  state.geometry() bytes, bind B once, fresh numeric/access/address/carry arrays and unchanged
  long[] entry ABI. Keep typed carriers, full-capacity overlap checks and direct segment entry.
  Existing sibling `CpuPreparedExecutable.bindOrdinaryBatch` binds ordinary numeric recipes;
  `CpuPreparedPartitionExecutable` state-aware hooks/children compose them with Scan, not a new runner.
- Engine `PreparedExecution` preindexes only outer numeric checks: include scoped unused/sibling
  locals. `AdvancedEngine.runOrdinaryBatchOpen` keeps fresh full BatchBindingPlan.bind/private
  obligations → numeric/RunMemoryGeometry.resolve/aggregate bounds → ALL current-storage validation
  → exact-prefix borrow → sole Runtime lease/run → actual RunResult.Publication descriptors.
  No memo, late outward-close gate or Compiler changes.

Identity: envelope/family 71, static canonical v1, `scan-shaped-v4`, `scan-matmul-v4` and
pointwise/fixed/ordinary keys AND generated bytes stay unchanged. Symbolic mode alone
adds canonical `:symbolic-batch-v1` and baked `|scan-symbolic-batch-v1=` in
`CpuGeneralScanIr.encodedKernelIr/bakedSpecializationIdentity`, consumed by BOTH
`CpuKernelSpecialization.classIdentityBytes/compatibilityBytes` (ordinaryBatch stays false for Scan).
Key all baked dtype/body/role/demand/mapping/recipe/packing/control facts, not instance B/resolved
bytes/slots/addresses. `CpuClassFileKernelGenerator.validate/verify/defineClassBytes` and
`CpuGeneratedKernelArtifactStore.readEnvelope` keep exact fingerprint/metadata/name/entry checks;
no decoder/schema change. Test static↔symbolic/changed-recipe rejection; no diagnostic-text identity.

## Acceptance criteria

- Actual Compiler-generated reverse, independently qualified forward/reverse bodies; both
  traversals, joint final/emission and emission-only nonuniform seeds, ties/reused occurrences,
  disconnected invariants and exact saved TANH/SIGMOID outputs. Compare all outputs/gradients
  with independent ordinary unrolling and specialized clean Java.
- 64→40→1→64 plus B=0,1,F,H and B=F=H: stable compile/prepare/artifact/code/routes/slots/positions;
  actual time/batch/features stride B*F, frame B*H, ALL port layouts/flags/order/identities and current
  input-view descriptors. Parameter-shaped accumulation, never B×parameter. Assert exact demanded
  row/shared bytes and zero dead history bytes/stores, including reverse residuals; shared [F,H]
  history is [T,F,H], not row-banked.
- Inclusive cut lifetimes/size dominance hold for EVERY B or regions stay disjoint. All count/
  stride/span/offset/padding/aggregate overflow rejects before borrow/allocation, including unused
  trailing local/outer domains behind zero; no per-B generation.
- Separate T=0 plan: no body/hoisted work, raw distinct final-carry copy, dInitial=final seed,
  empty stacks/dX and typed-zero invariant gradients. B=0 separately/with T=0: nonempty shared
  work/dW executes; K=0 MATMUL/SUM typed zero, mean(empty) NaN, actual mean divisor B*H.
  Retain singleton/raw-copy, special-value, signed-zero and represented saved-site semantics.
- Fresh arrays/resources; legal read aliases, rejected writable overlap/capacity tails; reverse
  attempt-all cleanup/suppression, failed run/no publication, retry, concurrent mixed-B/open-result
  isolation and close/publication races. Numeric-only/model-free Runtime, one RunState/lease,
  borrowed caller storage, Engine prepared-before-input precedence.
- Replace only valid symbolic rejection assertions with positives. PrepareContextTest's
  scanRejectsEveryDynamicLocalIncludingUnusedFormalAndUnusedSiblingEvenAtZeroTime and
  GraphPreparationTest's laterPartitionDynamicUnusedFormalFailsBeforeTheFirstAnalyzer keep malformed
  unused/mismatched-symbol/binding negatives. Retain static/ordinary/length metadata and CPU length,
  unsupported body/dtype/layout, cold/copy/wire, higher-order/residual-root rejection gates.

## Validation

Authorized worker commands. New fixtures: CpuSymbolicScanGeneratedKernelTest (CPU internal.codegen.emit),
EngineSymbolicScanTest (Engine), CpuSymbolicScanConformanceTest (conformance),
EngineCpuSymbolicScanIntegrationTest (integration). Extend existing PrepareContextTest/
GraphPreparationTest and CPU Scan capability/lowering/preparation/execution tests. All performance
opt-ins stay unset/false except the probe. Shell commands use this task-local option array:

```bash
scan_checks=(--max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false)
./gradlew :modules:prepare:test --tests '*PrepareContextTest' --tests '*GraphPreparationTest' :backends:cpu:test --tests '*CpuGeneralScan*Test' --tests '*CpuSymbolicScanGeneratedKernelTest' :modules:engine:test --tests '*EngineSymbolicScanTest' --tests '*EngineOrdinaryDynamicBatchTest' "${scan_checks[@]}"
./gradlew :testing:backend-conformance:test --tests '*CpuSymbolicScanConformanceTest' --tests '*CpuGeneralScanConformanceTest' --tests '*CpuOrdinaryDynamicBatchConformanceTest' :testing:integration-tests:test --tests '*EngineCpuSymbolicScanIntegrationTest' --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuOrdinaryDynamicBatchIntegrationTest' "${scan_checks[@]}"
./gradlew :modules:prepare:test :backends:cpu:test :modules:engine:test :modules:prepare:javadoc :backends:cpu:javadoc :modules:engine:javadoc "${scan_checks[@]}"
```

Last command: stabilized worker modules ONCE; reviewer reuses evidence. Class-File tests check
clean-Java algorithm/loops/dataflow, saved loads, dynamic M/K/N, hidden calls/allocation/boxing/
reflection/dispatch and static key/byte goldens. One probe: future CPU CpuSymbolicScanTimingTest +
Engine EngineSymbolicScanTimingTest; existing forwarded default-false dynamicBatch flag.
Separate cold compile/prepare/generate/define, BIND (full semantic/scoped/numeric/storage admission),
KERNEL generated/direct forward+actual reverse with matched resources, and Engine fresh-run/result/
materialization. Fixed cases: T3/F4/H3 at B1/40/64 and T4/F32/H32 at B64; 100 warmup iterations per
variant, three alternating paired samples of 300 iterations each. At most 30s per test body,
60s aggregate including cold setup/warmup; censor/report unfinished cases at cap. One JVM/module,
no adaptive loops, retries/forks/campaign or parity promise; unsupported environment is reported.

```bash
./gradlew :backends:cpu:test --tests '*CpuSymbolicScanTimingTest' :modules:engine:test --tests '*EngineSymbolicScanTimingTest' --no-parallel --max-workers=1 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=true
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc "${scan_checks[@]}"
```

After the original Engine cold-setup failure, the user explicitly approved ONE Engine-only
environment follow-up, without source/build edits or CPU timing repetition:

```bash
JAVA_TOOL_OPTIONS='--add-modules=jdk.incubator.vector' ./gradlew --no-daemon :modules:engine:test --tests '*EngineSymbolicScanTimingTest' --no-parallel --max-workers=1 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=true
```

The root command above is MAIN's ONCE source-matched join after review: JDK26/existing native library,
no rebuild. Unsupported/missing environment stops the gate; do not weaken skips. Record commands,
counts/skips, identity/resource evidence and timing limits, not speed/negligible-B-overhead claims.

## Documentation and review impact

Independent Class C pass finalizes Javadoc, partition-preparer/CPU/running-models guides, runtime-
prepare-backend explanation, API support-status and glossary Symbolic Scan/Symbolic batch binding/
Run memory geometry. MAIN owns final status integration; new normative decisions need approval.
Narrow Model exception: `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Scan.java`
class support-status Javadoc ONLY; correct execution status, retain cold/persistence/NN gates. No executable/API/header edit.

## Result

Status: Complete. Homogeneous F32 symbolic Scan forward AND actual Compiler reverse work through
one public compile/prepare for changing B, fixed T/F/H. Existing owners only: 25 implementation
paths plus targeted documentation; no new production type/API, Compiler/Runtime executable,
build/dependency/native change. Static six-Scan and 17 ordinary/static key/identity/Class-File
witnesses remain identical. Independent code and documentation reviews APPROVE, zero open P1/P2.

MAIN's final source-matched configured `test javadoc` PASS in 5m42s, 119 tasks (40 executed,
79 up-to-date). Current included-project XML: 4944 tests, 0 failures/errors, 53 skips; unchanged
UP-TO-DATE results are reused, not claimed fresh. Post-fix FULL CPU: 1253/40 skips; Prepare95/0,
Engine151/2, conformance66/0, integration214/1, architecture25/0. Skips:42 CPU/Engine optional or
conditional cases,10 optional Metal observer cases,1 host-ineligible tuning case. Required native
Metal cases ran; existing library hash unchanged. Javadoc passed with94 CPU/46 Metal warnings
and2 incubator warnings, not warning-free. Documentation and affected Javadocs finalized.

Original bounded probe: CPU four cases completed (XML2.948s), Engine cold setup failed missing
the Vector module (0.275s), preserved honestly. User-approved Engine-only environment follow-up
PASS: four cases, 100 warmups/three paired300-call samples, XML6.046s, no censorship or CPU rerun.
Aggregate bodies9.269s satisfy30s/body and60s total caps. BIND and fresh-run/materialization
samples are separate from kernels; samples vary and do NOT prove parity, a causal explanation
or negligible B overhead. No further performance campaign is authorized.

Evidence root `/private/tmp/synaptik-prepare0014-impl.a04A4j`: `handoff.txt`, `acceptance-matrix.txt`,
`final-code-review.txt`, `docs-final.txt`, `engine-only-result.txt`, `main-root-join.log` and
`root-test-results.tgz`. Final token/scope/link checks PASS; prior dirty work, HEAD/origin/index
unchanged. Memo, lengths/skipping, NN/Training and other excluded workflows remain future gates.
No required implementation follow-up, commit/push or successor launch.
