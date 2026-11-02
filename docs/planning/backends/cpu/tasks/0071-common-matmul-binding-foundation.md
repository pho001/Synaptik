# CPU0071: Common MATMUL binding and setup foundation

## Status and change class

Complete, 2026-10-29. Independent corrected source/Javadoc review APPROVE; MAIN accepted
implementation and validation. Source writer closed; no active source or measurement frontier.
CPU0072 stays Draft/unlaunched. Class C: generated-code hot paths and temporal/resource proofs.

## Goal and scope

Make existing ordinary rank-two MATMUL and current scalar Scan use common family normalization
and bound contraction emission. Separate invocation setup from operation regions in existing
owners; retain ordinary direct/tiled scalar/SIMD and CPU0065 TANH. Move Scan's existing indexed/
cursor variants and activity semantics into common ownership, then retire its MATMUL subemitter.
Binding model and [non-fused example](../existing-emitter-composition-plan.md#non-fused-matmul-to-operation-boundary)
remain in the replacement plan.

No all-family rewrite, new Scan family/dtype/layout/route, ordinary symbolic F64 MATMUL, second
body graph, runtime access layer, per-step callback/allocation, worker change or new Runtime owner.
Existing Scan pointwise/reduction math is retained pending CPU0072, not falsely reported retired.

Priority remains CPU0071 → separately approved CPU0072. Scan retains traversal/temporal bindings;
`CpuWorkerGroup` and per-submission Job stay unchanged. Optional Job reuse is not a prerequisite.
Later synchronous worker composition needs the [separate gates](../existing-emitter-composition-plan.md#simd-admission-and-parallel-composition-gates).

## Contracts and Ready gates

Read AGENTS and affected source/tests. Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
and [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts) route to backend
[CPU routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes) and
[Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract);
Scan [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
[Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates),
[Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
[Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates),
[Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)
and [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates);
Runtime [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).
The implemented compatibility exception is owned only by backend execution
[CPU0071 approved static MATMUL compatibility cutover](../../../../architecture/contracts/backend-execution.md#cpu0071-approved-static-matmul-compatibility-cutover).
This brief records implementation/evidence, not authority or a successor launch.

No new production type/public API/framework or Runtime handoff was added; test fixtures are separate.
Future boundary changes require exact package/name/purpose/signature and USER approval.

## Dependencies and integration

- Depends on: CPU0070 independently reviewed/MAIN accepted; retained CPU0064/0065 and accepted
  Compiler0015/0016/Prepare0015; explicit user implementation approval and Ready gates above.
- Conflicts with: CPU MATMUL/carrier/loop/Scan/cache/resource and CPU0048–0052 writers.
- Parallel group: None; one clean implementation writer, then independent targeted review.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty work;
  MAIN's preservation receipt below verifies this accepted base, not an assumed merged base.
- Integration order: this cutover → independent authority review + MAIN Ready/GO → fresh clean
  code worker → independent code/docs review → CPU module/Javadoc + named conformance/Engine
  gates → optional single measurement/evidence reuse → MAIN acceptance; CPU0072 stays Draft.
- Integration validation: worker and MAIN commands below, documentation links/anchors,
  whitespace, changed-file/hash preservation and exact retained-capability checks.
- Shared-document integration owner: MAIN; final eight-document pass delegated to Ptolemy.

MAIN verified accepted predecessor/authority/signature gates and the 2518-file `cpu71-code-start.json`
base; accepted dirty work is not merged history. Immutable before-source:
`/private/tmp/synaptik-cpu0071-main.wsdT4h/cpu-src`; MAIN owns preservation and the final join.

## Files and symbols

Source prefix: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`.

- `lowering/CpuMatmulLowering.lower`/`BatchGeometry`, `ir/CpuMatmulIr`/`CpuAccessPlan`: checked normalization.
- `codegen/emit/` MATMUL/carrier/loop emitters and generator: separate entry setup from contraction.
- Scan lowering `analyzeShaped/analyzeSymbolic`, IR and `CpuGeneralScanEmitter.Shaped` adapters:
  temporal adaptation; preserve `CpuGeneralScanStageSchedule.order`, resources and existing fallback.
- `cache/CpuKernelSpecialization`/`CpuGeneratorSchema`, IR identity and named tests/oracles below:
  both identity streams, frozen artifacts and generated/direct evidence.

## Implemented seams and approved policy

Existing `CpuMatmulLowering.lower` and Scan lowering now call package-local `validateRankTwo`
for exact Shape/Dimension equations, never sampled B. `BatchGeometry` retains F32 guards.
`CpuMatmulEmitter.emitCursorContraction`/`emitIndexedContraction` own increasing full-K control
and represented math. Generation-only callbacks inline bytecode, never runtime calls; cursor
loaders fill typed locals and indexed loaders leave operands on stack. Callers retain typed +0/
address initialization and known-empty static K omission. Scan supplies temporal/activity/store
bindings without passing Shaped/Stage to common math. No public signature or production type added.

`CpuCarrierEmitter.prepareSegmentLayouts` still reserves six layout locals first; combined entries
omit unused initialization. Compatible vector species/byte order remain invocation-local.
`Shaped.prepareMatmulStrides` binds static constants outside time, not evolving scalar→dense
factors. Symbolic/combined selected stride handles are unchanged. The approved exception is
implemented in both specialization streams; retained identities/bytes remain validation targets,
not universal proof. Unexpected changed namespaces still require review.

## Acceptance criteria

1. Ordinary and Scan rank-two callers reach common MATMUL normalization and contraction owners.
   No math entry accepts Scan `Shaped`/Stage as its semantic owner or delegates back to a renamed
   Scan-only implementation. Existing unrelated high-rank/low/integral ordinary forms retain
   their guards and code; no universal-loop refactor is required.
2. CodeBuilder alone allocates JVM locals. Distinguish those locals from packed `long[]` indices
   and shared Runtime selections. Layout block is reserved once; geometry/species/layout and
   invariant-stride setup dominates use outside time, including empty/tail/fallback paths.
   Compatible dtype/access setup is reused; F32/F64/mask species cannot be conflated. Repeated
   generation-time prepare calls must not emit repeated runtime setup within time.
3. Common activity/proof inputs express ROW output gating, ROW(1) active K and SHARED any-active
   work. Active-K guard precedes **both** operand loads and dependent math; ordinary unguarded
   bindings remain valid. Preserve increasing contributors, full K, K=0 typed zero, old/next
   carry simultaneous lifetimes, original-time histories and selected-forward exact saved bits.
4. Non-fused MATMUL → operation uses its declared intermediate/demanded port, completes the
   producer domain before consumers/cut reuse and adds no mandatory copy. No callback, nested
   prepared executable, geometry repack, graph lookup/re-lowering or allocation per step,
   including tensor buffers; no T-fold graph/code expansion.
5. Preserve v9 stable carriers/temporal bases, explicit-B affine/zero-base proofs, cell sharing,
   indexed active-row/feature contractions and bank-free schedule/fallback, inclusive all-B cut
   lifetimes and physical final-port hazards. Retain CPU0065 vector TANH/tails and ordinary
   worker/range behavior. No Scan SIMD/parallel admission follows from common math.
6. Qualify all retained static/symbolic/length forward and actual Compiler reverse occurrences
   independently: both traversals/seeds, F32/F64 only where already admitted, ties/two carries,
   shared parameter sums, poison padding, zero/full/mixed/invalid lengths, T=0/B=0/1/empty K/
   zero features, 64→40→1→64 reuse, dense/scalar reads, alias/overflow/cleanup/retry/concurrency.
   Do not weaken ordinary batch F32-only guards or admit symbolic all-valid F64.
7. Remove `CpuGeneralScanMatmulEmitter` only after its callers qualify. Preserve SUM algorithms
   unchanged and unaffected keys/bytes; list every changed artifact, advance both affected
   identity streams and reject stale inputs per the approved policy. No actual B/L/addresses/
   slots enter canonical identity. No new mutable prepared state or Runtime handoff.

## Validation

Recorded validation commands below completed; this historical record requests no rerun:

```bash
./gradlew :backends:cpu:test --tests '*CpuMatmulBindingSnapshotTest' :backends:cpu:javadoc
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

MAIN capability-preservation checkpoint, once after independent review:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*CpuSymbolicScanConformanceTest' --tests '*CpuMatmulStrategyConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuSymbolicScanIntegrationTest' --tests '*EngineCpuMatmulStrategyIntegrationTest'
```

ClassFile/clean-Java checks cover loop/dataflow, setup dominance, activity-before-load and v9 proofs.
Historical evidence remains immutable; v9 timings do not qualify changed static shapes.
After source review and named gates MAIN launched `run-conditional.sh --main-prelaunch-go` once.
The USER-approved protocol used one measured JVM, eight rows
(F32/F64 × static all-valid/static-length × forward/actual Compiler reverse), original traversal
forward and sibling reverse, T4/B7/F3/H2. All-valid: `next=emit=h+x@W`; length mode:
`next=emit=tanh(x@W+h@U+bias)`, L=`[4,0,1,3,2,4,1]`. Baseline/new/direct-Java arms;
10000 warmup calls per arm/row, 15 measured triples × 1000 calls, cyclic arm order; hard 60-second
JVM watchdog deadline, no retry/repeat/additional fork. Storage was preallocated with before/after
checks. Cold compile/generate/define, allocation/binding and warmed entry are separate; Engine
admission was not measured. Generated arms include MethodHandle/spreader versus direct-call costs.
Legacy timing selectors, including dynamicBatch.performance, stay disabled; no root/native rebuild.

## Result

Completed 2026-10-29: common normalization/cursor/indexed math and static stride setup in existing
owners; `CpuGeneralScanMatmulEmitter` deleted after caller qualification. Five test-only
`CpuMatmulBinding*` classes provide snapshot/binding tests, fixture, clean Java and explicit harness.
Source freeze: 31 CPU-src paths (14 modified, 16 added, one retired), plus CPU guide/brief.
No new production type, numerical rule, resource/Runtime handoff, worker/Job or Scan route.

Both streams implement only static all-valid MATMUL v4→v5 and retained-MATMUL static-length
v1→length-aware-matmul v2, with stale definition/persisted-cache rejection. Non-MATMUL shaped-v4/
length-v1, symbolic-v3, combined-v9, pointwise/fixed and ordinary controls retain identities/bytes
in the recorded corpus; canonical projections/schema 71/packs/resources remain unchanged.
Finite-corpus preservation is evidence, not proof for every specialization.

Evidence root: `/private/tmp/synaptik-cpu0071-code.BvOdPR`; original captures/raw logs are immutable.

- Frozen matrix: 226 cases, 68 approved changes/158 exact retained; 194 geometry witnesses and
  eight actual old ClassFiles. Another 21 ordinary controls cover batch-v1 F32 forward/dX/dW and
  typed-v1 F64 arithmetic/SUM/MEAN/AD, recovered unmeasured from immutable before-source, not a
  current-source self-baseline. Vector golden bytes require captured 128-bit preferred species;
  other hosts explicitly report unqualified vector rows without dropping scalar checks.
- Full CPU/Javadoc PASS: 1385 tests, zero failures/errors, 53 reported skips; 94 pre-existing
  Javadoc warnings (`final-cpu-test-javadoc.log`). Subsequent species-test/two-Javadoc correction:
  three snapshot tests, no skips/failures/errors, Javadoc PASS (`review-correction-validation.log`).
  Independent targeted re-review APPROVE, no unresolved correctness/documentation findings.
- MAIN joins PASS: conformance 19/Engine 26 tests, no skips/failures/errors
  (`main-named-joins-authorized.log`). Initial sandbox cache-lock failure preceded tests
  (`main-named-joins.log`); authorized existing-cache launch succeeded in six seconds.
- Sole measurement exit 0, eight before/after checks PASS (`conditional-performance.log`).
  Budget consumed/closed: no repeat, retry, additional fork or inherited CPU0072 budget.
  New/old median ratios 0.9881–1.0313 establish no speedup or steady-state parity.
  F64 length-forward direct samples transition from about 30 µs through 16.311 µs to 0.83 µs;
  its aggregate ratio is not valid performance evidence. F32 length-reverse new samples also
  transition from about 32 µs to 0.98 µs; its direct arm has an early transient.
  Fixed warmup was insufficient for some arms; no compilation logs prove a C2 or other cause.
  Length-reverse new/direct medians about 1.50 F32/1.42 F64 remain an unresolved performance
  question, not correctness failure or proof of a new regression. All-valid ratios are likewise
  harness diagnostics, not optimal-Java or emitted-inner-loop parity proof.

Documentation impact: affected Javadocs and CPU/Scan explanations updated; root/owning exception
and planning statuses synchronized. General/Architecture/Planning profiles applied; targeted glossary
review found no new reusable meaning. No new ADR/architecture edge. Final docs reuse executable
evidence without JVMs. No required foundation follow-up; CPU0072 stays Draft and unlaunched.

Status: Complete
