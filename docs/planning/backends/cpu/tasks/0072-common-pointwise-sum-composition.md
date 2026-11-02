# CPU0072: Common pointwise and SUM composition

## Status and change class

Complete, MAIN accepted implementation/review and combined module/Javadoc/named gates 2026-10-29.
Banach's independent Class C source/Javadoc/documentation/provenance APPROVE has no open findings.
The historical Avicenna planning approval and MAIN Ready/GO authorized CPU0072 only; writer closed.
Class C: generated composition, represented arithmetic, exact saved sites and resource invariants.
Next priority is complete backward-versus-optimized-Java code comparison, not another timing run.
No source/measurement frontier or automatic successor is authorized.

## Goal, scope and exclusions

Emit the ordinary compiled body under Scan traversal/bindings, without duplicate Scan math.
Share current pointwise composition/normalization and aggregate ownership through CPU0071's seam.
Retain distinct ordinary exact SUM/MEAN and Scan sequential SUM; retire the Scan reducer after closure.

No new Scan SIMD/parallel capability, body family/dtype/layout, K splitting, vector-reduction
reassociation, accuracy selector, Compiler body optimization, new worker protocol, runtime access
layer, body graph or public API. Ordinary workers and fixed/native routes remain unchanged.
Retain static all-valid F32/F64, symbolic all-valid F32 and static/symbolic length F32/F64, each
with independently qualified forward/actual Compiler reverse. F32/F64 mean FLOAT32/FLOAT64;
general composition is the direction, not a claim that all ordinary operations become supported.

Workers/current per-submission Job stay unchanged. Optional Job reuse is not a dependency/blocker;
later Scan SIMD/worker use requires [separate gates](../existing-emitter-composition-plan.md#simd-admission-and-parallel-composition-gates).

## Contracts and Ready gates

Read AGENTS, CPU0071's [seam](0071-common-matmul-binding-foundation.md#implemented-seams-and-approved-policy)/[Result](0071-common-matmul-binding-foundation.md#result), source/tests. Root
[Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
[scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts); backend
[CPU routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes) and
[Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract);
backend [CPU0071 approved static MATMUL compatibility cutover](../../../../architecture/contracts/backend-execution.md#cpu0071-approved-static-matmul-compatibility-cutover);
Scan [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
[Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates),
[Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
[Empty domains and numeric resources](../../../../architecture/contracts/recurrent-scan.md#empty-domains-and-numeric-resources),
[Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates),
[Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories),
[First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse)
and [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates);
Runtime [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).

MAIN approved helpers, retention/evidence policy and base on 2026-10-29. No production type,
public signature or Runtime handoff changed. A necessary type/handoff/byte-shape/namespace change
would require explicit user approval before implementation. `CpuAggregateIr.Kind.SUM` exact-state
guards remain mandatory; CPU0071's exception permits no further cutover or schema/version change.

## Dependencies and integration

- Depends on: MAIN-accepted Complete CPU0064/0065/0070/0071, Compiler0015/0016 and Prepare0015;
  independent planning approval and MAIN Ready/GO satisfied before the completed implementation.
- Conflicts with: common pointwise/aggregate/carrier/Scan/cache/resource and CPU0048–0052 writers.
- Parallel group: None; clean sole implementation writer, independent targeted review.
- Common base revision: HEAD=origin/main `07a281edd3086efde2853211b71b48288db027e1` plus accepted
  dirty work; MAIN snapshot `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu72-start.json` (2533 files,
  empty index); before-source/plans/contracts/root/roadmap `/private/tmp/synaptik-cpu0072-main.OHshGm`.
- Integration order: accepted CPU0071 → Ready/GO → frozen artifacts → extraction → independent
  review/fixes → stabilized module/Javadoc → MAIN joins/acceptance. No automatic launch.
- Integration validation: commands below, links/anchors/whitespace, capability/resource/byte retention.
- Shared-document integration owner: MAIN; final documentation/status exclusively delegated to Banach.

Historical admission: all seven predecessors Complete, CPU0071 independently accepted, 659 backup
and 31 CPU0071 source hashes/protected evidence preserved, HEAD/origin/index unchanged.
Final writer and review are closed; no further launch follows this completion.

## Files, current package map and proposed helper seams

Source: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`.

Implemented package-local seams: `CpuPartitionLowering.validatePointwise` and
`CpuAggregateLowering.normalizedAxes/reductionForm/keepDimensions` share current pointwise and
exact Dimension/axis/target facts, reusing existing opcode/immediate/batch-access normalization.
`CpuClassFileKernelGenerator.emitBoundPointwise` orders ordinary scalar/vector and flat/Shaped
Scan loads/math/stores; `CpuAggregateEmitter.emitRepresentedSum` owns sequential reset/fold/tail.
The reviewed signatures remain in the frozen source manifest below; no package/type/public API added.
The [binding/evidence policy](../existing-emitter-composition-plan.md#cpu0072-extraction-and-evidence-policy)
preserves entry versus producing-site stores, raw/direct-singleton bytes and generation-only
callbacks. `Shaped/Stage/View` remain adapters, never common math signatures. Actual singleton
filtering/count/resources stay in `BatchGeometry.bind`; IR/cuts/schedules/cache/fallback stay owned.

## Acceptance criteria

1. Both callers use common instruction/axis/target normalization and bound family emission;
   remove duplicated Scan cases as they qualify. Common math accepts no Scan `Shaped` semantic
   owner and does not delegate to a renamed Scan reduction. Remove `CpuGeneralScanReductionEmitter`
   after complete current forward/actual-reverse closure. Other inventory families stay untouched.
2. Ordinary SUM/MEAN retain `CpuExactSumEmitter` and guarded exact forms, one final rounding and
   exact rational MEAN/count: symbolic F32 five limbs/48 bytes, F64 34/280, static domain-sized.
   Scan SUM keeps increasing selected-axis fadd/dadd and zero exact-state workspace.
   Preserve raw copy/sumToShape singleton, empty +0, NaN/Inf/signed zero and F64 subnormals.
   `[1e20,1,-1e20]` gives exact 1 versus sequential 0 in F32/F64; use matching-algorithm oracles.
3. Preserve selected-forward exact TANH/SIGMOID/state bits, demanded/dead histories and fan-out,
   producer/reduction and epilogue represented sites, previous-carry-before-step histories,
   positional ties, two seeds/traversals and parameter-shaped shared carries. Activity dominates
   both loads and dependent contributor math for ROW(0), ROW(1) and SHARED/any-active work.
4. Keep CPU0071 setup dominance, typed layout/species distinctions and explicit carrier/address
   mappings. No new allocator, per-step geometry repack/callback/allocation, runtime graph/operation
   access/re-lowering, tensor-buffer allocation, T-fold graph/code expansion or extra RunState.
   No compulsory copy; producer completion, old/new carry versions and inclusive all-B lifetimes
   precede cut reuse. Reserve layout locals once before allocation; retain setup dominance.
5. Retain v9 direct carriers/temporal bases, explicit-B/zero-base proofs, cell address/prefix
   sharing, indexed contractions and proved bank-free schedule/fallback. Keep ordinary scalar/
   SIMD/tiled paths and workers intact; do not make preserved optimized paths inherit fallback
   decoding/dispatch overhead. Shared vector code alone admits no Scan vector route.
6. Independently qualify retained static/symbolic/length F32/F64 forward/actual Compiler reverse:
   current read/write predicates, mixed roles, empty K/T/B/features, 64→40→1→64 reuse, poison
   padding, saved sites, aliases/overflow, preflight-before-writes, retry/close/concurrency.
   Unsupported interior/binder/dtype/layout cases still reject, including symbolic all-valid F64.
7. Preserve pre-slot/zero-byte requirements, canonical keys, both identity streams, binary names,
   ABI/packs and ALL generated bytes: schema/family 71, shaped-v4, static MATMUL-v5, static length-v1/
   retained-MATMUL-v2, symbolic-v3, combined-v9, pointwise/fixed/ordinary batch-v1/typed-v1.
   Compatible old artifacts still hit and prior stale versions reject; no actual B/L/slots/addresses
   in identity. Keep every local reservation/emission order, including currently unused locals.

## Validation

After Ready/GO, freeze actual old ClassFiles, both streams/keys, ABI/carriers/species and full
geometry/resources/bindings BEFORE production edits. Add a CPU0072 baseline at accepted CPU0071
values for its 226 + 21 ordinary controls, including its 68 previously changed rows; never rewrite
CPU0071 historical artifacts. Extend current pointwise/SUM producer/tail/copy/singleton/exact-MEAN
controls. Expected bytes come from immutable before-source/artifacts, never candidate generation
or name-normalized diffs. Species-mismatched vector rows remain unqualified, not waived scalar rows.
Focused development, then one stabilized affected-module run; last two commands belong to MAIN:

```bash
./gradlew :backends:cpu:test --tests '*CpuPointwisePartitionLoweringTest' --tests '*CpuPointwiseGeneratedKernelTest' --tests '*CpuAggregateLoweringTest' --tests '*CpuAggregateGeneratedKernelTest' --tests '*CpuGeneralScanLoweringTest' --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuGeneralScanExecutionTest' --tests '*CpuGeneralScanStageScheduleTest' --tests '*CpuMatmulBindingSnapshotTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*CpuSymbolicScanConformanceTest' --tests '*CpuMatmulStrategyConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuSymbolicScanIntegrationTest' --tests '*EngineCpuMatmulStrategyIntegrationTest'
```

Use focused tests for raw singleton versus one-active-of-many, saved activation injection,
actual AD/ROW(1)/SHARED poison, empty domains and changing-B reuse. Compare whole ClassFiles with
specialized clean scalar/ordinary-vector Java: algorithm/loops/dataflow/setup, calls/allocation/
boxing/reflection/dispatch. Finite corpus needs source review of every changed emit/allocation path.
Reuse checks, report skips/limits; no root/native/performance run, all timing selectors disabled.

MAIN-accepted decision: literal executable/identity/ABI/resource/binding equality transfers
accepted evidence with its existing limits; no new numerical/performance qualification or broad
per-fixture certification. Use the [four-group evidence table](../existing-emitter-composition-plan.md#cpu0072-evidence-transfer-table),
not another ledger/planning task. Equality passed for the recorded finite corpus. Historical
Java-parity gaps do not block pure extraction; CPU0071 timing grants no SUM/optimal-parity claim.
No new timing protocol is needed for unchanged work. If work must change, STOP at compatibility/
scope gates; any later timing requires a separately approved finite matched protocol, never a retry.
Post-integration priority: [explain remaining backward gap by complete code comparison](../existing-emitter-composition-plan.md#post-integration-actual-scan-evaluation),
before optional Job reuse; not just repeated benchmarks. All previous timing budgets remain closed.

## Documentation, review and result

Affected Javadocs/CPU guide reviewed; General, Planning and Backend guide profiles applied.
Targeted glossary review found no new meaning. No contract, ADR, dependency or Runtime change.

## Result

Completed 2026-10-29: common facts-only normalization, ordinary scalar/vector and flat/Shaped
pointwise composition, common represented-sequential SUM; ordinary exact SUM/MEAN retained.
Six production owners changed; Scan reducer retired, recoverable from MAIN's immutable before-source.
The 16-path source-owned freeze is `/private/tmp/synaptik-cpu0072-source-freeze.YHQXJ1/source-freeze-final.json`;
it covers seven production paths, six test paths, one frozen TSV, CPU guide and this brief.
Evidence root `/private/tmp/synaptik-cpu0072-evidence.TU1EmY`: 278 whole ClassFiles/both identity
streams, 194 geometry/resource/pack and 60 value-binding witnesses match; ABI/resources/v9 unchanged.
Original 1105-file baseline stays immutable; eight identity-text binding diagnostics were corrected
only via verified 186-file before-source/off-tree javac recovery and eight exact old ClassFiles.
Supplement receipt: `/private/tmp/synaptik-cpu0072-binding-correction.zRb6Cx/receipt.json`; no self-baseline.
Focused 285-test run: 281 passed, one new axis-fixture failure, three skips; corrected target-only
fixture passed seven tests. Subsequent normalization/retention and F32/F64 cancellation checks PASS.
The listed CPU test/Javadoc command FAILED: 1393 tests, one stale-inventory failure, 54 skips;
One-entry test-only correction PASS (one test), no whole-module repeat:
`./gradlew :backends:cpu:test --tests '*CpuInternalPackageInventoryTest.exposesOnlyTheAuthorizedSupportedAndInternalTypeInventory'`.
Javadoc PASS with 94 existing warnings. Banach P2 closure/overall APPROVE; production hashes unchanged.
MAIN's listed named joins PASS: conformance 19 / Engine 26 tests, zero failures/errors/skips (5s).
Finite controls and changed-path review transfer existing evidence with species/old parity limits;
zero CPU0072 measurements, all old timing budgets closed. No wider family/SIMD/parallel/Job claim.
Documentation/Javadocs finalized; no unresolved review issue. Next: complete backward-versus-matched
optimized-Java code comparison under the linked policy; no new Ready task, launch or timing grant.
Status: Complete
