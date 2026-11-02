# Task CPU0050: AveragePool2d backward compact-source fold

## Status

Complete — MAIN accepted 2026-10-31 after Dirac FINAL APPROVE (0 P1/P2) and all gates PASS.
Frontier None; CPU0051/0052 Draft/held. No automatic successor, further timing or stage/commit/push.

## Change class

Class C — generated hot-loop/binding/resource invariants; clean implementation and independent code/docs review.

## Goal

Remove F32/F64 AveragePool2d backward's expanded columns by reading stored, rounded perPosition through
existing fold machinery. Preserve Compiler recipe/API, typed divisor/DIV and fold additions.

## Scope and non-goals

- CPU-private broadcast/view/copy → FOLD2D, static homogeneous FLOAT32/FLOAT64 ONLY. K=KH*KW>1,
  P=OH*OW: physical `[N,C,P]`, logical `[N,C*K,P]`, output `[N,C,H,W]`. Match
  `PoolingGradientRules.average` expandDims/EXPAND/CONTIGUOUS/RESHAPE and proved redundant copies/views.
  Follow producer edges, not adjacency; at most eight members including fold, excluding DIV.
- Keep FOLD2D window checks: positive kernel/stride/dilation, nonnegative padding, both ceil modes, exact
  OH/OW/effective-kernel counts/shapes. Dense rank-three source/nonnegative offset; existing nonnegative
  injective output layouts. Prove spans/inverse arithmetic fit long; int needs full proof. No new zero domains.
- SCALAR/single range; float[]/double[]/checked native-order MemorySegment, all heap/native/mixed pairs.
  Unsupported layout/strategy, K=1, low/mixed/integer/BOOL, symbolic geometry, nonmatches,
  MaxPool/Pool3d/Conv/unrelated folds retain existing routes/rejections.
- No other-module production, authority/dependency/default/tuning changes, native rebuild, Scan work,
  broad campaign, automatic successor or stage/commit/push. BF16/F16 untouched, NOT newly qualified.

## Contracts

Use the [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read these headings; plans are not authority:

- Root: [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Backend execution: [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules), [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes), [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
- Foundational modules: [Exact and discrete floor](../../../../architecture/contracts/foundational-modules.md#exact-and-discrete-floor), [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives), [Aggregate floor](../../../../architecture/contracts/foundational-modules.md#aggregate-floor), [Composite inheritance](../../../../architecture/contracts/foundational-modules.md#composite-inheritance), [Model low-precision contract](../../../../architecture/contracts/foundational-modules.md#model-low-precision-contract).
- Compiler/autograd: [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation) (read-only recipe/saved-site owner).
- Runtime/Prepare/Engine: [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare), [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle), [`modules/runtime`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime), [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).

## Dependencies and integration

- Depends on: CPU0049 Complete (MAIN accepted 2026-10-31); accepted CPU0048/0073–0076 retained.
- Conflicts with: CPU0051/0052; fold/movement/decomposition/binding/resource/cache/shared-doc writers.
- Parallel group: None.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` PLUS accepted dirty tree, not HEAD source.
  MAIN actual-before `/private/tmp/synaptik-cpu0050.ylFXyT/` contains `before-sha256.json` (2590 files, empty
  index) and `before.tar.gz`; preserve unrelated work.
- Integration order: frozen source + independent review → MAIN named conformance/Engine join ONCE →
  sole diagnostic → final worker filtered CPU/Javadoc → MAIN acceptance.
- Integration validation: commands below once; boundary/dependency change requires replanning, not a routine root suite.
- Shared-document integration owner: MAIN; exclusive serial delegation only.

## Files, types and reuse seams

- **New/moved production types/packages: None.** `internal.ir.CpuFoldIr` and
  `internal.lowering.CpuFoldLowering.Geometry`: `kernelBroadcastInput`, old constructors false.
  `TwoDimensionalGeometry` owns windows; affine strides cannot express floor(q/K) for C>1,K>1.
  IR/Geometry/admission reject true-mode low; physical and logical source facts remain distinct.
- Existing lowering/decomposer/plan, fold emitter/reference and specialization owners handle
  proof/resources, inverse gather, checked binding and cache identity; no new ABI or math owner.
- Tests: `CpuBroadcastFoldGeneratedKernelTest`, `CpuBroadcastFoldCleanJavaOracle`,
  `CpuFoldCompatibilityControlTest`; existing artifact-store/composed-validation tests; consumers below.
  Exact changed paths/hashes: evidence `final-gate/source-freeze.json` and `task-delta.json`.

## Acceptance criteria

1. Keep materialized typed perPosition DIV and existing typed-one expand/contiguous/SUM divisor
   recipe, including CPU0048 selection. No host K divisor, reciprocal, repeated/fused DIV or widened
   saved value. Only subsequent raw repetition/copies disappear.
2. For output `(n,c,y,x)`, start represented +0, visit ky then kx ascending. Compute
   `a=y+padH-ky*dilH`, `b=x+padW-kx*dilW`; require nonnegative, stride-divisible a,b and
   `(oh=a/strideH,ow=b/strideW)` within exact OH/OW. Read `[n,c,oh*OW+ow]`; columns q maps
   to floor(q/K). At most one pooled position matches each kernel, preserving canonical q,p order.
   O(N*C*H*W*K); no all-columns/all-input scan, expanded odometer or generic fallback overhead.
3. Same-format sequential F32/F64 additions, no reassociation/FMA/new aggregate state. Padding/
   ceil tails add nothing, gaps stay +0; legal subranges leave other cells untouched. Empty outputs
   perform no source read or coordinate division/modulo.
4. Removed values are exact same-type private single-use raw mappings. Full-graph public/saved/
   autograd/fan-out/cross-partition/outside-unit/alias escapes block absorption; keep split storage,
   no duplication. perPosition stays a real boundary, including its public/saved/fan-out consumers.
5. Fold child: compact source + output, zero workspace/materialization. Removed values keep logical
   metadata but no CPU declaration/slot/creator/argument/store/hidden allocation. Prove before/after
   resources: N*C*P*dtypeWidth source, not N*C*K*P*dtypeWidth; retain constants/other units.
   Correct ValueId/carrier injection, compact spans/type/alignment/access/writable-output and complete
   input/output physical-overlap checks precede writes, including heap-segment aliases.
6. Both dtypes: N>1/C>1/K>1, nonsquare kernels, overlap/gaps, stride>1/dilation/padding/ceil tails,
   legal zero batch/channel/spatial extents, empty/subranges, offsets/strided outputs. Raw finite/
   zero parity against accepted split + independent oracle; separate NaN class (not payload), inf,
   signed zero, subnormal, cancellation/ties/overflow and division-rounding counterexample.
7. Assert topology/resources and deceptive-near-match fallback: wrong expansion axis/permutation/
   dimensions/window, noncanonical source, arithmetic/cast in chain, all protected-use barriers,
   interleaved unrelated nodes/partial chains, unsupported strategy and failed address/arithmetic
   proof. Negative binding tests prove no writes; prepared reuse proves fresh injection.
   True mode rejects BF16/F16 at IR/Geometry/admission; low chains retain original path/artifact,
   with no claim of low conformance.
8. Geometry-free canonical, selected-only `:kernel-broadcast-input-v1` fold encoding/fingerprint binds family/dtype/
   access/ranks/addition/physical-input mode. BOTH compatibilityBytes()/classIdentityBytes() inherit
   it; test names/store keys/reuse/stale rejection. Envelope71/fold projection52 unchanged; geometry/
   offsets/strides/ranges cold, existing carriers/access keyed. Every newly baked fact enters BOTH
   streams. False-mode encoding/packed ABI/keys/ClassFiles and CPU0048/0049/Scan controls unchanged.
9. BEFORE first production edit capture ordinary FOLD_AXIS/FOLD2D compatibility/class-identity
   bytes, keys/ClassFiles: F32 dense heap, low segment, mixed padded/dilated, F64/offset representatives.
   Include BOTH BF16 and F16 literal old keys/ClassFiles as compatibility controls ONLY, not numerical oracles.
   Hash accepted actual-before source/fixtures/toolchain; compare literal bytes after, using that build or
   audited isolated harness. Same-source comparisons or HEAD-only baselines are insufficient.
10. Optimal clean Java uses criterion 2 with equal prebound carriers/geometry and no expanded buffer.
    Inspect ClassFiles for matching algorithm/loop/dataflow, int/long proof, typed add/store; no hidden
    math/dispatch helpers, allocation/boxing/reflection/maps/callbacks per element. Checked segment
    access allowed; do not optimize unrelated ordinary folds.

## Validation

Use JDK 26; optional `--offline --no-daemon --max-workers=1 --console=plain`. Focused checks first;
final **filtered** CPU/Javadoc gate follows the join and sole diagnostic:

```bash
./gradlew :backends:cpu:test --tests '*CpuFold*Test' --tests '*CpuBroadcastFoldGeneratedKernelTest' --tests '*CpuPartitionDagDecomposerTest' --tests '*CpuPartitionDagResourceTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest' --tests '*CpuStaticTypedOneCount*Test'
./gradlew -I /private/tmp/synaptik-cpu0048.G18rM6/no-campaigns.gradle -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false :backends:cpu:test :backends:cpu:javadoc
```

MAIN join: `CpuBroadcastFoldConformanceTest` covers compact resources/binding/fail-before-write;
`EngineCpuAveragePoolBackwardIntegrationTest` covers actual F32/F64 AD, nonuniform seeds, reuse/barriers.
Both prove untouched BF16/F16 split paths, not low conformance. No consumer production changes.
Run after review, before timing; reuse success absent executable change/relevant failure or fix.

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuBroadcastFoldConformanceTest' --tests '*CpuProfileFreeConformanceTest' --tests '*CpuStaticTypedOneCountConformanceTest' :testing:integration-tests:test --tests '*EngineCpuAveragePoolBackwardIntegrationTest' --tests '*EngineCpuComposedValidationIntegrationTest' --tests '*EngineCpuLongPartitionIntegrationTest'
```

Verify init excludes ONLY `CpuPartitionDagGeneratedEvidenceTest.fiveFixedHeapForksAndAggregateStayWithinGate`
and `.freshFusionEvidenceRetainsEverySampleAndBothRequiredComparisons`. Other performance flags/env off;
disclose counts/skips. CPU0049 is not authority/proof.
After independent review and successful MAIN join, sole default-disabled diagnostic:

```bash
SYNAPTIK_CPU0050_DIAGNOSTIC=true ./gradlew :backends:cpu:test --tests '*CpuBroadcastFoldGeneratedKernelTest.boundedGeneratedVersusDirectDiagnostic'
```

One JVM/60-second body cap; F32/F64 output `[2,3,16,17]`, 3x3/stride2/pad1/dilation1/floor,
rounded source `[2,3,72]`, equal prebound heap carriers/preallocated output. Six warmup and nine alternating
measured pairs, 25 calls/arm; retain raw/class checks, all samples/failures. Separate generation/binding/
entry/kernel costs, no subtraction; disclose dispersion/JIT/structural limits. No parity threshold,
retry/fork/Scan/historical campaign. Check links/size/whitespace and actual-before/HEAD/index preservation.

## Documentation and review impact

Independent Class C review covers Javadocs, CPU guide `Current overlap fold family`/F32/F64 example and
glossary impact under documentation rules/general/backend/API profiles; final Javadoc follows edits.
MAIN owns plans; reuse executable evidence.

## Follow-up

Separate BF16/F16 fix required: old `CpuFoldEmitter.add` narrows each term, contrary to `Aggregate floor`/
`Model low-precision contract` (F32 accumulator, final narrowing). Repair accumulator/add/store with separate
compatibility/review and cancellation/rounding/AD qualification. CPU0050 neither fixes nor certifies low;
MAIN tracks this separately gated follow-up; CPU0051/0052 remain held.

## Result

MAIN accepted 2026-10-31; Dirac FINAL source/docs/protocol/result/gate-evidence APPROVE, 0 P1/P2.

- F32/F64 static SCALAR/single-range compact fold delivered via existing IR/Geometry boolean/emitter;
  stored DIV boundary retained, private chain buffers removed; int proof/long fallback, full-use/physical
  checks, both cache streams and literal old artifacts retained. Semantic/oracle/ClassFile checks PASS.
- 19 paths: 8 existing production + 7 tests (5 new/2 existing) + guide + 3 plans; no new production type.
  Exact paths/hashes in evidence `final-gate/source-freeze.json`/`task-delta.json`; 15 Java + guide freeze unchanged.
- Validation commands above, JDK26/offline/no-daemon/max-workers1/plain: MAIN join session55231 exit0/8s,
  6 conformance + 14 integration PASS, 0 skip; reused, not repeated.
- Sole diagnostic command above, DIAGNOSTIC=true: session43140 exit0/5s, body0.202s; all 18 pairs retained.
  G/D duration-median ratios F32 3.009318/F64 3.316406; strong F32 transition. Parity/cause UNRESOLVED;
  no steady-state/C2/speedup claim or new campaign.
- Final filtered CPU/Javadoc command above: session77597 true exit0/6m2s; 239 suites, 1493 total,
  1436 executed PASS, 57 skipped, 0 failures/errors. Exactly two historical timing methods excluded,
  outside those 57. CONTROLS=root/controls, CAPTURE/DIAGNOSTIC/STRUCTURE unset, old performance flags off;
  full settings in `final-gate/command.md`. All 160 before payloads byte-identical; 94 Javadoc warnings disclosed.
- Javadocs/guide independently approved; glossary unchanged (existing terms sufficient); no authority/ADR change.
  MAIN protection: 2590 original/2596 current, 2577 original hashes unchanged, 19 expected paths, no removal/
  unexpected change; HEAD/base unchanged, index empty. Known low final-narrowing defect remains separate/unqualified.
- Evidence root: `/private/tmp/synaptik-cpu0050.ylFXyT/`; `final-gate/completion-packet.md`,
  `sole-diagnostic/result-review-packet.md`, `MAIN-named-join.log`. No automatic next task/timing/commit/push.

Status: Complete
