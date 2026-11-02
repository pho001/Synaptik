# Task CPU0049: Ordinary producer/reduction fusion and executable epilogues

## Status

Complete — MAIN accepted 2026-10-31 after independent approval and final gates.
CPU frontier None; CPU0050–0052 Draft/held. No automatic successor, measurement or stage/commit/push.

## Change class

Class C — generated-code/hot-loop composition and exact-state/resource invariants;
clean implementation and independent targeted code/Javadoc review required.

## Goal

Remove private producer and reduction-result buffers in one bounded ordinary CPU unit:
typed pointwise producer → existing exact SUM/MEAN → typed output-cell epilogue.
Conceptual F32 `[2,3]`: `gradient.mul(normalized).sum([1])` on `[1,2,3,4,5,6]` and all twos → `[12,30]`.

## Scope and non-goals

- Fully static homogeneous FLOAT32/FLOAT64; portable generated scalar, one executing range.
  SUM/MEAN full, single-axis and normalized multi-axis forms, including kept dimensions and
  empty axis lists. At least one producer or epilogue must be removed; bare aggregates stay unchanged.
- Producers use the existing bounded, same-type pointwise lowering/instruction vocabulary;
  one single-use chain, no CAST, predicate/mask/selection/state crossing or new movement absorption.
  Keep the existing eight-node unit bound, including reduction and tail, not a partition-size cap.
- Epilogues are only the current recognized ADD (either operand order), RELU, or ADD→RELU.
  ADD's external operand broadcasts to the output; producer-only, tail-only and combined
  forms are one capability. Other terminals stay split.
- Boundaries are resolved canonical dense reads/writes or already-supported right-aligned
  broadcast reads, with nonnegative offsets and proved injective output. Use existing access
  normalization; no new layout admission. Typed `float[]`/`double[]` and checked `MemorySegment`
  carriers retain current heap/native/mixed span/alignment/alias checks.
- No SUM-to-Shape fusion, PROD/extrema/Boolean/advanced/statistical fusion, low/integer/mixed
  types, symbolic-B fusion, SIMD/parallel fusion, native work, defaults/tuning changes or public
  Tensor/Compiler semantics. Exclusions keep their existing supported route or rejection.
  No Scan widening, row-region extension, duplicate fuser/math or runtime abstraction.
  Paused Scan performance and every historical timing budget remain closed.

## Contracts

Read the root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
first, then only these applicable headings:

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Backend execution [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
  [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes) and
  [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
- Foundational modules [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives),
  [Aggregate floor](../../../../architecture/contracts/foundational-modules.md#aggregate-floor) and
  [Exact and discrete floor](../../../../architecture/contracts/foundational-modules.md#exact-and-discrete-floor).
- Runtime/Prepare/Engine [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).
- Backend execution [Physical region and admission](../../../../architecture/contracts/backend-execution.md#physical-region-and-admission)
  under CPU0073: ordinary exact SUM/MEAN are excluded from **that row-region mechanism**.
  This task does not extend it: `CpuRowMapReductionIr` and represented Scan SUM remain unchanged.
  [CPU0073's explicit distinction](0073-row-map-multi-reduction-fusion.md#non-goals) confirms it
  did not close CPU0049; no other predecessor narrative is needed.

Existing ordinary fusion permission suffices; no authority amendment/ADR task is needed.
Stop on contract ambiguity/conflict or necessary scope/type expansion; do not widen CPU0073.

## Dependencies and integration

- Depends on: CPU0048 and retained CPU0073–0076, all Complete.
- Conflicts with: CPU0050–0052 and any aggregate/pointwise lowering, recognition, emission,
  generated identity, resource/binder or shared-document writer; serialize these scopes.
- Parallel group: None; one serial writer, read-only independent review.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty work, not merged predecessors.
- Integration order: planning/admission → implementation/review → worker gate → MAIN join/acceptance (closed).
- Integration validation: named conformance/Engine join below once; no repository-wide gate
  absent a boundary/dependency change.
- Shared-document integration owner: MAIN; exclusive serial delegation only.

MAIN verified Ready/dependencies/base/conflicts, 2026-10-31 13:50 UTC; user authorized implementation.

## Files, types and composition seams

Production paths use `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.

- **Sole new production type:** `internal.ir.CpuAggregateFusionIr`, immutable `CpuPortableKernelIr`
  composing `CpuAggregateIr`, producer/tail `CpuKernelIr` facts and explicit bindings. Separate
  domains cannot use the old instruction-free/two-boundary aggregate or one pointwise domain.
  No new math/opcodes/semantic graph or row-map extension; package inventory updated.
- `internal.lowering.CpuAggregateLowering`, `CpuPartitionLowering`, `CpuPartitionDagDecomposer`:
  reuse geometry/axes/state and pointwise normalization; full-use proofs precede cuts/resources.
- `internal.lowering.CpuSpecializedSubgraphRecognizer.reduction` and
  `internal.ir.CpuSpecializedSubgraph.ReductionEpilogue`: extend former `ORDINARY_SPLIT` facts
  only for executable tails; retain unsupported/bare facts, budgets and baseline association.
- `internal.codegen.emit.CpuAggregateEmitter`, `CpuExactSumEmitter`, `CpuClassFileKernelGenerator`,
  `CpuScalarEmitter`/carrier emission: pointwise contribution/tail sites reuse exact-state math;
  no clones, represented-SUM substitution or runtime callbacks. ExactSum/Loop emitters unchanged.
- Existing `CpuPortableKernelIr` permits, `CpuPortableRoutePlan`, `CpuKernelSpecialization`,
  preparation/representation owners and `CpuPreparedExecutable`: extend two-boundary packing
  to explicit producer/tail inputs/final output, bases/spans/aliases/scratch; no new public protocol.

## Acceptance criteria

1. Producer instructions retain each FLOAT32/FLOAT64 producing-site rounding and operand order.
   Feed that represented local to the current exact SUM/MEAN state, never an unrounded product,
   FMA, sequential represented sum or reassociated dot product. Preserve current limb calculation,
   flags, domain count, reset/update/finalization, ties-to-even result, specials and zero behavior.
   Static FLOAT64 remains domain-sized; symbolic F32 five-limb/48-byte and typed F64
   34-limb/280-byte paths are untouched, not templates for a changed static algorithm.
2. Finish and round SUM/MEAN to its result dtype **before** ADD/RELU; round ADD before RELU.
   An epilogue cannot consume exact accumulator limbs or an unrounded quotient. Preserve both
   ADD operand orders and current RELU NaN/signed-zero behavior through the existing emitter.
3. Full-graph public, saved/autograd, fan-out, cross-partition and physical-alias uses prohibit
   removing their stores. Reject that fusion boundary; independently legal portions may still
   fuse. Do not duplicate producers, replay activations or erase a live producer/reduction result.
   Retain legality/profitability separation and existing safe split selection when proof fails.
4. Selected plans remove only private full-domain producer/result buffers and declarations;
   retain exact aggregate scratch bytes, alignment, slots/creators and run isolation. Analysis
   fixes topology/resources before slots; finalization only binds/emits. Ranges mean complete
   output cells, including legal empty/subranges; never split contributors across workers.
5. Test full/single/multi axes, keepDimensions, scalar/no-axis, selected/unselected zero extents,
   empty MEAN, singleton, cancellation, halfway rounding, finite-product overflow, NaN/infinities,
   signed zero and subnormals. Include a product-rounding counterexample and an epilogue case
   that differs if accumulator precision leaks. Compare raw results to current split execution
   and an independent numerical oracle; retain current permitted NaN representation rules.
6. Cover both dtypes, all admitted carrier combinations/offsets/broadcasts, alias rejection before
   writes, public/saved/fan-out barriers and unsupported-layout/type/strategy/terminal fallbacks.
   Assert selected units/resource removal, not only equal output or recognition.
   CPU0048-proven counts retain constant-write selection/resource removal: new recognition must
   not resurrect contributor loops, exact-state workspace or private materialization. No new folding.
7. Use an additive ordinary fused structural discriminator in both compatibility/class-identity
   streams, covering topology, typed sites, axes, boundary accesses and baked code-shaping facts.
   Keep global envelope 71 and unrelated family projections; do not reuse row-region projection
   72/markers. Reject stale fused artifacts; preserve bare/split/Scan/CPU0048 controls and reuse.
8. Maintain a specialized clean-Java oracle with the same typed sites, exact limb algorithm,
   prebound carriers and preallocated state. Inspect generated ClassFiles against its loop/dataflow:
   one contribution traversal, completed-result tail, no materialized private buffer, new math
   switch, hidden fallback/helper overhead, allocation/boxing/reflection/maps or callback dispatch
   per element/operation. Keep allowed JDK calls explicit.

## Validation

Added `internal.codegen.emit.CpuAggregateFusionGeneratedKernelTest`/`CpuAggregateFusionCleanJavaOracle`;
extended existing tests, retaining row-map/snapshot and CPU0048 count-fold/constant-write controls.

```bash
./gradlew :backends:cpu:test --tests '*CpuAggregate*Test' --tests '*CpuSpecializedSubgraphRecognizerTest' --tests '*CpuPartitionDagDecomposerTest' --tests '*CpuStaticTypedOneCount*Test' --tests '*CpuConstantWriteGeneratedKernelTest'
./gradlew -I /private/tmp/synaptik-cpu0048.G18rM6/no-campaigns.gradle -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false :backends:cpu:test :backends:cpu:javadoc
./gradlew :testing:backend-conformance:test --tests '*CpuProfileFreeConformanceTest' --tests '*CpuSharedExternalReadConformanceTest' --tests '*CpuStaticTypedOneCountConformanceTest' :testing:integration-tests:test --tests '*EngineCpuComposedValidationIntegrationTest' --tests '*EngineCpuLongPartitionIntegrationTest'
```

One worker module/Javadoc gate and MAIN named join cover fused numerics/barriers and fresh-input
prepared reuse, including actual Compiler gradients; no Engine/Compiler production changes.
The accepted init script excludes only these two methods of
`io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPartitionDagGeneratedEvidenceTest`:
`fiveFixedHeapForksAndAggregateStayWithinGate` and
`freshFusionEvidenceRetainsEverySampleAndBothRequiredComparisons`. All other tests remain eligible;
other performance opt-ins stay off. Record **filtered**, not unfiltered/full PASS; review reuses
successful evidence, and final Javadoc follows final edits.

After untimed/source review, the sole authorized ordinary generated/direct
hot-path comparison: one JVM, F32/F64 `[64,256]` MUL→axis-1 SUM, prebound equal carriers/state,
six warmup and nine alternating paired samples, 25 calls/sample and a 60-second cap. Verify outputs,
separate setup/entry/kernel costs, report dispersion/JIT limitations and structural explanation
of any gap; no speed threshold, extra forks/retries, Scan case or historical-budget reuse.
Protocol completed/closed; no further measurement. Check links/anchors, size and scoped whitespace.

## Documentation and review impact

Independent Class C source/Javadoc/backend-guide review complete; changed contracts documented.
Targeted glossary review required no new term. No public API, authority/ADR or module-edge change.

## Result

MAIN accepted Complete, 2026-10-31. Darwin FINAL source/docs/protocol/results APPROVE: 0 P1/P2,
all six findings closed. Static F32/F64 scalar SUM/MEAN + ADD/RELU delivered; typed rounding,
exact state, complete-use barriers, safe split fallbacks and CPU0048 constant-write precedence retained.
Files: 24 worker-owned (14 production, nine tests, one CPU guide).
External `/private/tmp/synaptik-cpu0049-impl.IpgUDg/` retains exact paths/hashes in
`source-freeze-01.json` and development failures/corrections in `implementation-evidence.md`.
Canonical topology/types/accesses retain existing aggregate facts; baked geometry is separate in
both identity streams. ClassFile/clean-Java exact-limb loop/dataflow/hidden-overhead checks passed.
Commands above used JDK 26, `--offline --no-daemon --max-workers=1 --console=plain`; performance flags false.
Original **filtered** gate: BUILD SUCCESSFUL, 13m2s, 23 tasks/two executed; 237 suites, 1476 tests,
0 failures/errors, 56 skips (1420 executed); the two excluded timing methods are outside counts.
Javadoc PASS, 94 warnings (baseline 94). Prior missing status recovered; MAIN same-command
exit-0/2s confirmation was all 23 UP-TO-DATE, not a second functional execution.
MAIN named join exit 0/9s, 46 tasks/eight executed: conformance three suites/five tests and Engine
two suites/11 tests, all PASS with zero skips. Focused-09 had 55 passing tests, zero skips.
Sole diagnostic: one JVM, 3.416s, all raw outputs verified; nine pairs/outlier retained.
F32/F64 generated/direct medians: 215.357/314.383 and 811.713/877.930 us; paired median ratios
0.641657/0.915203, ranges 0.548517–0.821644 / 0.875952–2.194482. Nonstationary observations,
not steady profitability, parity, universal speed or a proved C2 cause; entry observations not subtracted.
All 24 freeze hashes match after gates. Baseline 2587→2590: 24 existing changed/three new,
2563 other originals identical, no removals/unexpected files; HEAD/index unchanged.
No unresolved issue/follow-up; frontier None, CPU0050–0052 held; no automatic successor/measurement/commit/push.
Status: Complete
