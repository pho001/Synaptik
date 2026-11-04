# Task CPU0049: Ordinary [producer](../../../../glossary.md#term-producer)/reduction [fusion](../../../../glossary.md#term-fusion) and executable epilogues

## Status

Complete — MAIN accepted 2026-10-31 after independent approval and final gates.
[CPU](../../../../glossary.md#term-cpu) frontier None; CPU0050–0052 Draft/held. No automatic successor, measurement or stage/[commit](../../../../glossary.md#term-commit)/push.

## Change class

Class C — generated-code/hot-loop composition and exact-state/resource invariants;
clean implementation and independent targeted code/[Javadoc](../../../../glossary.md#term-javadoc) review required.

## Goal

Remove private [producer](../../../../glossary.md#term-producer) and reduction-result [buffers](../../../../glossary.md#term-buffer) in one bounded ordinary [CPU](../../../../glossary.md#term-cpu) unit:
typed [pointwise](../../../../glossary.md#term-pointwise) producer → existing exact SUM/[MEAN](../../../../glossary.md#term-mean) → typed output-cell [epilogue](../../../../glossary.md#term-epilogue).
Conceptual [F32](../../../../glossary.md#floating-point-precision-and-rounding) `[2,3]`: `gradient.mul(normalized).sum([1])` on `[1,2,3,4,5,6]` and all twos → `[12,30]`.

## [Scope](../../../../glossary.md#term-scope) and non-goals

- Fully static homogeneous FLOAT32/FLOAT64; portable generated scalar, one executing range.
  SUM/MEAN full, single-axis and normalized multi-axis forms, including kept dimensions and
  empty axis lists. At least one producer or epilogue must be removed; bare aggregates stay unchanged.<span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [scalar glossary](../../../../glossary.md#term-scalar); [MEAN glossary](../../../../glossary.md#term-mean); [axis glossary](../../../../glossary.md#term-axis); [dimensions glossary](../../../../glossary.md#dimension); [producer glossary](../../../../glossary.md#term-producer); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- Producers use the existing bounded, same-type pointwise lowering/instruction vocabulary;
  one single-use chain, no CAST, predicate/mask/selection/state crossing or new movement absorption.
  Keep the existing eight-node unit bound, including reduction and tail, not a partition-size cap.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [lowering glossary](../../../../glossary.md#lowering); [mask glossary](../../../../glossary.md#term-boolean-mask); [node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition))</span>
- Epilogues are only the current recognized ADD (either operand order), RELU, or ADD→RELU.
  ADD's external operand broadcasts to the output; producer-only, tail-only and combined
  forms are one capability. Other terminals stay split.
- Boundaries are resolved canonical dense reads/writes or already-supported right-aligned
  broadcast reads, with nonnegative offsets and proved injective output. Use existing access
  normalization; no new layout admission. Typed <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [injective glossary](../../../../glossary.md#term-injective); [normalization glossary](../../../../glossary.md#term-normalization); [layout glossary](../../../../glossary.md#layout); [admission glossary](../../../../glossary.md#term-admission))</span>`float[]`/`double[]` and checked [`MemorySegment`](../../../../glossary.md#term-memorysegment)
  carriers retain current heap/native/mixed span/alignment/alias checks.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [alignment glossary](../../../../glossary.md#term-alignment); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- No SUM-to-Shape fusion, PROD/extrema/Boolean/advanced/statistical fusion, low/integer/mixed
  types, symbolic-B fusion, SIMD/parallel fusion, native work, defaults/tuning changes or public
  Tensor/Compiler semantics. Exclusions keep their existing supported route or rejection.
  No Scan widening, row-region extension, duplicate fuser/math or runtime abstraction.
  Paused Scan performance and every historical timing budget remain closed.<span data-glossary-navigation="true"> ([SUM-to-Shape glossary](../../../../glossary.md#sum-to-shape); [fusion glossary](../../../../glossary.md#term-fusion); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [Boolean glossary](../../../../glossary.md#data-type--datatype); [SIMD glossary](../../../../glossary.md#simd); [Tensor glossary](../../../../glossary.md#tensor); [Compiler glossary](../../../../glossary.md#term-compiler); [Scan glossary](../../../../glossary.md#term-cumulative-scan); [widening glossary](../../../../glossary.md#term-widening); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Contracts

Read the root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>
first, then only these applicable headings:

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span> and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Backend](../../../../glossary.md#backend) execution [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>,
  [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> and
  [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
- Foundational [modules](../../../../glossary.md#term-module) [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives),
  [Aggregate floor](../../../../architecture/contracts/foundational-modules.md#aggregate-floor)<span data-glossary-navigation="true"> ([Aggregate floor glossary](../../../../glossary.md#term-aggregate-floor))</span> and
  [Exact and discrete floor](../../../../architecture/contracts/foundational-modules.md#exact-and-discrete-floor).
- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine) [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.
- Backend execution [Physical region and admission](../../../../architecture/contracts/backend-execution.md#physical-region-and-admission)<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>
  under CPU0073: ordinary exact SUM/MEAN are excluded from <span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean))</span>**that row-region mechanism**.
  This task does not extend it: `CpuRowMapReductionIr` and represented [Scan](../../../../glossary.md#term-cumulative-scan) SUM remain unchanged.
  [CPU0073's explicit distinction](0073-row-map-multi-reduction-fusion.md#non-goals) confirms it
  did not close CPU0049; no other predecessor narrative is needed.

Existing ordinary [fusion](../../../../glossary.md#term-fusion) permission suffices; no authority amendment/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) task is needed.
Stop on contract ambiguity/conflict or necessary [scope](../../../../glossary.md#term-scope)/type expansion; do not widen CPU0073.

## Dependencies and integration

- Depends on: CPU0048 and retained CPU0073–0076, all Complete.
- Conflicts with: CPU0050–0052 and any aggregate/pointwise lowering, recognition, emission,
  generated identity, resource/binder or shared-document writer; serialize these scopes.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [lowering glossary](../../../../glossary.md#lowering); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parallel group: None; one serial writer, read-only independent review.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty work, not merged predecessors.
- Integration order: [planning](../../../../glossary.md#term-planning)/[admission](../../../../glossary.md#term-admission) → implementation/review → worker gate → MAIN join/acceptance (closed).
- Integration validation: named conformance/Engine join below once; no repository-wide gate
  absent a boundary/dependency change.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; exclusive serial delegation only.

MAIN verified [Ready](../../../../glossary.md#term-ready)/dependencies/base/conflicts, 2026-10-31 13:50 UTC; user authorized implementation.

## Files, types and composition seams

Production paths use `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.

- **Sole new production type:** `internal.ir.CpuAggregateFusionIr`, [immutable](../../../../glossary.md#term-immutable) `CpuPortableKernelIr`
  composing `CpuAggregateIr`, [producer](../../../../glossary.md#term-producer)/tail `CpuKernelIr` facts and explicit bindings. Separate
  domains cannot use the old instruction-free/two-boundary aggregate or one pointwise domain.
  No new math/opcodes/semantic graph or row-map extension; package inventory updated.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [pointwise glossary](../../../../glossary.md#term-pointwise); [graph glossary](../../../../glossary.md#graph))</span>
- `internal.lowering.CpuAggregateLowering`, `CpuPartitionLowering`, `CpuPartitionDagDecomposer`:
  reuse geometry/axes/state and pointwise normalization; full-use proofs precede cuts/resources.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- `internal.lowering.CpuSpecializedSubgraphRecognizer.reduction` and
  `internal.ir.CpuSpecializedSubgraph.ReductionEpilogue`: extend former `ORDINARY_SPLIT` facts
  only for executable tails; retain unsupported/bare facts, budgets and baseline association.
- `internal.codegen.emit.CpuAggregateEmitter`, `CpuExactSumEmitter`, `CpuClassFileKernelGenerator`,
  `CpuScalarEmitter`/carrier emission: pointwise contribution/tail sites reuse exact-state math;
  no clones, represented-SUM substitution or runtime callbacks. ExactSum/Loop emitters unchanged.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Existing `CpuPortableKernelIr` permits, `CpuPortableRoutePlan`, `CpuKernelSpecialization`,
  preparation/representation owners and <span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>`CpuPreparedExecutable`: extend two-boundary packing
  to explicit producer/tail inputs/final output, bases/spans/aliases/scratch; no new public protocol.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>

## Acceptance criteria

1. Producer instructions retain each FLOAT32/FLOAT64 producing-site rounding and operand order.
   Feed that represented local to the current exact SUM/MEAN state, never an unrounded product,
   FMA, sequential represented sum or reassociated dot product. Preserve current limb calculation,
   flags, domain count, reset/update/finalization, ties-to-even result, specials and zero behavior.
   Static FLOAT64 remains domain-sized; symbolic F32 five-limb/48-byte and typed F64
   34-limb/280-byte paths are untouched, not templates for a changed static algorithm.<span data-glossary-navigation="true"> ([Producer glossary](../../../../glossary.md#term-producer); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [rounding glossary](../../../../glossary.md#term-rounding); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [dot product glossary](../../../../glossary.md#term-dot-product); [limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
2. Finish and round SUM/[MEAN](../../../../glossary.md#term-mean) to its result [dtype](../../../../glossary.md#data-type--datatype) **before** ADD/RELU; round ADD before RELU.
   An epilogue cannot consume exact accumulator limbs or an unrounded quotient. Preserve both
   ADD operand orders and current RELU NaN/signed-zero behavior through the existing emitter.<span data-glossary-navigation="true"> ([epilogue glossary](../../../../glossary.md#term-epilogue); [accumulator glossary](../../../../glossary.md#accumulator-type); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
3. Full-graph public, saved/autograd, fan-out, cross-partition and physical-alias uses prohibit
   removing their stores. Reject that fusion boundary; independently legal portions may still
   fuse. Do not duplicate producers, replay activations or erase a live producer/reduction result.
   Retain legality/profitability separation and existing safe split selection when proof fails.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [autograd glossary](../../../../glossary.md#autograd); [fan-out glossary](../../../../glossary.md#term-fan-out); [partition glossary](../../../../glossary.md#partition); [alias glossary](../../../../glossary.md#aliasing-and-arena); [fusion glossary](../../../../glossary.md#term-fusion); [activations glossary](../../../../glossary.md#term-activation))</span>
4. Selected plans remove only private full-domain producer/result buffers and declarations;
   retain exact aggregate scratch bytes, alignment, slots/creators and run isolation. Analysis
   fixes topology/resources before slots; finalization only binds/emits. Ranges mean complete
   output cells, including legal empty/subranges; never split contributors across workers.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [alignment glossary](../../../../glossary.md#term-alignment); [run glossary](../../../../glossary.md#term-run); [topology glossary](../../../../glossary.md#term-topology))</span>
5. Test full/single/multi axes, keepDimensions, scalar/no-axis, selected/unselected zero extents,
   empty MEAN, singleton, cancellation, halfway rounding, finite-product overflow, NaN/infinities,
   signed zero and subnormals. Include a product-rounding counterexample and an epilogue case
   that differs if accumulator precision leaks. Compare raw results to current split execution
   and an independent numerical oracle; retain current permitted NaN representation rules.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [scalar glossary](../../../../glossary.md#term-scalar); [singleton glossary](../../../../glossary.md#term-singleton); [cancellation glossary](../../../../glossary.md#term-cancellation); [finite glossary](../../../../glossary.md#term-finite); [overflow glossary](../../../../glossary.md#term-overflow); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [subnormals glossary](../../../../glossary.md#term-subnormal); [precision glossary](../../../../glossary.md#term-precision); [numerical oracle glossary](../../../../glossary.md#adr-ci-and-qualification))</span>
6. Cover both dtypes, all admitted carrier combinations/offsets/broadcasts, alias rejection before
   writes, public/saved/fan-out barriers and unsupported-layout/type/strategy/terminal fallbacks.
   Assert selected units/resource removal, not only equal output or recognition.
   CPU0048-proven counts retain constant-write selection/resource removal: new recognition must
   not resurrect contributor loops, exact-state workspace or private materialization. No new folding.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [constant glossary](../../../../glossary.md#term-constant); [materialization glossary](../../../../glossary.md#materialization))</span>
7. Use an additive ordinary fused structural discriminator in both compatibility/class-identity
   streams, covering topology, typed sites, axes, boundary accesses and baked code-shaping facts.
   Keep global envelope 71 and unrelated family projections; do not reuse row-region projection
   72/markers. Reject stale fused artifacts; preserve bare/split/Scan/CPU0048 controls and reuse.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion); [identity glossary](../../../../glossary.md#term-identity-meanings); [artifacts glossary](../../../../glossary.md#term-artifact); [Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
8. Maintain a specialized clean-Java oracle with the same typed sites, exact limb algorithm,
   prebound carriers and preallocated state. Inspect generated ClassFiles against its loop/dataflow:
   one contribution traversal, completed-result tail, no materialized private buffer, new math
   switch, hidden fallback/helper overhead, allocation/boxing/reflection/maps or callback dispatch
   per element/operation. Keep allowed JDK calls explicit.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow); [callback glossary](../../../../glossary.md#term-callback); [dispatch glossary](../../../../glossary.md#term-dispatch); [element glossary](../../../../glossary.md#term-element); [operation glossary](../../../../glossary.md#operation); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>

## [Validation](../../../../glossary.md#term-validation)

Added `internal.codegen.emit.CpuAggregateFusionGeneratedKernelTest`/`CpuAggregateFusionCleanJavaOracle`;
extended existing tests, retaining row-map/[snapshot](../../../../glossary.md#term-snapshot) and CPU0048 count-fold/[constant](../../../../glossary.md#term-constant)-write controls.

```bash
./gradlew :backends:cpu:test --tests '*CpuAggregate*Test' --tests '*CpuSpecializedSubgraphRecognizerTest' --tests '*CpuPartitionDagDecomposerTest' --tests '*CpuStaticTypedOneCount*Test' --tests '*CpuConstantWriteGeneratedKernelTest'
./gradlew -I /private/tmp/synaptik-cpu0048.G18rM6/no-campaigns.gradle -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false :backends:cpu:test :backends:cpu:javadoc
./gradlew :testing:backend-conformance:test --tests '*CpuProfileFreeConformanceTest' --tests '*CpuSharedExternalReadConformanceTest' --tests '*CpuStaticTypedOneCountConformanceTest' :testing:integration-tests:test --tests '*EngineCpuComposedValidationIntegrationTest' --tests '*EngineCpuLongPartitionIntegrationTest'
```

One worker [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) gate and MAIN named join cover [fused](../../../../glossary.md#specialization-and-fusion) numerics/barriers and fresh-input
prepared reuse, including actual [Compiler](../../../../glossary.md#term-compiler) [gradients](../../../../glossary.md#term-gradient); no [Engine](../../../../glossary.md#term-engine)/Compiler production changes.
The accepted init script excludes only these two methods of
`io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPartitionDagGeneratedEvidenceTest`:
`fiveFixedHeapForksAndAggregateStayWithinGate` and
`freshFusionEvidenceRetainsEverySampleAndBothRequiredComparisons`. All other tests remain eligible;
other performance opt-ins stay off. Record **filtered**, not unfiltered/full PASS; review reuses
successful evidence, and final Javadoc follows final edits.

After untimed/source review, the sole authorized ordinary generated/direct
hot-path comparison: one [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) `[64,256]` MUL→[axis](../../../../glossary.md#term-axis)-1 SUM, prebound equal carriers/state,
six [warmup](../../../../glossary.md#term-warmup) and nine alternating paired [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), 25 calls/sample and a 60-second cap. Verify outputs,
separate setup/entry/[kernel](../../../../glossary.md#kernel) costs, report dispersion/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit) limitations and structural explanation
of any gap; no speed threshold, extra [forks](../../../../glossary.md#term-fork-meanings)/retries, [Scan](../../../../glossary.md#term-cumulative-scan) case or historical-budget reuse.
Protocol completed/closed; no further measurement. Check links/anchors, size and scoped whitespace.

## Documentation and review impact

Independent Class C source/[Javadoc](../../../../glossary.md#term-javadoc)/[backend](../../../../glossary.md#backend)-guide review complete; changed contracts documented.
Targeted glossary review required no new term. No public [API](../../../../glossary.md#term-application-programming-interface-api), authority/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) or [module](../../../../glossary.md#term-module)-edge change.

## Result

MAIN accepted Complete, 2026-10-31. Darwin FINAL source/docs/protocol/results APPROVE: 0 P1/P2,
all six findings closed. Static [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) [scalar](../../../../glossary.md#term-scalar) SUM/[MEAN](../../../../glossary.md#term-mean) + ADD/RELU delivered; typed [rounding](../../../../glossary.md#term-rounding),
exact state, complete-use barriers, safe split fallbacks and CPU0048 [constant](../../../../glossary.md#term-constant)-write precedence retained.
Files: 24 worker-owned (14 production, nine tests, one [CPU](../../../../glossary.md#term-cpu) guide).
External `/private/tmp/synaptik-cpu0049-impl.IpgUDg/` retains exact paths/[hashes](../../../../glossary.md#term-hash) in
`source-freeze-01.json` and development failures/corrections in `implementation-evidence.md`.
[Canonical](../../../../glossary.md#term-canonical-form) [topology](../../../../glossary.md#term-topology)/types/accesses retain existing aggregate facts; baked geometry is separate in
both [identity](../../../../glossary.md#term-identity-meanings) streams. ClassFile/clean-Java exact-[limb](../../../../glossary.md#exact-superaccumulator-and-limbs) loop/[dataflow](../../../../glossary.md#term-dataflow)/hidden-overhead checks passed.
Commands above used [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26, `--offline --no-daemon --max-workers=1 --console=plain`; performance flags false.
Original **filtered** gate: BUILD SUCCESSFUL, 13m2s, 23 tasks/two executed; 237 suites, 1476 tests,
0 failures/errors, 56 skips (1420 executed); the two excluded timing methods are outside counts.
[Javadoc](../../../../glossary.md#term-javadoc) PASS, 94 warnings (baseline 94). Prior missing status recovered; MAIN same-command
exit-0/2s confirmation was all 23 UP-TO-DATE, not a second functional execution.
MAIN named join exit 0/9s, 46 tasks/eight executed: [conformance](../../../../glossary.md#term-conformance) three suites/five tests and [Engine](../../../../glossary.md#term-engine)
two suites/11 tests, all PASS with zero skips. Focused-09 had 55 passing tests, zero skips.
Sole diagnostic: one [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), 3.416s, all raw outputs verified; nine pairs/[outlier](../../../../glossary.md#term-outlier) retained.
F32/F64 generated/direct medians: 215.357/314.383 and 811.713/877.930 us; paired [median](../../../../glossary.md#term-median) ratios
0.641657/0.915203, ranges 0.548517–0.821644 / 0.875952–2.194482. [Nonstationary](../../../../glossary.md#term-nonstationary) observations,
not steady profitability, parity, universal speed or a proved [C2](../../../../glossary.md#term-c2) cause; entry observations not subtracted.
All 24 freeze hashes match after gates. Baseline 2587→2590: 24 existing changed/three new,
2563 other originals identical, no removals/unexpected files; HEAD/index unchanged.
No unresolved issue/follow-up; frontier None, CPU0050–0052 held; no automatic successor/measurement/[commit](../../../../glossary.md#term-commit)/push.
Status: Complete
