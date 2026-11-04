# Task CPU0047: Absorb private [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) transposes into [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) reads

## Reader context

Here “private” means that CPU analysis has proved no other consumer needs the transpose result.
`CpuMatmulLowering` gives the existing MATMUL emitter the original backing value with swapped
read strides; it does not transfer ownership or expose a public alias. The Tensor expression
still describes a transpose, but the selected executable needs no separate transpose buffer.
Public, saved or multiply used results retain their stores. OpenBLAS is a different callee:
its canonical-input requirement can still require a declared copy.

The current [private-transpose explanation](../../../../backend-guide/cpu-backend.md#private-transpose-inputs-of-matmul)
shows backing addresses, before/after resources and the native-copy limit. The source-array
mistake in the Result is about `fromFlatArray` copying input values, not failed prepared reuse.

## Status

Complete — MAIN accepted the bounded repair and targeted review on 2026-10-28.

## Change class

Class C: cold read mapping, resource declarations and generated/native route composition.
MAIN implements/reviews under the user's explicit context-isolation exception; no architectural
or numerical exception is granted.

## Goal and [scope](../../../../glossary.md#term-scope)

For one or two adjacent, private [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `PERMUTE [1,0]` inputs before [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), read their original
backing [buffers](../../../../glossary.md#term-buffer) with swapped source [strides](../../../../glossary.md#term-stride) instead of executing separate copies. Preserve existing
MATMUL [bias](../../../../glossary.md#term-bias)/[activation](../../../../glossary.md#term-activation) alternatives, native-route eligibility and all arithmetic/[rounding](../../../../glossary.md#term-rounding) sites.
Reuse existing geometry, [bindings](../../../../glossary.md#term-binding), generated MATMUL [IR](../../../../glossary.md#term-intermediate-representation-ir) and entry [ABI](../../../../glossary.md#term-application-binary-interface-abi); add no production type.

## Non-goals

No public [view](../../../../glossary.md#term-view-meanings) [aliasing](../../../../glossary.md#term-aliasing), general affine-chain absorption, symbolic/batched [transpose](../../../../glossary.md#term-transpose) [admission](../../../../glossary.md#term-admission),
[Scan](../../../../glossary.md#term-cumulative-scan) changes, new [OpenBLAS](../../../../glossary.md#term-openblas) transpose [API](../../../../glossary.md#term-application-programming-interface-api), provider/native [ABI](../../../../glossary.md#term-application-binary-interface-abi) change, code generator/[schema](../../../../glossary.md#term-schema) change,
new tuning policy, performance campaign, [commit](../../../../glossary.md#term-commit) or push. Unsupported/observable cases remain split.
The current provider takes [canonical](../../../../glossary.md#term-canonical-form) non-transposed [matrices](../../../../glossary.md#term-matrix-meanings): its selected native route may still
need one declared input [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), never an extra preceding PERMUTE [buffer](../../../../glossary.md#term-buffer).

## Contracts

Start at [the authority index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), and [backend](../../../../glossary.md#backend) contract
[Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>,
[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>,
[OpenBLAS provider](../../../../architecture/contracts/backend-execution.md#openblas-provider)<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>.
The existing shared consumer is defined by [runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare) contract
[Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
and [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>.
Backend-owned cold [fusion](../../../../glossary.md#term-fusion) must preserve public/saved/[fan-out](../../../../glossary.md#term-fan-out) obligations; Runtime receives fixed
executables and declared resources, not [graphs](../../../../glossary.md#graph) or per-[operation](../../../../glossary.md#operation) decisions.

## Dependencies and integration

- Depends on: CPU0046 and Compiler0016 Complete, accepted by MAIN.
- Conflicts with: [CPU](../../../../glossary.md#term-cpu) [decomposition](../../../../glossary.md#term-decomposition), [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [lowering](../../../../glossary.md#lowering)/recognition/native selection, affected tests/docs.
- Parallel group: None; no concurrent writer or measurement.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted
  dirty tree `/private/tmp/synaptik-cpu-repairs.8TWk5O/first-two-repairs-accepted.json`.
- Integration order: verified brief → source/tests → focused/[module](../../../../glossary.md#term-module)/[conformance](../../../../glossary.md#term-conformance) gates → targeted review/docs.
- Integration validation: CPU tests/Javadoc once excluding only the two identified unconditional
  legacy benchmark methods by a temporary external Gradle test filter; focused MATMUL conformance.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

[Ready](../../../../glossary.md#term-ready) verification: MAIN checked predecessor results, actual accepted dirty baseline and protected
evidence, unchanged HEAD/origin/index, current contracts and no competing source frontier. CPU0047
is the sole active task. The previous full-[run](../../../../glossary.md#term-run) performance failure is not a correctness grant;
no benchmark retry or threshold change is allowed here.

## Files and symbols

Production paths are under `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`:

- `lowering/CpuMatmulLowering.java`: prefix privacy proof, backing/read [descriptors](../../../../glossary.md#term-descriptor) and existing geometry.
- `lowering/CpuPartitionLowering.java`, `CpuPartitionDagDecomposer.java`: bounded seed delegation and
  projection of external-consumer obligations; never lose an original cross-partition requirement.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [partition glossary](../../../../glossary.md#partition))</span>
- `lowering/CpuSpecializedSubgraphRecognizer.java`: include proved prefix members in existing [epilogue](../../../../glossary.md#term-epilogue) facts.
- `route/nativeblas/openblas/CpuOpenBlasRouteSelector.java`: permit the proved prefix unit while retaining
  current canonical/native and explicit-copy checks.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Matching focused [CPU](../../../../glossary.md#term-cpu) tests and `testing/backend-conformance` [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) strategy coverage.
- Existing `EngineCpuMatmulStrategyIntegrationTest`: shared Prepare and repeated heap-backed runs,
  including publication of the transpose; no Engine implementation change.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [heap glossary](../../../../glossary.md#term-heap); [publication glossary](../../../../glossary.md#term-publication); [transpose glossary](../../../../glossary.md#term-transpose); [Engine glossary](../../../../glossary.md#term-engine))</span>
- CPU guide/[Javadoc](../../../../glossary.md#term-javadoc) and this brief/master/roadmap; no authority or [dependency](../../../../glossary.md#term-dependency) change is intended.

## Acceptance criteria

1. Left/right/both private rank-two inputs produce one MATMUL unit, original backing boundaries,
   swapped actual strides, no PERMUTE resource/copy, and correct typed/empty-domain results.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [strides glossary](../../../../glossary.md#term-stride))</span>
2. Public, outside-unit/partition and multiply-consumed transpose results stay materialized;
   malformed or wider geometry does not acquire unsupported lowering.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [transpose glossary](../../../../glossary.md#term-transpose); [lowering glossary](../../../../glossary.md#lowering))</span>
3. Bias/activation suffix alternatives and existing bare MATMUL plans remain available. Scalar,
   vector-eligible and parallel orchestration still follow the existing MATMUL selector.<span data-glossary-navigation="true"> ([Bias glossary](../../../../glossary.md#term-bias); [activation glossary](../../../../glossary.md#term-activation); [Scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
4. Qualified native selection can retain its one required canonical copy, with the original backing
   and effective read mapping in its exact workload identity. No native provider contract changes.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
5. Equivalent bare strided MATMUL generates identical class bytes; no hidden helper, allocation,
   additional arithmetic or runtime transpose is introduced by absorption.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>

## [Validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused [lowering](../../../../glossary.md#lowering)/[decomposition](../../../../glossary.md#term-decomposition)/recognizer/native-selector and generated execution tests while
developing. Final [CPU](../../../../glossary.md#term-cpu) tests/[Javadoc](../../../../glossary.md#term-javadoc) use false generalScan.performance,
generalScan.pointwisePerformance and dynamicBatch.performance properties, `--max-workers=1 --console=plain`.
An external temporary init script excludes exactly
`CpuPartitionDagGeneratedEvidenceTest.fiveFixedHeapForksAndAggregateStayWithinGate` and
`CpuPartitionDagGeneratedEvidenceTest.freshFusionEvidenceRetainsEverySampleAndBothRequiredComparisons`;
all other tests in that class stay enabled. Do not change build sources or repeat those timings.
Run affected [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) tests once; no repository-wide/native rebuild.
Run `:testing:integration-tests:test --tests '*EngineCpuMatmulStrategyIntegrationTest'` for
the actual [Compiler](../../../../glossary.md#term-compiler) → shared [Prepare](../../../../glossary.md#prepare) → [Engine](../../../../glossary.md#term-engine) consumer after [module](../../../../glossary.md#term-module) validation. Retry only an
identified failed correctness assertion after its repair; do not repeat successful module gates.

## Documentation and review impact

[Planning](../../../../glossary.md#term-planning) profile for coordination; [Backend](../../../../glossary.md#backend)-guide profile for a concrete before/after [buffer](../../../../glossary.md#term-buffer) example
and native-copy limitation. Update affected implementation [Javadoc](../../../../glossary.md#term-javadoc), review glossary terms, and
record MAIN's targeted actual-diff review under the explicit context exception.

## Result

Status: Complete

MAIN accepted on 2026-10-28 under the explicit context exception, not an independent-context review.
Five production files preserve private backing reads, external-consumer obligations, [epilogue](../../../../glossary.md#term-epilogue)
membership and native eligibility. Four test files plus a new eight-[test fixture](../../../../glossary.md#adr-ci-and-qualification) cover left/right/both
transposes, all six numeric types on segments, [FLOAT32](../../../../glossary.md#term-float32) [heap](../../../../glossary.md#term-heap) arrays, padded/offset and valid empty reads,
[publication](../../../../glossary.md#term-publication)/[fan-out](../../../../glossary.md#term-fan-out) barriers, shared backings, route copying and repeated public [Engine](../../../../glossary.md#term-engine) runs.
Equivalent bare strided [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) has identical [specialization](../../../../glossary.md#term-specialization) and generated class bytes; no emitter,
[schema](../../../../glossary.md#term-schema), public [API](../../../../glossary.md#term-application-programming-interface-api), [dependency](../../../../glossary.md#term-dependency), native [ABI](../../../../glossary.md#term-application-binary-interface-abi) or authority contract changed. No timing claim is made.

[Validation](../../../../glossary.md#term-validation): [CPU](../../../../glossary.md#term-cpu) 1,353 tests, zero failures/errors, 52 standard skips; exactly the two historical
timing methods named above were externally excluded, so this is not an unfiltered-default result.
CPU [Javadoc](../../../../glossary.md#term-javadoc) passed with 94 existing out-of-[scope](../../../../glossary.md#term-scope) warnings. MATMUL/[OpenBLAS](../../../../glossary.md#term-openblas) [conformance](../../../../glossary.md#term-conformance): 16/16 pass;
Engine integration: 2/2 pass. Its initial new assertion incorrectly mutated the source array copied
by `fromFlatArray`; updating actual [Tensor](../../../../glossary.md#tensor) storage fixed the test without an Engine change.
Successful owning gates were reused. [Logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings): `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0047-*`.

Documentation/Javadoc, glossary and actual-source review are complete; the CPU guide explains the
three-[buffer](../../../../glossary.md#term-buffer) example, privacy barriers and native-copy limitation. Existing repeated logical
MATMUL input-ID [admission](../../../../glossary.md#term-admission) is not broadened; backing-ID collisions keep a required copy. The current
OpenBLAS provider can still require one [canonical](../../../../glossary.md#term-canonical-form) copy per input. General affine/symbolic absorption
and the [residual](../../../../glossary.md#scan-residual-port-and-history-demand) [Scan](../../../../glossary.md#term-cumulative-scan) performance gap remain outside this repair. No benchmark, [commit](../../../../glossary.md#term-commit) or push ran.
HEAD/origin/index and all 1,176 protected predecessor evidence files were preserved. Next: CPU0048.

Changed files: the five production paths listed above; new `CpuMatmulTransposeAbsorptionTest`,
existing `CpuOpenBlasRouteSelectorTest`, `CpuMatmulStrategyConformanceTest`,
`CpuOpenBlasRouteConformanceTest` and `EngineCpuMatmulStrategyIntegrationTest`; CPU guide, glossary,
this new brief, CPU master and roadmap. No obsolete production type or API was removed.

## Standalone [publication](../../../../glossary.md#term-publication)

The user selected publication of only CPU0047 on 2026-10-28. The isolated [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) is based on
`6da23a0b42f015979a88f5e84416799f5ada3f40` and excludes the uncommitted general-[Scan](../../../../glossary.md#term-cumulative-scan),
BatchGeometry, Compiler0016 and CPU0046 changes. Static [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) needs no new source [dependency](../../../../glossary.md#term-dependency)
from that series. Shared guide/glossary/master/roadmap files contain only the corresponding
CPU0047 edits in this payload, not their other locally accepted changes.

Isolated [validation](../../../../glossary.md#term-validation) used `cpu0047-publish-validation.gradle`: [CPU](../../../../glossary.md#term-cpu) 49/49, [conformance](../../../../glossary.md#term-conformance) 16/16 and
[Engine](../../../../glossary.md#term-engine) 2/2 passed without skips; CPU [Javadoc](../../../../glossary.md#term-javadoc) passed. This is additional validation of the isolated
publication tree, not a repeat claim about the larger accepted working tree. Detailed command/[log](../../../../glossary.md#term-log):
`/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0047-publish-validation.log`.

Independent clean-context publication review: APPROVE, zero open P1/P2 findings. It checked the
actual isolated 15-path payload and reused the successful validation; no test or benchmark rerun.
