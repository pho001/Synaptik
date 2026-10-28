# Task CPU0047: Absorb private rank-two transposes into MATMUL reads

## Status

Complete — MAIN accepted the bounded repair and targeted review on 2026-10-28.

## Change class

Class C: cold read mapping, resource declarations and generated/native route composition.
MAIN implements/reviews under the user's explicit context-isolation exception; no architectural
or numerical exception is granted.

## Goal and scope

For one or two adjacent, private rank-two `PERMUTE [1,0]` inputs before MATMUL, read their original
backing buffers with swapped source strides instead of executing separate copies. Preserve existing
MATMUL bias/activation alternatives, native-route eligibility and all arithmetic/rounding sites.
Reuse existing geometry, bindings, generated MATMUL IR and entry ABI; add no production type.

## Non-goals

No public view aliasing, general affine-chain absorption, symbolic/batched transpose admission,
Scan changes, new OpenBLAS transpose API, provider/native ABI change, code generator/schema change,
new tuning policy, performance campaign, commit or push. Unsupported/observable cases remain split.
The current provider takes canonical non-transposed matrices: its selected native route may still
need one declared input canonicalization, never an extra preceding PERMUTE buffer.

## Contracts

Start at [the authority index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), and backend contract
[Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes),
[OpenBLAS provider](../../../../architecture/contracts/backend-execution.md#openblas-provider).
The existing shared consumer is defined by runtime/Prepare contract
[Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
and [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution).
Backend-owned cold fusion must preserve public/saved/fan-out obligations; Runtime receives fixed
executables and declared resources, not graphs or per-operation decisions.

## Dependencies and integration

- Depends on: CPU0046 and Compiler0016 Complete, accepted by MAIN.
- Conflicts with: CPU decomposition, MATMUL lowering/recognition/native selection, affected tests/docs.
- Parallel group: None; no concurrent writer or measurement.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted
  dirty tree `/private/tmp/synaptik-cpu-repairs.8TWk5O/first-two-repairs-accepted.json`.
- Integration order: verified brief → source/tests → focused/module/conformance gates → targeted review/docs.
- Integration validation: CPU tests/Javadoc once excluding only the two identified unconditional
  legacy benchmark methods by a temporary external Gradle test filter; focused MATMUL conformance.
- Shared-document integration owner: MAIN.

Ready verification: MAIN checked predecessor results, actual accepted dirty baseline and protected
evidence, unchanged HEAD/origin/index, current contracts and no competing source frontier. CPU0047
is the sole active task. The previous full-run performance failure is not a correctness grant;
no benchmark retry or threshold change is allowed here.

## Files and symbols

Production paths are under `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`:

- `lowering/CpuMatmulLowering.java`: prefix privacy proof, backing/read descriptors and existing geometry.
- `lowering/CpuPartitionLowering.java`, `CpuPartitionDagDecomposer.java`: bounded seed delegation and
  projection of external-consumer obligations; never lose an original cross-partition requirement.
- `lowering/CpuSpecializedSubgraphRecognizer.java`: include proved prefix members in existing epilogue facts.
- `route/nativeblas/openblas/CpuOpenBlasRouteSelector.java`: permit the proved prefix unit while retaining
  current canonical/native and explicit-copy checks.
- Matching focused CPU tests and `testing/backend-conformance` MATMUL strategy coverage.
- Existing `EngineCpuMatmulStrategyIntegrationTest`: shared Prepare and repeated heap-backed runs,
  including publication of the transpose; no Engine implementation change.
- CPU guide/Javadoc and this brief/master/roadmap; no authority or dependency change is intended.

## Acceptance criteria

1. Left/right/both private rank-two inputs produce one MATMUL unit, original backing boundaries,
   swapped actual strides, no PERMUTE resource/copy, and correct typed/empty-domain results.
2. Public, outside-unit/partition and multiply-consumed transpose results stay materialized;
   malformed or wider geometry does not acquire unsupported lowering.
3. Bias/activation suffix alternatives and existing bare MATMUL plans remain available. Scalar,
   vector-eligible and parallel orchestration still follow the existing MATMUL selector.
4. Qualified native selection can retain its one required canonical copy, with the original backing
   and effective read mapping in its exact workload identity. No native provider contract changes.
5. Equivalent bare strided MATMUL generates identical class bytes; no hidden helper, allocation,
   additional arithmetic or runtime transpose is introduced by absorption.

## Validation

Run focused lowering/decomposition/recognizer/native-selector and generated execution tests while
developing. Final CPU tests/Javadoc use false generalScan.performance,
generalScan.pointwisePerformance and dynamicBatch.performance properties, `--max-workers=1 --console=plain`.
An external temporary init script excludes exactly
`CpuPartitionDagGeneratedEvidenceTest.fiveFixedHeapForksAndAggregateStayWithinGate` and
`CpuPartitionDagGeneratedEvidenceTest.freshFusionEvidenceRetainsEverySampleAndBothRequiredComparisons`;
all other tests in that class stay enabled. Do not change build sources or repeat those timings.
Run affected MATMUL backend-conformance tests once; no repository-wide/native rebuild.
Run `:testing:integration-tests:test --tests '*EngineCpuMatmulStrategyIntegrationTest'` for
the actual Compiler → shared Prepare → Engine consumer after module validation. Retry only an
identified failed correctness assertion after its repair; do not repeat successful module gates.

## Documentation and review impact

Planning profile for coordination; Backend-guide profile for a concrete before/after buffer example
and native-copy limitation. Update affected implementation Javadoc, review glossary terms, and
record MAIN's targeted actual-diff review under the explicit context exception.

## Result

Status: Complete

MAIN accepted on 2026-10-28 under the explicit context exception, not an independent-context review.
Five production files preserve private backing reads, external-consumer obligations, epilogue
membership and native eligibility. Four test files plus a new eight-test fixture cover left/right/both
transposes, all six numeric types on segments, FLOAT32 heap arrays, padded/offset and valid empty reads,
publication/fan-out barriers, shared backings, route copying and repeated public Engine runs.
Equivalent bare strided MATMUL has identical specialization and generated class bytes; no emitter,
schema, public API, dependency, native ABI or authority contract changed. No timing claim is made.

Validation: CPU 1,353 tests, zero failures/errors, 52 standard skips; exactly the two historical
timing methods named above were externally excluded, so this is not an unfiltered-default result.
CPU Javadoc passed with 94 existing out-of-scope warnings. MATMUL/OpenBLAS conformance: 16/16 pass;
Engine integration: 2/2 pass. Its initial new assertion incorrectly mutated the source array copied
by `fromFlatArray`; updating actual Tensor storage fixed the test without an Engine change.
Successful owning gates were reused. Logs: `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0047-*`.

Documentation/Javadoc, glossary and actual-source review are complete; the CPU guide explains the
three-buffer example, privacy barriers and native-copy limitation. Existing repeated logical
MATMUL input-ID admission is not broadened; backing-ID collisions keep a required copy. The current
OpenBLAS provider can still require one canonical copy per input. General affine/symbolic absorption
and the residual Scan performance gap remain outside this repair. No benchmark, commit or push ran.
HEAD/origin/index and all 1,176 protected predecessor evidence files were preserved. Next: CPU0048.

Changed files: the five production paths listed above; new `CpuMatmulTransposeAbsorptionTest`,
existing `CpuOpenBlasRouteSelectorTest`, `CpuMatmulStrategyConformanceTest`,
`CpuOpenBlasRouteConformanceTest` and `EngineCpuMatmulStrategyIntegrationTest`; CPU guide, glossary,
this new brief, CPU master and roadmap. No obsolete production type or API was removed.

## Standalone publication

The user selected publication of only CPU0047 on 2026-10-28. The isolated payload is based on
`6da23a0b42f015979a88f5e84416799f5ada3f40` and excludes the uncommitted general-Scan,
BatchGeometry, Compiler0016 and CPU0046 changes. Static MATMUL needs no new source dependency
from that series. Shared guide/glossary/master/roadmap files contain only the corresponding
CPU0047 edits in this payload, not their other locally accepted changes.

Isolated validation used `cpu0047-publish-validation.gradle`: CPU 49/49, conformance 16/16 and
Engine 2/2 passed without skips; CPU Javadoc passed. This is additional validation of the isolated
publication tree, not a repeat claim about the larger accepted working tree. Detailed command/log:
`/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0047-publish-validation.log`.

Independent clean-context publication review: APPROVE, zero open P1/P2 findings. It checked the
actual isolated 15-path payload and reused the successful validation; no test or benchmark rerun.
