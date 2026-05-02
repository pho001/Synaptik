# Verification: Phase 30 BF16 Metal Compute And Output

**Status:** Verified
**Verified:** 2026-05-02

## Requirement Coverage

| Requirement | Verdict | Evidence |
|---|---|---|
| `METALBF16-01` | PASS | BF16 dtype metadata reaches native compile descriptors, scoped BF16 operation families are admitted by `MetalMpsCapabilities` and `MetalPartitionSupport`, unsupported BF16 families reject with stable operation/dtype diagnostics, and native `metalTest` covers BF16 RELU, MATMUL, SUM, LayerNorm, and softmax execution. |
| `METALBF16-02` | PASS | BF16 buffer allocation/materialization handles raw storage, exact BF16 upload/readback roundtrip is tested, numeric tolerance policy is explicit for matmul/reduction and normalization/softmax, and suite reports expose dtype residency evidence. |
| `METALBF16-03` | PASS | BF16 hot-path targets for MLP, LayerNorm, RMSNorm, and reductions have hard Metal gates requiring native buffer binding, BF16 dtype evidence, zero CPU materialization, zero CPU fallback, and zero tensor-array fallback while focused FLOAT32 Metal gates remain green. |

## Verification Commands

```bash
./gradlew classes
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests StandardWorkloadsTest
./gradlew test --tests SourceTreeHygieneTest
./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.lowering.MetalRegionLowererTest --tests GpuCoverageSummaryTest
./gradlew metalTest
git diff --check
git status --short profiles/platform
```

All build and test commands passed. `git status --short profiles/platform` reports only local tuning/profile artifacts and they were not staged.

## Scope Boundaries

- BF16 support is scoped to selected Metal operation families, not universal Metal BF16.
- `FLOAT64`, `INT32` compute/output, BOOL-producing compute, conv/pool, gather/take/scatter, loss-adjacent ops, and masked/causal SDPA remain future phases or explicit rejection/fallback.
- Public `Tensor` API remains logical; device residency stays in runtime execution state.
