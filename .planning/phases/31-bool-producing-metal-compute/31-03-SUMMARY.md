# 31-03 Summary: Mask Chain Residency And WHERE Consumer Gates

**Completed:** 2026-05-02
**Requirements:** METALBOOL-02

## Delivered

- Allowed Metal internal BOOL values in the accelerator dtype residency policy so Metal-produced masks can remain region-internal values.
- Hardened `compare -> WHERE -> elementwise` prepared execution evidence to prove BOOL compare output is not exposed as a BOOL external input and has compute/internalValue residency evidence.
- Promoted `bool_compare_where_small` to a Metal hard-native hot-path target with native buffer, multi-op region, lowered primitive, and zero fallback/materialization requirements.
- Kept CUDA `bool_compare_where_small` capability-gated with visible BOOL/compare blocker expectations.
- Added coverage regression gates that fail `bool_compare_where_small` when supported BOOL dtype residency evidence is missing or only unsupported/rejection evidence is present.

## Verification

```bash
./gradlew test --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests PreparedExecutionBuildTest --tests GpuCoverageSummaryTest --tests BenchmarkSuiteSessionTest
./gradlew metalTest
git diff --check
```

## Notes

- Masked SDPA admission remains out of scope for Phase 31; Phase 34 will consume this BOOL residency foundation.
- Local platform tuning profile artifacts under `profiles/platform/...` were intentionally left unstaged.
