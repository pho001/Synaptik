# Phase 28 Research

## Existing Coverage Infrastructure

`GpuCoverageSummary` derives backend coverage from prepare and run traces. It already records accelerator-step ratio, selected region count and length, multi-op region count, lowered primitive count, rejected candidate reasons, buffer-binding steps, tensor-array steps, CPU fallback steps, CPU materialization count/reasons/bytes/duration, copy duration, device handoffs, layout materialization counts, dtype residency reasons, fused subpattern counts, reason codes, and fallback reasons.

`GpuCoverageRegressionGate` already evaluates most regression dimensions needed by Phase 28:

- minimum GPU coverage ratio,
- minimum selected region length,
- minimum multi-op GPU region count,
- minimum lowered primitive count,
- minimum fused subpattern count,
- maximum CPU materializations,
- maximum fallback count,
- maximum tensor-array bridge steps,
- maximum device handoffs,
- required native buffer binding.

`GpuHotPathCoverageTargets` already maps representative v1.4 workloads to expectations. The current gap is policy precision: some workloads still use permissive partial-blocker policies even after v1.4 phases turned their operation family into supported native/lowered execution.

`GpuTargetCoverageTruth` classifies operation families as:

- `NATIVE_EXECUTABLE`
- `MATRIX_SUPPORTED_ONLY`
- `EXPLICIT_CPU_FALLBACK`
- `UNSUPPORTED_REJECTION`

This is the right source for distinguishing hard native gates from expected visible fallback/rejection gates.

## Current Policy Gaps

- `reduction_chain_small` still uses `partialBlockerPolicy` even though Phase 23 made reductions native-executable for Metal/CUDA legal dense FLOAT32 cases.
- `layer_norm_small` and `rms_norm_small` already use hard supported policies and should remain strict.
- Metal `transformer_block_hot_path` already requires native evidence for supported SDPA; CUDA should continue to require visible `CAPABILITY_MISSING` evidence instead of pretending support.
- `mlp_classifier_small` already has a hard hot-path policy, but Phase 28 should verify it is represented in final evidence and not weakened by target report rendering.
- `conv2d_resnet_3x3`, `max_pool2d_small`, `cross_entropy_small`, and `bool_compare_where_small` are valid partial/visible-blocker targets until their operation families become native supported. Their policies should still require stable visible reasons and should not let hidden tensor-array or CPU replay masquerade as native coverage for adjacent supported regions.

## Report And Benchmark Evidence

Report renderers already include:

- `targetCoverageGates` in suite reports,
- `nativeEvidence` and `capabilitySkipped`,
- `targetCoverageTruth` in per-workload reports,
- backend coverage fields for buffer binding, tensor-array path, fallback, materialization, and device handoffs.

`GpuCoverageBaseline` and `GpuCoverageComparison` already compare current coverage against a stable baseline without timing thresholds. Phase 28 can use these concepts for deterministic before/after coverage evidence instead of treating local benchmark timing medians as the milestone proof.

## Implementation Implications

Phase 28 should prioritize three code paths:

1. Tighten target policies so native-executable v1.4 families fail hidden fallback.
2. Add deterministic tests proving every supported target has a hard gate and every unsupported target has visible reason evidence.
3. Add final report/docs evidence that maps Phase 28 requirements to checked-in code/tests/docs while keeping local profile artifacts out of evidence.

## Verification Strategy

Focused gates should prefer deterministic Java tests over timing-heavy benchmark/debug suites:

```bash
./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests StandardWorkloadsTest
```

When report renderers or docs change:

```bash
./gradlew test --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests GpuCoverageSummaryTest
git diff --check
```

Native optional gates remain useful but environment-dependent:

```bash
./gradlew metalTest
./gradlew cudaTest
```

`./gradlew cudaTest` is expected to be skipped or unavailable locally when `nvcc` is not installed.
