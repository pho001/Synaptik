---
phase: 15
slug: gpu-region-internal-lowered-dag-contract
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 15 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter 5.11.2 via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests CompiledGraphTraceTest` |
| **Full suite command** | `./gradlew classes && ./gradlew test --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests CompiledGraphTraceTest --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest` |
| **Estimated runtime** | ~120 seconds focused; native Metal/CUDA execution remains capability-gated |

## Sampling Rate

- **After every task commit:** Run the focused test class for the touched area.
- **After every plan wave:** Run the quick run command.
- **Before `$gsd-verify-work`:** Run the full suite command and `git status --short`.
- **Max feedback latency:** 120 seconds for focused tests.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 15-01-01 | 01 | 1 | GPUDAG-01, GPUDAG-02 | T-15-01 | Manifest model cannot mutate native DAG ABI or public tensor semantics. | unit | `./gradlew test --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest` | W0 | pending |
| 15-01-02 | 01 | 1 | GPUDAG-03 | T-15-02 | Stable reason codes exist for primitive, boundary, shortening, and fused-subpattern attribution. | unit | `./gradlew test --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest` | W0 | pending |
| 15-02-01 | 02 | 2 | GPUDAG-01, GPUDAG-02 | T-15-03 | Lowerer creates bidirectional original-op to lowered-primitive mapping for multi-primitive expansions. | unit | `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest` | W0 | pending |
| 15-02-02 | 02 | 2 | GPUDAG-02 | T-15-04 | Metal and CUDA selected plans expose the shared manifest without native ABI changes. | unit | `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | W0 | pending |
| 15-03-01 | 03 | 3 | GPUDAG-02, GPUDAG-03 | T-15-05 | Prepare trace exposes structured manifests and stable text/JSON fields for selected GPU regions. | trace/report | `./gradlew test --tests CompiledGraphTraceTest --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest` | W0 | pending |
| 15-04-01 | 04 | 4 | GPUDAG-01, GPUDAG-02, GPUDAG-03 | T-15-06 | Docs, source hygiene, CPU guardrails, and local artifact hygiene are verified before phase closure. | docs/test | `./gradlew classes && ./gradlew test --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests CompiledGraphTraceTest --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest` | W0 | pending |

## Wave 0 Requirements

Existing infrastructure covers all phase requirements:

- `src/main/java/backend/accelerator/dag/AcceleratorDagSpec.java`
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java`
- `src/main/java/backend/accelerator/lowering/GpuCompoundRegionSummary.java`
- `src/main/java/graph/execution/trace/BackendSelectionDecisionTrace.java`
- `src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java`
- `src/test/java/CompiledGraphTraceTest.java`

## Manual-Only Verifications

All Phase 15 behaviors have automated verification. Native CUDA execution remains optional and capability-gated outside this phase.

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all existing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 120s.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending
