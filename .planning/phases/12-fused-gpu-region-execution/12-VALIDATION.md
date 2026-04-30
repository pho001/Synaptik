---
phase: 12
slug: fused-gpu-region-execution
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
updated: 2026-04-30
---

# Phase 12 - Validation Strategy

Per-phase validation contract and Nyquist audit for fused GPU region execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | JUnit 5 via Gradle |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests 'backend.accelerator.lowering.*' --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` |
| Full suite command | `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` |
| Native Metal gate | `./gradlew metalTest` |
| Native CUDA gate | `./gradlew buildCudaGraphShim cudaTest` |
| Estimated runtime | ~1-180 seconds focused depending on Gradle cache and native capability gates |

---

## Sampling Rate

- After every task commit: run the task's focused `./gradlew test --tests ...` command from the plan verify block.
- After every plan wave: run the quick run command.
- Before `$gsd-verify-work`: run the full suite command plus available native gates.
- Max feedback latency: one focused Gradle invocation per task; optional native gates may be skipped by local capability probes.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | Evidence | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|----------|--------|
| 12-01-01 | 01 | 1 | GPUFUSE-03, GPUFUSE-04 | T-12-01 / T-12-02 | CPU `Operation.OpType.FUSED` rejects explicitly on GPU paths; compound summaries do not bypass DAG legality. | unit | `./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest` | `GpuCompoundPatternDetectorTest`, `GpuLoweringCoverageMatrixTest` | green |
| 12-01-02 | 01 | 1 | GPUFUSE-03 | T-12-03 | Backend-neutral compound summaries and matrix rows expose stable Metal/CUDA support and rejection reasons. | unit | `./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest` | `GpuCompoundPatternDetectorTest`, `GpuLoweringCoverageMatrixTest`, `docs/gpu-lowering-coverage.md` | green |
| 12-02-01 | 02 | 2 | GPUFUSE-01, GPUFUSE-03 | T-12-04 / T-12-06A | Full `linear + bias + activation` regions lower as one supported compound region for Metal and CUDA metadata paths. | unit/integration | `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest` | `AcceleratorSubgraphLowererTest`, `MetalRegionLowererTest`, `CudaRegionLowererTest`, `PreparedExecutionBuildTest` | green |
| 12-02-02 | 02 | 2 | GPUFUSE-01 | T-12-05 | Required accelerator buffer mode fails before hidden CPU fallback when native buffer execution is unavailable. | integration | `./gradlew test --tests PreparedExecutionBuildTest` | `PreparedExecutionBuildTest` | green |
| 12-03-01 | 03 | 3 | GPUFUSE-02, GPUFUSE-03 | T-12-06B / T-12-07 / T-12-08 | Representative `ADD -> RELU -> EXP` chains publish one compound GPU step and keep supported interiors device-owned without `CPU_CONSUMER` materialization. | integration | `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | `PreparedExecutionBuildTest`, `CompiledGraphTraceTest`, Metal/CUDA buffer binding tests | green |
| 12-04-01 | 04 | 4 | GPUFUSE-04 | T-12-09 | Reduction-adjacent candidates reject with visible, stable `REDUCTION_ADJACENT` / `DEFERRED_FUSED_REGION` diagnostics. | unit/integration | `./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests CompiledGraphTraceTest` | `GpuCompoundPatternDetectorTest`, `MetalRegionLowererTest`, `CudaRegionLowererTest`, `CompiledGraphTraceTest` | green |
| 12-04-02 | 04 | 4 | GPUFUSE-03 | T-12-10 / T-12-11 / T-12-12 | CPU fused paths remain independent, docs avoid CUDA/public residency overclaims, and local profile artifacts stay out of the index. | regression/docs/hygiene | `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest`; `rg -n "import backend.cpu.fused" src/main/java/backend/accelerator src/main/java/backend/metal src/main/java/backend/cuda`; `git diff --cached --name-only` | `PreparedExecutionBuildTest`, `CompiledGraphTraceTest`, docs, source hygiene check | green |

Status: green = targeted command passed or was already covered by the consolidated successful validation gate.

---

## Requirement Coverage

| Requirement | Coverage | Status |
|-------------|----------|--------|
| GPUFUSE-01 | Linear+bias+activation lowerer tests, prepared execution tests, required-buffer-mode failure test, `metalTest`, CUDA capability-gated `cudaTest`. | covered |
| GPUFUSE-02 | Elementwise-chain prepared execution, trace metadata, Metal/CUDA synthetic buffer-binding residency tests, `metalTest`, CUDA capability-gated `cudaTest`. | covered |
| GPUFUSE-03 | Backend-neutral compound summaries, Metal/CUDA lowerer metadata, CPU `FUSED` rejection, no `backend.cpu.fused` imports in accelerator/Metal/CUDA production packages, docs coverage. | covered |
| GPUFUSE-04 | Reduction-adjacent detector, Metal/CUDA rejection diagnostics, coverage matrix entries, trace rejection metadata. | covered |

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements:

- `build.gradle` provides Gradle/JUnit execution.
- Accelerator lowering tests exist under `src/test/java/backend/accelerator/lowering`.
- Backend lowerer tests exist under `src/test/java/backend/metal/lowering` and `src/test/java/backend/cuda/lowering`.
- Prepared execution, trace, and buffer residency tests exist in `PreparedExecutionBuildTest`, `CompiledGraphTraceTest`, `PreparedMetalExecutableBufferBindingTest`, and `PreparedCudaExecutableBufferPolicyTest`.

No generated test files were needed during this validation pass.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Native CUDA execution on a CUDA host | GPUFUSE-01, GPUFUSE-02 | Requires local CUDA runtime/toolchain and hardware. The Gradle gate is automated but capability-skips when CUDA is unavailable. | Run `./gradlew buildCudaGraphShim cudaTest` on a CUDA-capable host and require task success without skips before claiming local native CUDA execution. |

Native Metal execution is not manual-only on this host; `./gradlew metalTest` completed successfully.

---

## Validation Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Requirements audited | 4 |
| Task verification rows | 7 |
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Generated test files | 0 |

Verification commands:

- `./gradlew test --tests 'backend.accelerator.lowering.*' --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - PASS
- `./gradlew metalTest` - PASS
- `./gradlew buildCudaGraphShim cudaTest` - PASS with `buildCudaGraphShim SKIPPED` and `cudaTest SKIPPED` on this host
- `rg -n "import backend.cpu.fused" src/main/java/backend/accelerator src/main/java/backend/metal src/main/java/backend/cuda` - PASS, no matches during Phase 12 security audit

The first sandboxed CUDA gate attempt failed before Gradle execution because the wrapper could not access the `~/.gradle` lock file. The escalated Gradle run completed successfully.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands or capability-gated native commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency bounded by focused Gradle filters.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-04-30
