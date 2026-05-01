---
phase: 11-gpu-lowering-coverage-matrix
status: passed
score: 5/5
requirements_verified: [GPULOWER-01, GPULOWER-02, GPULOWER-03]
human_verification: []
gaps: []
verified: 2026-04-30
---

# Phase 11 Verification: GPU Lowering Coverage Matrix

## Verdict

Passed. Phase 11 achieved its roadmap goal: Metal and CUDA lowering coverage is now represented by a checked-in backend-neutral matrix, backend legality adapters consume that matrix while preserving backend-owned gates, `LOG_SOFTMAX` can stay in GPU regions by lowering through existing DAG primitives, and selected/rejected GPU coverage is visible in tests, traces, and docs.

## Requirement Coverage

| Requirement | Status | Evidence |
|---|---|---|
| GPULOWER-01 | Passed | `GpuLoweringCoverageMatrix` lists Metal/CUDA operation-family rows as supported, fallback, or unsupported with stable reason codes; docs mirror the matrix and drift tests pin required families/statuses/reasons. |
| GPULOWER-02 | Passed | `LOG_SOFTMAX` lowers as `SOFTMAX` followed by `LOG` without a new native ABI op code, and prepared-selection tests prove `matmul/linear -> LOG_SOFTMAX` can remain in Metal/CUDA GPU regions when cost, layout, dtype, and capability gates allow it. |
| GPULOWER-03 | Passed | Metal/CUDA legality tests, prepared-selection tests, and trace tests prove unsupported reductions, normalization, loss-adjacent paths, CUDA direct non-dense inputs, and capability gaps reject visibly without regressing CPU fallback safeguards. |

## Success Criteria

| # | Status | Evidence |
|---|---|---|
| 1. A checked-in coverage matrix lists Metal and CUDA support status for common NN/tensor patterns | Passed | Plan 11-01 added `GpuLoweringCoverageStatus`, `GpuLoweringUnsupportedReason`, `GpuLoweringOperationFamily`, `GpuLoweringCoverageEntry`, `GpuLoweringCoverageMatrix`, matrix tests, and `docs/gpu-lowering-coverage.md`. |
| 2. Lowering support expands for highest-value supported patterns while preserving stable rejection reasons | Passed | Plan 11-03 adds `LOG_SOFTMAX` decomposition through existing `SOFTMAX` and `LOG` DAG primitives; reductions, normalization, loss-adjacent paths, and fused regions remain explicit fallback/unsupported rows. |
| 3. Backend selection keeps supported patterns in GPU regions when contracts allow it | Passed | Metal/CUDA lowerer and prepared-execution tests prove selected `matmul/linear -> LOG_SOFTMAX` GPU regions and rejected unsupported loss/reduction candidates. |
| 4. Portable tests prove lowering decisions and CPU fallback safeguards for selected and rejected candidates | Passed | Focused JUnit suites cover the shared matrix, lowerer decomposition, Metal/CUDA legality, prepared selection, trace visibility, and layout-heavy flow interactions. |
| 5. Documentation explains how to add new GPU-lowerable operation families | Passed | `docs/gpu-lowering-coverage.md` documents the matrix, reason-code taxonomy, planner rejection sources, and operation-family extension workflow; `docs/development.md` lists focused verification commands. |

## Plan Completion

| Plan | Status | Summary |
|---|---|---|
| 11-01 Shared GPU Lowering Coverage Contract | Complete | Added backend-neutral coverage statuses, operation families, unsupported reasons, matrix records, source/docs drift tests, and coverage docs. |
| 11-02 Metal/CUDA Legality Coverage Alignment | Complete | Routed Metal and CUDA planner legality through the shared matrix while preserving dtype, layout, SDPA, capability, and native ABI gates. |
| 11-03 Softmax-Ish Lowering Expansion | Complete | Implemented `LOG_SOFTMAX` lowering as `SOFTMAX` followed by `LOG`, with selected/rejected Metal/CUDA candidate tests. |
| 11-04 Lowering Coverage Trace And Docs Closure | Complete | Added trace visibility, layout-heavy Metal/CUDA flow coverage, developer docs, focused verification commands, native Metal evidence, CUDA capability-gated evidence, and profile artifact hygiene. |

## Automated Checks

- `./gradlew classes` - passed.
- `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` - passed.
- `./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest` - passed.
- `./gradlew metalTest` - passed after the native `LOG_SOFTMAX_GRAD` tolerance was tightened to `5e-5` for FLOAT32 MPSGraph parity.
- `./gradlew buildCudaGraphShim cudaTest` - build successful; CUDA native tasks were skipped by local capability/toolchain gates.
- `rg -n "LOG_SOFTMAX|lowered as SOFTMAX followed by LOG|UNSUPPORTED_OPERATION" src/main/java src/test/java docs/gpu-lowering-coverage.md` - passed.

## UAT

Phase 11 UAT is complete with 4/4 developer-observable checks passed: coverage matrix, Metal/CUDA legality routing, `LOG_SOFTMAX` lowering, and trace/docs closure. No human UI workflow is required for this infrastructure phase.

## Security And Validation

- `11-SECURITY.md` records 8/8 threats closed with `threats_open: 0`.
- `11-VALIDATION.md` is Nyquist-compliant with 8/8 task verification rows covered, focused portable commands passing, native Metal evidence passing, and CUDA native checks capability-gated.

## Human Verification

None required. Phase 11 changes are accelerator/runtime infrastructure, tests, and documentation; acceptance criteria are covered by focused automated tests, trace assertions, and capability-gated native checks.

## Gaps Summary

No gaps found. Phase goal achieved. Ready to proceed to Phase 12 planning.

## Residual Risk

Native CUDA execution was not exercised on a CUDA-capable host in this run because local CUDA Gradle tasks were capability-skipped. CUDA direct non-dense compute remains intentionally conservative unless metadata-only view propagation or dense materialization makes the consumer layout legal. Reductions, normalization, loss-adjacent paths, and fused GPU compound execution remain explicit future scope for Phase 12 and later coverage work.

## Git Hygiene

Local profile tuning changes under `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/tuning/abc/*` remain unstaged and are not part of Phase 11.
