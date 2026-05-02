# Phase 40: CUDA Parity Gap Triage And Capability Baseline - Research

**Phase:** 40 CUDA Parity Gap Triage And Capability Baseline
**Status:** Complete
**Date:** 2026-05-02

## Research Complete

Phase 40 is a baseline and truth-setting phase, not a broad CUDA implementation phase. The codebase already has CUDA graph-region scaffolding, shared accelerator DAG lowering, dense `FLOAT32` CUDA buffer execution, layout ABI v2 probes, and trace/report fields. The gap is that CUDA parity evidence is not yet comparable to the v1.5 Metal evidence standard.

## Current CUDA Baseline

- CUDA bridge entry point: `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`.
- CUDA bridge SPI: `src/main/java/backend/cuda/bridge/CudaGraphBridge.java`.
- Capability state today: `CudaBridgeCapabilities` distinguishes native library, runtime, context, graph ABI, buffer execution, layout ABI v2 version/support, and one stable capability code.
- Native shim: `src/main/native/cuda/synaptik_cuda_graph_stub.cu`.
- CUDA runtime executable: `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java`.
- CUDA buffer preflight: `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java`.
- CUDA legality: `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java`.
- Shared source-of-truth coverage rows: `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java`.
- Coverage reports and gates: `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java`, `GpuCoverageGapTriage.java`, and `GpuHotPathCoverageTargets.java`.

CUDA currently has support rows for matmul/linear, elementwise, selected layout/view-adjacent nodes, softmax/log-softmax, reductions, normalization, and selected backward-adjacent rows. It has explicit unsupported or fallback rows for dense loss, SDPA, conv/pool, forward gather/take, scatter/index-gradient, bool-producing compute, and several training/backward families. Native dense CUDA execution remains `FLOAT32` oriented; the bridge accepts `FLOAT32` and `BOOL` external inputs in tensor-array transport, but output and buffer-binding paths remain `FLOAT32`.

## Current Metal Baseline Used For Parity

The v1.5 Metal baseline in `docs/gpu-lowering-coverage.md` and `docs/metal-backend.md` is much more detailed:

- Metal has dtype capability truth for role-specific BF16, BOOL, INT32, and FLOAT64 rejection.
- Metal has supported rows for dense loss, masked/causal SDPA, conv/pool forward, forward gather/take, BOOL-producing compute, selected backward-adjacent rows, and route/copy evidence.
- Metal route evidence distinguishes MPSGraph, custom-kernel unavailable, tensor-array fallback, CPU fallback, and `MPSGRAPH_RESULT_COPY`.
- Unsupported Metal rows retain stable reason codes rather than silently shortening regions.

Phase 40 should not mark CUDA supported by analogy. A CUDA row should become supported later only after semantic contract, lowering, legality, native/routed execution, parity tests, trace/report fields, and regression gates exist.

## Planning Implications

1. Add a backend-neutral CUDA parity report rather than hiding information only in docs tables.
2. Preserve the existing `GpuLoweringCoverageMatrix` as the operation-family source of truth, but add a derived parity view comparing Metal rows to CUDA rows.
3. Split capability evidence into dimensions:
   - native library
   - CUDA runtime/device
   - context
   - graph execution ABI
   - buffer binding ABI
   - layout ABI v2
   - dtype roles
   - DAG primitive availability
   - vendor-library route availability, explicitly `NOT_INTEGRATED` for cuBLAS/cuDNN until real routes exist
   - hardware/toolchain availability
4. Extend hot-path target metadata so CUDA exits are classified as blocker, accepted capability gap, future scope, or requires native evidence.
5. Keep local CUDA native runs capability-gated because many dev machines lack `nvcc` or CUDA hardware.

## Recommended File Areas

- `src/main/java/backend/accelerator/lowering/` for backend parity report records derived from `GpuLoweringCoverageMatrix`.
- `src/main/java/backend/cuda/bridge/` for capability dimension records and bridge capability projection.
- `src/main/java/tuning/benchmark/report/` for CUDA hot-path blocker policy and report-visible classifications.
- `src/test/java/backend/accelerator/lowering/`, `src/test/java/backend/cuda/bridge/`, and existing top-level coverage tests for focused portable coverage.
- `docs/gpu-lowering-coverage.md`, `docs/development.md`, and a new `docs/cuda-backend.md` for baseline documentation.

## Validation Architecture

Phase 40 should validate the planning baseline with portable Java tests first. Optional native CUDA verification remains a capability-gated check and must not be required for default success.

Required quick gates:

```bash
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.cuda.bridge.CudaFfmBridgeTest
./gradlew test --tests GpuCoverageTriageReportTest --tests GpuHotPathCoverageTargetsTest
./gradlew classes
git diff --check
```

Optional native gate when CUDA toolkit/hardware is available:

```bash
./gradlew buildCudaGraphShim cudaTest
```

Validation must prove:

- capability skip is never counted as support;
- CUDA parity rows distinguish `SUPPORTED`, `FALLBACK`, `UNSUPPORTED`, and `CAPABILITY_MISSING`;
- dense `FLOAT32` CUDA rows are not confused with BF16/BOOL/INT32 native compute/output support;
- vendor-library routes are explicit `NOT_INTEGRATED` evidence, not implied by CUDA presence;
- hot-path blocker classification identifies v1.6 CUDA implementation targets.

## Open Risks

- Local hardware may not provide CUDA native execution; plans must keep portable tests mandatory and native CUDA tests optional/capability-gated.
- Existing tests may encode old assumptions such as `CudaFfmBridge.supportsBufferBindings()` always being false; Phase 40 should make such assertions capability-sensitive.
- Docs can easily overstate parity; every support claim must be backed by code, tests, and trace/report evidence.

---

*Research completed inline because GSD subagents are not installed in this runtime.*
