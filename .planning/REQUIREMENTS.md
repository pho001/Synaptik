# Requirements: Synaptik

**Defined:** 2026-04-30
**Milestone:** v1.1 CUDA Native Runtime
**Core Value:** Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## v1.1 Requirements

These requirements scope the next brownfield milestone after v1.0. They focus on turning the existing CUDA scaffolding into a checked-in, capability-gated native runtime path without expanding into broad GPU fusion or public device tensors.

### CUDA Native Runtime

- [ ] **CUDA-01**: CUDA has a checked-in native shim, documented build workflow, and runtime capability probe that fail gracefully when CUDA tooling or hardware is unavailable.
- [ ] **CUDA-02**: CUDA bridge and prepared executable seams consume the shared accelerator buffer layout/access ABI for supported dense layouts without adding CUDA-specific assumptions to common runtime contracts.
- [ ] **CUDA-03**: CUDA buffer execution can allocate native device buffers and run at least one representative supported accelerator operation through a capability-gated native path.
- [ ] **CUDA-04**: CUDA native device-to-CPU materialization is correct for graph output and CPU consumer boundaries, with CPU parity tests for supported dtype/layout combinations.
- [ ] **CUDA-05**: Adjacent CUDA accelerator regions can pass device-owned buffers without Java array round trips when layout and capability contracts allow it.
- [ ] **CUDA-06**: CUDA fallback and required-mode failures remain explicit through stable reason codes for unavailable native runtime, unsupported dtype, unsupported layout, and required-but-unavailable buffer execution.

### CUDA Observability And Documentation

- [ ] **CUDADOC-01**: CUDA traces and benchmark reports expose the same accelerator evidence contract as Metal: backend, buffer execution path, reason code, fallback reason, prepared input usage, CPU materialization count/reason, copy timing, and storage residency.
- [ ] **CUDADOC-02**: Developer documentation explains CUDA build prerequisites, capability probing, native shim troubleshooting, fallback interpretation, and how CUDA differs from Metal while sharing the same accelerator ABI.
- [ ] **CUDADOC-03**: Source hygiene and verification gates prevent accidental commits of local CUDA build outputs, machine-local benchmark/profile artifacts, and generated native scratch files.

## Future Requirements

### Broader Accelerator Coverage

- **ACCEL-01**: Accelerator lowering supports a broader set of neural-network operations beyond current Metal/CUDA coverage.
- **ACCEL-02**: Accelerator regions can lower larger fused GPU kernels where backend APIs make that profitable.
- **ACCEL-03**: Higher-rank native shape ABI support is available where backend runtimes support it.

### Storage And Memory Scaling

- **MEM-01**: Runtime memory binding supports BFLOAT16, INT32, and BOOL slot reuse, not only FLOAT64 and FLOAT32.
- **MEM-02**: Tensor shape and element-count helpers use checked long arithmetic where large tensors can overflow int products.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Public user-facing device tensor API | The project direction keeps public tensors logical and handles backend residency in compile/prepare/execute runtime state. |
| Broad CUDA operation coverage | This milestone proves the native runtime path and buffer/materialization contracts first; broader lowering can follow once the runtime is trustworthy. |
| Large fused CUDA kernels | Fusion strategy depends on broader accelerator coverage and should not be mixed with first native runtime bring-up. |
| CUDA profile calibration churn | Local performance profiles remain machine-specific; this milestone should test capability and observability without committing local benchmark/calibration output. |
| Removing CPU fallback | CPU remains the correctness oracle and required fallback for unsupported operations, dtypes, layouts, or missing CUDA runtime. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CUDA-01 | Phase 6 | Pending |
| CUDA-02 | Phase 6 | Pending |
| CUDA-03 | Phase 7 | Pending |
| CUDA-04 | Phase 7 | Pending |
| CUDA-05 | Phase 7 | Pending |
| CUDA-06 | Phase 8 | Pending |
| CUDADOC-01 | Phase 8 | Pending |
| CUDADOC-02 | Phase 8 | Pending |
| CUDADOC-03 | Phase 8 | Pending |

**Coverage:**
- v1.1 requirements: 9 total
- Mapped to phases: 9
- Unmapped: 0

---
*Requirements defined: 2026-04-30*
*Last updated: 2026-04-30 after v1.1 definition*
