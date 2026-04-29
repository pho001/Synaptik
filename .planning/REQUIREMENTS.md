# Requirements: Synaptik

**Defined:** 2026-04-29
**Core Value:** Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## v1 Requirements

These requirements define the next brownfield milestone. They are intentionally scoped to the accelerator/runtime architecture work that follows the existing codebase state.

### Device Buffer ABI

- [x] **ABI-01**: Runtime device buffer bindings can represent backend id, native handle identity, dtype, shape, strides, storage offset, logical element count, byte length, and access mode. Completed in Phase 1 and verified by `001-VERIFICATION.md`.
- [x] **ABI-02**: The shared accelerator buffer model is backend-neutral and reusable by Metal now and CUDA later. Completed in Phase 1 and verified by `001-VERIFICATION.md`.
- [x] **ABI-03**: Buffer compatibility checks distinguish dense contiguous tensors, zero-offset views, non-zero-offset views, permuted/strided views, broadcast/zero-stride views, and unsupported layouts. Completed in Phase 1 and verified by `001-VERIFICATION.md`.
- [x] **ABI-04**: Buffer binding decisions expose stable reason codes for success, fallback, unsupported dtype, unsupported layout, unavailable native ABI, and required-but-unavailable buffer execution. Completed in Phase 1 and verified by `001-VERIFICATION.md`.

### Metal Layout Execution

- [ ] **METAL-01**: Metal buffer execution can keep legal view-like values device-resident instead of forcing CPU materialization solely because a tensor is non-contiguous or has a storage offset.
- [ ] **METAL-02**: Metal handles `LINEAR -> RESHAPE -> PERMUTE` style accelerator regions without accidental CPU fallback when a safe device layout or device contiguous transform is available.
- [ ] **METAL-03**: Metal capability checks remain conservative for unsupported dtype, rank, layout, and native ABI combinations.
- [ ] **METAL-04**: Metal device-to-CPU materialization remains correct for graph outputs, CPU consumers, and gradient publication boundaries.

### Accelerator Region Planning

- [ ] **PLAN-01**: Accelerator region planning scores CPU materialization cost, device upload/download cost, tensor-array copy fallback, layout fallback, dispatch overhead, and estimated compute work.
- [ ] **PLAN-02**: The planner prefers longer profitable device-owned regions over short accelerator islands when that reduces CPU/GPU boundaries.
- [ ] **PLAN-03**: The planner rejects or splits accelerator regions when expected materialization or layout costs erase the compute benefit.
- [ ] **PLAN-04**: CPU natural regions and CPU fused execution remain available and competitive when GPU offload is not profitable.

### Tuning And Profiles

- [ ] **TUNE-01**: Graph autotune owns graph/workload-specific decisions such as offload policy, accelerator region strategy, CPU region policy, fusion policy, and layout/materialization policy.
- [ ] **TUNE-02**: Platform calibration owns hardware/dtype-specific thresholds such as BLAS, CPU vector/parallel, fused ASM, reduction, scheduler, and accelerator runtime thresholds.
- [ ] **TUNE-03**: Profile persistence validates schema/version fields and does not silently default unsupported accelerator buffer or layout settings.
- [ ] **TUNE-04**: Benchmark commands remain read-only; autotune/calibration are the only flows that intentionally persist profile results.

### Verification And Observability

- [ ] **OBS-01**: Execution traces report accelerator backend, buffer execution path, fallback reason, prepared input usage, CPU materialization count, materialization reason, Java/native copy time, native device copy time, and storage residency.
- [ ] **OBS-02**: Benchmarks include at least one workload that stresses matmul/linear, view/layout transforms, elementwise fusion, reductions, and backward/gradient publication.
- [ ] **OBS-03**: Tests prove adjacent accelerator regions can pass device buffers without Java array round trips when layout and capability contracts allow it.
- [ ] **OBS-04**: Regression tests compare CPU and Metal results for representative forward and forward-backward graphs.

### Documentation And Maintenance

- [ ] **DOC-01**: Documentation explains the backend-neutral accelerator buffer ABI, layout/view handling, CPU materialization boundaries, and Metal/CUDA implementation responsibilities.
- [ ] **DOC-02**: Documentation distinguishes platform calibration from graph autotune and lists which knobs belong to each.
- [ ] **DOC-03**: Developer docs describe how to diagnose accelerator fallback and how to interpret trace/benchmark reports.
- [ ] **DOC-04**: Source hygiene rules prevent accidental commits of local generated classes, debug artifacts, and unintentional calibration/benchmark outputs.

## v2 Requirements

### CUDA Native Runtime

- **CUDA-01**: CUDA has a checked-in native shim, build workflow, capability probe, and buffer binding implementation equivalent in concept to Metal.
- **CUDA-02**: CUDA supports native device-to-CPU materialization and adjacent region buffer handoff tests.
- **CUDA-03**: CUDA benchmark reports expose the same accelerator trace fields as Metal.

### Broader Accelerator Coverage

- **ACCEL-01**: Accelerator lowering supports a broader set of neural-network operations beyond current Metal coverage.
- **ACCEL-02**: Accelerator regions can lower larger fused GPU kernels where backend APIs make that profitable.
- **ACCEL-03**: Higher-rank native shape ABI support is available where backend runtimes support it.

### Storage And Memory Scaling

- **MEM-01**: Runtime memory binding supports BFLOAT16, INT32, and BOOL slot reuse, not only FLOAT64 and FLOAT32.
- **MEM-02**: Tensor shape and element-count helpers use checked long arithmetic where large tensors can overflow int products.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Public user-facing device tensor API | The near-term design keeps tensors logical and handles backend residency in runtime state. |
| Full CUDA native implementation in the first phase | The shared ABI must be CUDA-ready, but Metal is the available native backend today. |
| Removing CPU fallback entirely | CPU fallback remains necessary for unsupported ops, unsupported dtype/layout combinations, and correctness comparisons. |
| Universal Metal dtype/rank support | Capability expansion must be deliberate and tested; unsupported cases should be rejected cleanly. |
| Benchmark-driven local profile churn in every commit | Profiles should be committed only when intentionally updating canonical winners or fixtures. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ABI-01 | Phase 1 | Complete — verified in 001-VERIFICATION |
| ABI-02 | Phase 1 | Complete — verified in 001-VERIFICATION |
| ABI-03 | Phase 1 | Complete — verified in 001-VERIFICATION |
| ABI-04 | Phase 1 | Complete — verified in 001-VERIFICATION |
| METAL-01 | Phase 2 | Pending |
| METAL-02 | Phase 2 | Pending |
| METAL-03 | Phase 2 | Pending |
| METAL-04 | Phase 2 | Pending |
| PLAN-01 | Phase 3 | Pending |
| PLAN-02 | Phase 3 | Pending |
| PLAN-03 | Phase 3 | Pending |
| PLAN-04 | Phase 3 | Pending |
| TUNE-01 | Phase 4 | Pending |
| TUNE-02 | Phase 4 | Pending |
| TUNE-03 | Phase 4 | Pending |
| TUNE-04 | Phase 4 | Pending |
| OBS-01 | Phase 5 | Pending |
| OBS-02 | Phase 5 | Pending |
| OBS-03 | Phase 5 | Pending |
| OBS-04 | Phase 5 | Pending |
| DOC-01 | Phase 5 | Pending |
| DOC-02 | Phase 5 | Pending |
| DOC-03 | Phase 5 | Pending |
| DOC-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0

---
*Requirements defined: 2026-04-29*
*Last updated: 2026-04-29 after Phase 1 verification*
