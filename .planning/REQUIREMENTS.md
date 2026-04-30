# Requirements: Synaptik

**Defined:** 2026-04-30
**Milestone:** v1.2 GPU Region Coverage
**Core Value:** Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## v1.2 Requirements

These requirements scope the next brownfield milestone after v1.1. They focus on keeping larger portions of realistic tensor graphs resident on Metal or CUDA by expanding layout support, operation lowering, fused GPU region execution, and coverage reporting without hiding fallback or weakening CPU correctness.

### Native Layout ABI

- [x] **GPULAYOUT-01**: Metal and CUDA native bridge contracts can describe rank, shape, strides, storage offset, logical element count, physical byte span, access mode, backend id, and native handle identity for non-contiguous/view-capable accelerator buffers.
- [x] **GPULAYOUT-02**: Metal and CUDA expose capability/version checks for layout ABI v2 so Java can select native non-contiguous support only when the backend advertises matching symbols and semantics.
- [x] **GPULAYOUT-03**: Unsupported layout metadata, rank, dtype, aliasing, or native ABI mismatch produces stable fallback/required-mode reason codes instead of silent CPU materialization.

### GPU Layout Transform And View Path

- [ ] **GPUVIEW-01**: `reshape`, `permute`, `expand`, `contiguous`, alias outputs, and legal view-like values can remain device-resident across compatible Metal and CUDA regions when layout and capability contracts allow it.
- [ ] **GPUVIEW-02**: GPU-side layout transform or logical-view materialization produces CPU-parity graph outputs and CPU-consumer values while avoiding intermediate CPU materialization between accelerator-compatible nodes.
- [ ] **GPUVIEW-03**: Metal and CUDA share backend-neutral layout transform/request/decision records while native handles, layout kernels, and capability details remain backend-owned.

### GPU Lowering Coverage

- [ ] **GPULOWER-01**: A checked-in Metal/CUDA operation coverage matrix classifies common NN/tensor patterns as supported, fallback, or unsupported with stable reason codes.
- [ ] **GPULOWER-02**: The lowering pipeline can keep supported matmul/linear, elementwise-chain, reduction, softmax-like, normalization, and loss-adjacent patterns in GPU regions when layout, dtype, and backend capability contracts are satisfied.
- [ ] **GPULOWER-03**: Backend selection and lowering tests prove unsupported operations, dtypes, layouts, or backend capability gaps reject cleanly without regressing CPU, Metal, or CUDA fallback safeguards.

### Fused GPU Region Execution

- [ ] **GPUFUSE-01**: Metal and CUDA can execute at least one linear + bias + activation fused GPU region without Java array round trips between the fused operations.
- [ ] **GPUFUSE-02**: Metal and CUDA can execute representative elementwise-chain fused GPU regions and preserve device-owned intermediate values through region execution.
- [ ] **GPUFUSE-03**: Fused GPU region planning is implemented as backend-specific compound DAG execution and does not depend on or regress the CPU fused ASM/vector execution path.
- [ ] **GPUFUSE-04**: Reduction-adjacent fusion candidates are either implemented with CPU parity tests or explicitly rejected with stable unsupported reason codes and coverage matrix entries.

### Coverage Benchmark And Regression Gates

- [ ] **GPUCOV-01**: Trace and benchmark reports expose GPU coverage ratio, selected region length, rejected candidate reasons, fallback counts, CPU materialization count/reason, copy timing, storage residency, and device handoff counts for Metal and CUDA.
- [ ] **GPUCOV-02**: Representative workloads such as transformer block, MLP, and a convolution- or normalization-heavy graph demonstrate fewer GPU-to-CPU exits or longer GPU-covered regions than the v1.1 baseline.
- [ ] **GPUCOV-03**: Regression gates fail when supported target workloads lose GPU coverage, add unexpected CPU materialization boundaries, or hide fallback behind tensor-array execution.

## Future Requirements

### Broader Accelerator Coverage

- **ACCEL-01**: Accelerator lowering supports most remaining neural-network operations beyond the v1.2 target matrix.
- **ACCEL-02**: GPU fused regions cover larger multi-stage kernels where backend APIs make that profitable.
- **ACCEL-03**: Higher-rank native shape/layout ABI support is available beyond the ranks required for v1.2 target workloads.

### Storage And Memory Scaling

- **MEM-01**: Runtime memory binding supports BFLOAT16, INT32, and BOOL slot reuse, not only FLOAT64 and FLOAT32.
- **MEM-02**: Tensor shape and element-count helpers use checked long arithmetic where large tensors can overflow int products.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Public user-facing device tensor API | The project direction keeps public tensors logical and handles backend residency in compile/prepare/execute runtime state. |
| Universal GPU support for every operation, dtype, rank, and layout | v1.2 should broaden high-value graph regions first and keep unsupported cases explicit. |
| Removing CPU fallback | CPU remains the correctness oracle and required fallback for unsupported operations, dtypes, layouts, missing native runtimes, or failed capability checks. |
| Replacing CPU fused ASM/vector execution | GPU fusion should be backend-native compound region execution; CPU fused paths remain independent and performance-critical. |
| Committing local benchmark/profile output | Benchmark evidence should verify report contracts and selected workload behavior without committing machine-local calibration artifacts. |
| OpenCL runtime expansion | This milestone focuses on Metal and CUDA parity; OpenCL remains separate placeholder/runtime scope. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| GPULAYOUT-01 | Phase 9 | Complete |
| GPULAYOUT-02 | Phase 9 | Complete |
| GPULAYOUT-03 | Phase 9 | Complete |
| GPUVIEW-01 | Phase 10 | Pending |
| GPUVIEW-02 | Phase 10 | Pending |
| GPUVIEW-03 | Phase 10 | Pending |
| GPULOWER-01 | Phase 11 | Pending |
| GPULOWER-02 | Phase 11 | Pending |
| GPULOWER-03 | Phase 11 | Pending |
| GPUFUSE-01 | Phase 12 | Pending |
| GPUFUSE-02 | Phase 12 | Pending |
| GPUFUSE-03 | Phase 12 | Pending |
| GPUFUSE-04 | Phase 12 | Pending |
| GPUCOV-01 | Phase 13 | Pending |
| GPUCOV-02 | Phase 13 | Pending |
| GPUCOV-03 | Phase 13 | Pending |

**Coverage:**
- v1.2 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

---
*Requirements defined: 2026-04-30*
*Last updated: 2026-04-30 after Phase 9 verification*
