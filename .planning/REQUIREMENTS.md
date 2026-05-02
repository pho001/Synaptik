# Requirements: Synaptik v1.6 Accelerator Backend Parity And Native Kernel Closure

**Defined:** 2026-05-02
**Core Value:** Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## v1.6 Requirements

Requirements for the v1.6 milestone. Each requirement maps to exactly one roadmap phase.

### CUDA Parity Gap Triage And Capability Baseline

- [x] **CUDAPARITY-01**: CUDA coverage reports compare every v1.5 Metal-supported family against CUDA status, reason code, native capability, and required evidence before any row is promoted.
- [x] **CUDAPARITY-02**: CUDA native bridge and planner capability probes distinguish dtype, layout, DAG primitive, vendor-library, buffer-binding, and hardware/toolchain availability without treating capability skips as support.
- [x] **CUDAPARITY-03**: CUDA hot-path targets define which CPU exits are blockers, which are accepted capability gaps, and which require real CUDA native execution evidence.

### CUDA DType Layout And Index Residency

- [ ] **CUDADTYPE-01**: CUDA buffer binding and runtime residency can represent BF16, BOOL, and INT32 roles needed by v1.6 without claiming unsupported compute/output dtypes.
- [ ] **CUDADTYPE-02**: CUDA layout routing handles metadata-only views and legal dense materialization or rejects with stable layout reason codes matching the shared accelerator contract.
- [ ] **CUDAINDEX-01**: CUDA forward `GATHER` and `TAKE_ALONG_AXIS` either execute for the scoped INT32 index contract or reject with explicit bounds/layout/dtype evidence while preserving adjacent GPU producers.

### CUDA NN Operation Parity

- [ ] **CUDANN-01**: CUDA masked/causal SDPA forward has native/lowered execution or stable capability-gated rejection with mask polarity, scale, rank, dtype, and layout evidence.
- [ ] **CUDANN-02**: CUDA conv/pool forward coverage for scoped `CONV2D`, `CONV2D_GEMM`, `MAX_POOL2D`, and `AVG_POOL2D` either executes natively/lowered or reports precise capability/layout/reduction-divisor blockers.
- [ ] **CUDANN-03**: CUDA dense `NLL_LOSS` and dense `CROSS_ENTROPY_LOSS` lower or reject under the same dense loss contract as Metal, with CPU parity and report-visible boundaries.

### CUDA Training And Index Semantics

- [ ] **CUDATRAIN-01**: CUDA training/backward coverage distinguishes supported backward rows, capability-missing rows, and gradient publication from hidden internal CPU materialization.
- [ ] **CUDATRAIN-02**: CUDA scatter/index-gradient blockers for `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` preserve duplicate-index, bounds, and dtype semantics before any native support claim.
- [ ] **CUDATRAIN-03**: Representative CUDA training hot paths report native execution, tensor-array fallback, CPU fallback, materialization reason, and shortened-region evidence under hard gates.

### Custom Metal Kernel Execution Route

- [ ] **METALKERNEL-01**: The custom Metal kernel route executes at least one scoped kernel family through a real native bridge instead of returning `CUSTOM_KERNEL_UNAVAILABLE`.
- [ ] **METALKERNEL-02**: Custom Metal kernel execution has CPU parity, dtype/layout legality checks, buffer-binding evidence, and stable fallback behavior when the route is unavailable or unprofitable.
- [ ] **METALKERNEL-03**: Route reports distinguish MPSGraph, custom Metal kernel, tensor-array fallback, and CPU fallback decisions with selected and rejected route evidence.

### Metal Output Buffer Write And Copy Closure

- [ ] **METALCOPY-01**: Metal output-buffer behavior is proven with sentinel/alias tests as `TRUE_OUTPUT_BUFFER_WRITE` or remains explicitly classified as `MPSGRAPH_RESULT_COPY`.
- [ ] **METALCOPY-02**: If true output-buffer writes cannot be proven, a lower-copy alias/materialization strategy reduces avoidable native copies without changing public `Tensor` semantics.
- [ ] **METALCOPY-03**: Benchmark and trace reports expose native device copy strategy, native copy timing, output alias/write status, and regression gates for unexpected copy reintroduction.

### Cross-Backend Router Calibration And Regression Gates

- [ ] **BACKENDROUTE-01**: The accelerator router uses calibrated shape, dtype, layout, route, copy, and fallback evidence to choose among MPSGraph, custom Metal kernels, CUDA, tensor-array fallback, and CPU fallback.
- [ ] **BACKENDROUTE-02**: Representative transformer, MLP, conv/pool, indexing, loss, and training workloads fail gates on hidden CPU exits, tensor-array replay, unsupported route overclaims, or unexpected native-copy regressions.
- [ ] **BACKENDROUTE-03**: Documentation, coverage summaries, and milestone audit artifacts make backend parity, custom-kernel scope, CUDA capability skips, and copy strategy boundaries reviewable.

## Future Requirements

Deferred to later milestones.

### Extended Accelerator Coverage

- **ACCELNEXT-01**: Add sparse, dynamic-shape, higher-rank, grouped/depthwise/dilated, and full training variants after v1.6 high-impact parity paths are stable.
- **ACCELNEXT-02**: Promote real CUDA hardware performance profiles into canonical fixtures only after a CUDA-capable validation lane is available.
- **ACCELNEXT-03**: Consider public device tensor APIs only if a later architecture decision explicitly changes the public `Tensor` boundary.

## Out of Scope

Explicitly excluded from v1.6 to keep the milestone reviewable.

| Feature | Reason |
|---------|--------|
| Public GPU tensor/device API | Public `Tensor` remains logical; device residency belongs in compile/prepare/execute runtime state. |
| Universal CUDA coverage | v1.6 targets the v1.5 Metal-supported high-impact families and keeps unsupported rows explicit. |
| Universal custom Metal kernel replacement for MPSGraph | v1.6 proves a real scoped custom-kernel route; MPSGraph remains the default for supported graph primitives. |
| Claiming zero-copy without proof | Output-buffer write must be proven by alias/sentinel evidence or remain classified as copy/lower-copy. |
| Committing local benchmark/calibration artifacts | Local profile output remains non-canonical unless intentionally promoted as a fixture. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CUDAPARITY-01 | Phase 40 | Complete |
| CUDAPARITY-02 | Phase 40 | Complete |
| CUDAPARITY-03 | Phase 40 | Complete |
| CUDADTYPE-01 | Phase 41 | Pending |
| CUDADTYPE-02 | Phase 41 | Pending |
| CUDAINDEX-01 | Phase 41 | Pending |
| CUDANN-01 | Phase 42 | Pending |
| CUDANN-02 | Phase 42 | Pending |
| CUDANN-03 | Phase 42 | Pending |
| CUDATRAIN-01 | Phase 43 | Pending |
| CUDATRAIN-02 | Phase 43 | Pending |
| CUDATRAIN-03 | Phase 43 | Pending |
| METALKERNEL-01 | Phase 44 | Pending |
| METALKERNEL-02 | Phase 44 | Pending |
| METALKERNEL-03 | Phase 44 | Pending |
| METALCOPY-01 | Phase 45 | Pending |
| METALCOPY-02 | Phase 45 | Pending |
| METALCOPY-03 | Phase 45 | Pending |
| BACKENDROUTE-01 | Phase 46 | Pending |
| BACKENDROUTE-02 | Phase 46 | Pending |
| BACKENDROUTE-03 | Phase 46 | Pending |

**Coverage:**
- v1.6 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

---
*Requirements defined: 2026-05-02*
*Last updated: 2026-05-02 after v1.6 milestone start*
