# Requirements: Synaptik v1.5 Production-Grade Metal Backend Expansion

**Defined:** 2026-05-02
**Core Value:** Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## v1.5 Requirements

Requirements for the v1.5 milestone. Each requirement maps to exactly one roadmap phase.

### Metal DType ABI And Capability Truth

- [x] **METALDTYPE-01**: Metal capability probing distinguishes representable storage dtypes, legal compute dtypes, legal output dtypes, and operation-specific dtype support for `FLOAT32`, `BFLOAT16`, `BOOL`, `INT32`, and `FLOAT64`.
- [x] **METALDTYPE-02**: A versioned Metal native ABI can carry dtype metadata beyond the current `_f32` path without letting older `.dylib` builds silently claim unsupported dtype execution.
- [x] **METALDTYPE-03**: Coverage reports and planner diagnostics expose dtype-specific support, fallback, and capability reasons without conflating dtype residency with native dtype compute.

### BF16 Metal Compute And Output

- [x] **METALBF16-01**: Legal `BFLOAT16` Metal regions can execute supported matmul/linear, elementwise, softmax/log-softmax, reduction, and normalization flows or reject with stable operation-specific capability reasons.
- [x] **METALBF16-02**: BF16 buffer binding, materialization, tolerance policy, and report evidence preserve CPU parity against the existing CPU BF16 semantics.
- [x] **METALBF16-03**: BF16 hot-path workloads show reduced CPU exits without regressing existing `FLOAT32` Metal or CPU hot paths.

### BOOL-Producing Metal Compute

- [x] **METALBOOL-01**: Metal can produce native device-resident `BOOL` outputs for supported compare and logical operations instead of supporting `BOOL` only as an external `WHERE` predicate input.
- [x] **METALBOOL-02**: Device-resident BOOL masks can feed `WHERE`, masked attention lowering, and other legal consumers without CPU materialization between supported GPU-region steps.
- [x] **METALBOOL-03**: `REDUCE_ALL` and `REDUCE_ANY` either execute as native Metal BOOL reductions or reject with stable dtype/rank/layout reasons.

### INT32 Index Tensor And Gather Take Path

- [x] **METALINTIDX-01**: Metal can carry `INT32` index tensors through native buffer bindings, runtime residency, and trace/report evidence without treating index residency as compute support.
- [x] **METALINTIDX-02**: Forward `GATHER` and `TAKE_ALONG_AXIS` have Metal lowering, bounds behavior, rank/axis semantics, CPU parity, and explicit fallback for unsupported layouts or index dtypes.
- [x] **METALINTIDX-03**: Supported index-forward regions preserve adjacent GPU residency and do not force materialization of legal producers or consumers.

### GPU Layout Router And Strided Materialization

- [ ] **METALLAYOUT-01**: Metal planning distinguishes metadata-only views, dense GPU-side materialization, zero-stride broadcast materialization, and unsupported strided compute with stable reason codes.
- [ ] **METALLAYOUT-02**: Legal layout transforms needed by v1.5 dtype, SDPA, conv/pool, indexing, and loss flows can run GPU-side without CPU materialization between compatible Metal regions.
- [ ] **METALLAYOUT-03**: Coverage gates fail unexpected CPU materialization caused by layout repair on representative non-contiguous/view workloads.

### Masked And Causal SDPA

- [ ] **METALSDPAMASK-01**: Metal direct SDPA semantics cover public `BOOL` masks, floating additive masks, causal masks, scale, rank, broadcast, and dtype legality before planner admission.
- [ ] **METALSDPAMASK-02**: Supported masked and causal SDPA paths execute through verified MPSGraph/native primitive DAGs or lowered sub-DAGs with CPU parity and trace-visible mask handling.
- [ ] **METALSDPAMASK-03**: Transformer attention coverage gates prove masked/causal attention stays on Metal for supported cases and rejects unsupported mask semantics visibly.

### Conv Pool Native Execution

- [ ] **METALCONVPOOL-01**: Metal supports selected `CONV2D` and `CONV2D_GEMM` forward paths for legal NCHW `FLOAT32` or supported dtype/layout combinations with stride, padding, dilation, and groups semantics locked.
- [ ] **METALCONVPOOL-02**: Metal supports selected `MAX_POOL2D` and `AVG_POOL2D` forward paths with pooling tie behavior and average-pool divisor semantics matching CPU.
- [ ] **METALCONVPOOL-03**: Conv/pool benchmark targets report native backend path, lowered primitive count, region length, and CPU exit reduction.

### Scatter And Index Gradient Semantics

- [ ] **METALSCATTER-01**: `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` have explicit duplicate-index accumulation semantics and either native Metal execution or stable unsupported rejection.
- [ ] **METALSCATTER-02**: Index gradient paths preserve CPU parity for duplicate indices, bounds behavior, and dtype/layout edge cases.
- [ ] **METALSCATTER-03**: Supported scatter/index-gradient regions can remain device-owned across adjacent Metal producers and consumers.

### Loss-Adjacent Metal Lowering

- [ ] **METALLOSS-01**: Dense `NLL_LOSS` and `CROSS_ENTROPY_LOSS` have Metal lowering or backend primitives for supported reduction modes and dtype/layout contracts.
- [ ] **METALLOSS-02**: Index-target CE/NLL variants preserve `INT32` target semantics, ignore-index masking, class weights, bounds behavior, and reduction denominator semantics before any supported Metal admission.
- [ ] **METALLOSS-03**: Loss-adjacent training workloads reduce CPU boundaries while keeping unsupported loss variants explicitly visible in traces and benchmark reports.

### Metal Training Backward Coverage

- [ ] **METALTRAIN-01**: Backward coverage for v1.5-supported forward families keeps gradients on Metal where semantics and backend capability permit it.
- [ ] **METALTRAIN-02**: Conv/pool, masked SDPA, loss/indexing, normalization variants, BF16, BOOL, and INT32-related backward paths have either verified Metal execution or stable capability-gated rejection.
- [ ] **METALTRAIN-03**: Training-mode coverage gates distinguish true gradient publication boundaries from avoidable CPU materialization inside the backward graph.

### Metal Backend Router And Zero-Copy Closure

- [ ] **METALROUTER-01**: Metal execution planning can choose among MPSGraph, custom Metal kernels, existing buffer binding, tensor-array fallback, and CPU fallback using calibrated shape/dtype/layout evidence.
- [ ] **METALROUTER-02**: The remaining native MPSGraph result-copy path is either proven safe as a true output-buffer write contract or replaced with an explicit lower-copy alias/materialization strategy.
- [ ] **METALROUTER-03**: Final v1.5 coverage reports prove supported hot paths stay on Metal, quantify router decisions, and fail hidden CPU exits, tensor-array replay, or unexpected native copy regressions.

## Future Requirements

Deferred to later milestones.

### CUDA Parity

- **CUDAPARITY-01**: Bring CUDA coverage to parity with the Metal v1.5 supported operation families after Metal semantics and gates are stable.
- **CUDAPARITY-02**: Validate native CUDA hardware performance and correctness on a real CUDA lane instead of only portable capability-gated tests.

### Extended Metal Coverage

- **METALEXT-01**: Support sparse, dynamic-shape, and higher-rank operation variants beyond the v1.5 representative workload set.
- **METALEXT-02**: Add user-facing device tensor APIs only if a later architecture decision explicitly changes the public Tensor boundary.

## Out of Scope

Explicitly excluded from v1.5 to keep the milestone reviewable.

| Feature | Reason |
|---------|--------|
| Public GPU tensor/device API | Public `Tensor` remains logical; device residency belongs in compile/prepare/execute runtime state. |
| CUDA implementation parity | v1.5 is Metal-first; shared contracts must stay backend-neutral, but CUDA native implementation follows later. |
| Universal Metal operation coverage | v1.5 targets the listed high-impact dtype, NN, indexing, layout, training, and routing gaps first. |
| Silent fallback or support labels without execution proof | A supported row must have lowering, legality, native execution or verified router behavior, CPU parity, and report evidence. |
| Committing local benchmark/calibration artifacts | Local profile output remains non-canonical unless intentionally promoted as a fixture. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| METALDTYPE-01 | Phase 29 | Complete |
| METALDTYPE-02 | Phase 29 | Complete |
| METALDTYPE-03 | Phase 29 | Complete |
| METALBF16-01 | Phase 30 | Complete |
| METALBF16-02 | Phase 30 | Complete |
| METALBF16-03 | Phase 30 | Complete |
| METALBOOL-01 | Phase 31 | Complete |
| METALBOOL-02 | Phase 31 | Complete |
| METALBOOL-03 | Phase 31 | Complete |
| METALINTIDX-01 | Phase 32 | Complete |
| METALINTIDX-02 | Phase 32 | Complete |
| METALINTIDX-03 | Phase 32 | Complete |
| METALLAYOUT-01 | Phase 33 | Pending |
| METALLAYOUT-02 | Phase 33 | Pending |
| METALLAYOUT-03 | Phase 33 | Pending |
| METALSDPAMASK-01 | Phase 34 | Pending |
| METALSDPAMASK-02 | Phase 34 | Pending |
| METALSDPAMASK-03 | Phase 34 | Pending |
| METALCONVPOOL-01 | Phase 35 | Pending |
| METALCONVPOOL-02 | Phase 35 | Pending |
| METALCONVPOOL-03 | Phase 35 | Pending |
| METALSCATTER-01 | Phase 36 | Pending |
| METALSCATTER-02 | Phase 36 | Pending |
| METALSCATTER-03 | Phase 36 | Pending |
| METALLOSS-01 | Phase 37 | Pending |
| METALLOSS-02 | Phase 37 | Pending |
| METALLOSS-03 | Phase 37 | Pending |
| METALTRAIN-01 | Phase 38 | Pending |
| METALTRAIN-02 | Phase 38 | Pending |
| METALTRAIN-03 | Phase 38 | Pending |
| METALROUTER-01 | Phase 39 | Pending |
| METALROUTER-02 | Phase 39 | Pending |
| METALROUTER-03 | Phase 39 | Pending |

**Coverage:**
- v1.5 requirements: 33 total
- Mapped to phases: 33
- Unmapped: 0

---
*Requirements defined: 2026-05-02*
*Last updated: 2026-05-02 after v1.5 milestone planning*
