# Requirements: Synaptik v1.3 Coverage-Driven GPU Region Expansion

**Defined:** 2026-05-01
**Core Value:** Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## v1.3 Requirements

Requirements for the v1.3 milestone. Each requirement maps to exactly one roadmap phase.

### Coverage Gap Triage

- [x] **GPUTRIAGE-01**: Engineers can inspect v1.2 coverage reports and identify the top fallback, CPU materialization, hidden tensor-array, and device handoff reasons by backend and workload.
- [x] **GPUTRIAGE-02**: Engineers can run deterministic transformer block, MLP, and conv/normalization-style workloads that expose GPU region length, CPU exits, and unsupported reason codes.
- [x] **GPUTRIAGE-03**: Engineers can rank GPU coverage gaps by hot-path impact so v1.3 operation, dtype, fusion, and region work closes measured exits before speculative coverage.

### GPU Region Internal Lowered DAG Contract

- [x] **GPUDAG-01**: A GPU region can represent multiple original graph operations as one backend-owned lowered DAG instead of assuming one selected region equals one operation.
- [x] **GPUDAG-02**: Region metadata records original operation IDs/types, lowered primitive IDs/types, fused subpattern summaries, backend ID, dtype/layout assumptions, and region length.
- [x] **GPUDAG-03**: Region planning and traces expose stable rejection, fallback, and materialization reason codes for each rejected original operation, lowered primitive, and fused subpattern.

### DType And Storage Residency

- [x] **GPUSTORAGE-01**: BFLOAT16, INT32, and BOOL device residency is represented in memory binding, slot reuse, and device buffer metadata where those dtypes currently force CPU exits.
- [x] **GPUSTORAGE-02**: Metal and CUDA residency decisions for non-FLOAT32/FLOAT64 values are capability-gated and report stable backend-specific unsupported reasons when native execution is not legal.
- [x] **GPUSTORAGE-03**: Focused tests prove supported dtype residency, CPU parity at true materialization boundaries, and no hidden CPU materialization for supported internal GPU-region values.

### Normalization, Reduction, And Loss-Adjacent Lowering

- [x] **GPUNORM-01**: The shared Metal/CUDA lowering coverage contract includes layer norm, RMS norm, reductions, softmax-ish residual flows, and loss-adjacent operation families with explicit support status.
- [x] **GPUNORM-02**: Highest-impact normalization, reduction, and softmax-ish hot-path gaps receive GPU lowering or a stable rejection reason tied to the measured coverage gap that blocked the region.
- [x] **GPUNORM-03**: Loss-adjacent and numerically sensitive flows preserve CPU parity, dtype/layout legality checks, and visible fallback when backend primitive coverage is incomplete.

### Fused Elementwise And Epilogue Subregions

- [x] **GPUFUSEX-01**: GPU regions can fuse supported elementwise chains internally without Java array round trips or intermediate CPU materialization between fused operations.
- [x] **GPUFUSEX-02**: GPU regions can lower supported linear or matmul epilogues such as bias and activation as region-internal fused subpatterns when backend legality and dtype/layout gates allow it.
- [x] **GPUFUSEX-03**: GPU fusion is implemented as region-internal lowering/fusion and must not reuse CPU `Operation.OpType.FUSED`, CPU fused ASM, or CPU vector dispatch internals.

### Multi-Op GPU Region Execution

- [x] **GPUMULTI-01**: A selected GPU region can execute multiple lowered operations, including matmul/linear, layout/view steps, elementwise chains, and selected softmax-ish or normalization primitives.
- [x] **GPUMULTI-02**: Supported internal steps hand off device-owned values through `ExecutionState` and device buffer bindings without CPU materialization until a true CPU consumer, graph output, or gradient publication boundary.
- [x] **GPUMULTI-03**: Multi-op GPU region execution remains backend-neutral at the planning contract level while allowing Metal and CUDA to choose backend-specific primitive execution strategies.

### Coverage Regression Hardening

- [x] **GPUHARDEN-01**: Coverage reports include lowered operation count, fused subpattern count, selected region length, rejected candidate count, CPU exit count, materialization reasons, and device handoff evidence.
- [x] **GPUHARDEN-02**: Regression gates fail when target hot paths unexpectedly shorten GPU regions, add CPU materialization, hide tensor-array fallback, or lose required lowered/fused coverage.
- [x] **GPUHARDEN-03**: Portable Java gates and capability-gated native Metal/CUDA checks document which evidence proves hot paths stayed on GPU and which native evidence was skipped by environment.

### Milestone Verification Evidence Closure

- [ ] **GPUAUDIT-01**: Phase 14 has a verification report that proves `GPUTRIAGE-01`, `GPUTRIAGE-02`, and `GPUTRIAGE-03` from existing summaries, validation evidence, docs, and focused test commands.
- [ ] **GPUAUDIT-02**: Phase 18 has a verification report that proves `GPUFUSEX-01`, `GPUFUSEX-02`, and `GPUFUSEX-03` from existing summaries, validation evidence, docs, and focused test commands.
- [ ] **GPUAUDIT-03**: The v1.3 milestone audit can pass without stale missing-phase findings, and Phase 20 validation metadata is either Nyquist-compliant or explicitly documented as strategy-only.

## Future Requirements

Deferred to later milestones. Tracked but not in the v1.3 roadmap.

### Broad Native Library Routing

- **GPULIB-01**: Route supported GPU lowered primitives through vendor libraries such as cuBLAS/cuDNN/MPSGraph where they measurably improve region execution and fit the shared contract.
- **GPULIB-02**: Add a backend-native primitive cost model that can choose among custom kernels, vendor library calls, and CPU fallback using calibrated backend evidence.

### Extended Coverage

- **GPUEXT-01**: Support additional high-rank, sparse, dynamic-shape, and advanced indexing patterns beyond the v1.3 hot-path workload set.
- **GPUEXT-02**: Validate native CUDA behavior on real CUDA hardware as canonical evidence, not only capability-skipped portable checks.

## Out of Scope

Explicitly excluded from v1.3 to keep the milestone coverage-driven and reviewable.

| Feature | Reason |
|---------|--------|
| Public GPU tensor/device API | Public `Tensor` remains logical; residency belongs in compile/prepare/execute runtime state. |
| Universal operation coverage | v1.3 closes measured high-impact coverage gaps first and keeps unsupported cases explicit. |
| Replacing CPU fused execution | CPU fused ASM/vector paths remain CPU-owned and performance-critical; GPU fusion must be region-internal lowering/fusion. |
| Silent fallback or best-effort replay | Hidden CPU exits invalidate the milestone goal; fallback must be traceable and gateable. |
| Backend-specific architecture forks | Metal and CUDA can execute differently, but shared planning contracts must remain backend-neutral. |
| Committing local benchmark or calibration artifacts | Local profiles under `profiles/platform/.../tuning/abc/*` are not milestone artifacts unless intentionally promoted to fixtures. |

## Traceability

Which phases cover which requirements. Updated during phase execution as requirements complete.

| Requirement | Phase | Status |
|-------------|-------|--------|
| GPUTRIAGE-01 | Phase 14 | Complete |
| GPUTRIAGE-02 | Phase 14 | Complete |
| GPUTRIAGE-03 | Phase 14 | Complete |
| GPUDAG-01 | Phase 15 | Complete |
| GPUDAG-02 | Phase 15 | Complete |
| GPUDAG-03 | Phase 15 | Complete |
| GPUSTORAGE-01 | Phase 16 | Complete |
| GPUSTORAGE-02 | Phase 16 | Complete |
| GPUSTORAGE-03 | Phase 16 | Complete |
| GPUNORM-01 | Phase 17 | Complete |
| GPUNORM-02 | Phase 17 | Complete |
| GPUNORM-03 | Phase 17 | Complete |
| GPUFUSEX-01 | Phase 18 | Complete |
| GPUFUSEX-02 | Phase 18 | Complete |
| GPUFUSEX-03 | Phase 18 | Complete |
| GPUMULTI-01 | Phase 19 | Complete |
| GPUMULTI-02 | Phase 19 | Complete |
| GPUMULTI-03 | Phase 19 | Complete |
| GPUHARDEN-01 | Phase 20 | Complete |
| GPUHARDEN-02 | Phase 20 | Complete |
| GPUHARDEN-03 | Phase 20 | Complete |
| GPUAUDIT-01 | Phase 21 | Pending |
| GPUAUDIT-02 | Phase 21 | Pending |
| GPUAUDIT-03 | Phase 21 | Pending |

**Coverage:**
- v1.3 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0

---
*Requirements defined: 2026-05-01*
*Last updated: 2026-05-01 after v1.3 milestone audit gap planning*
