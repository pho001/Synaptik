# Phase 17: Normalization Reduction And Loss-Adjacent Lowering - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning
**Source:** Auto context from roadmap, requirements, Phase 14 targets, Phase 15 manifest contract, Phase 16 dtype residency, and codebase inspection.

<domain>
## Phase Boundary

Phase 17 expands the shared Metal/CUDA lowering contract for high-impact normalization, reduction, softmax-ish, conv/normalization-style, and loss-adjacent gaps. The phase should either implement a supported GPU lowering path or record a stable, precise rejection that is tied to the hot-path gap that blocked the region.

The phase is not a broad native library routing milestone. Vendor library routing remains deferred to `GPULIB-*`. Phase 17 should prefer backend-neutral legality, matrix, manifest, trace, and parity coverage that later Phase 18 and Phase 19 can consume.
</domain>

<decisions>
## Implementation Decisions

### Coverage Contract
- D-01: `GpuLoweringCoverageMatrix` remains the shared Metal/CUDA source of truth for Phase 17 families.
- D-02: Phase 17 must explicitly cover `LAYER_NORM`, `RMS_NORM`, `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX`, `SOFTMAX`, `LOG_SOFTMAX`, `NLL_LOSS`, `CROSS_ENTROPY_LOSS`, `CROSS_ENTROPY_LOSS_INDICES`, and `CROSS_ENTROPY_LOSS_INDICES_GRAD` for both `GPU_METAL` and `GPU_CUDA`.
- D-03: Coverage rows must distinguish `SUPPORTED`, `FALLBACK`, and `UNSUPPORTED`; unsupported cases must use stable `GpuLoweringUnsupportedReason` codes instead of generic prose-only CPU replay.
- D-04: Phase 14 hot-path targets `layer_norm_small`, `conv2d_resnet_3x3`, and `transformer_block_hot_path` must be visible in Phase 17 target evidence or notes.

### Backend Legality
- D-05: Metal and CUDA legality remain shared at the semantic contract level through `GpuLoweringCoverageMatrix`, while backend-specific capability gates stay in `MetalPartitionSupport` and `CudaGpuRegionLegalityAdapter`.
- D-06: Reduction and normalization rejections should include the `REDUCTION_ADJACENT` prefix when the reason is region shape or compound execution, not native dtype residency.
- D-07: Layout and dtype legality from Phases 9, 10, and 16 must continue to run before any new lowering is considered legal.
- D-08: CUDA direct non-dense compute remains conservative unless a task explicitly materializes or propagates a legal device-side layout path.

### Softmax, Loss, And Numeric Parity
- D-09: `LOG_SOFTMAX` remains a supported softmax-ish lowering as `SOFTMAX` followed by `LOG`; this support must stay distinct from loss-adjacent rejection.
- D-10: Loss-adjacent operations with INT32 targets must report `UNSUPPORTED_DTYPE` when the target dtype blocks accelerator execution.
- D-11: Numerically sensitive flows must compare against CPU with explicit tolerances appropriate to dtype and operation family.
- D-12: True CPU consumers, graph outputs, and gradient publication may materialize; internal supported GPU-region values must not silently materialize.

### Observability And Closure
- D-13: Prepare traces, coverage summaries, and benchmark reports must expose Phase 17 support/rejection evidence without scraping native bridge logs.
- D-14: Native Metal/CUDA execution remains capability-gated; portable JUnit tests must prove contract behavior without requiring local GPU hardware.
- D-15: Local files under `profiles/platform/.../tuning/abc/*` are calibration artifacts and must remain unstaged.
</decisions>

<canonical_refs>
## Canonical References

### Planning And Requirements
- `.planning/ROADMAP.md` - Phase 17 goal, dependencies, and success criteria.
- `.planning/REQUIREMENTS.md` - `GPUNORM-01`, `GPUNORM-02`, and `GPUNORM-03`.
- `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md` - source-of-truth target workloads for v1.3.
- `.planning/phases/15-gpu-region-internal-lowered-dag-contract/15-VERIFICATION.md` - lowered-region manifest contract.
- `.planning/phases/16-dtype-and-storage-residency-expansion/16-VERIFICATION.md` - dtype residency and `UNSUPPORTED_DTYPE` contract.

### Lowering And Legality
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` - shared Metal/CUDA coverage matrix.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageEntry.java` - coverage row record.
- `src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java` - stable unsupported reason vocabulary.
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` - Metal planner legality and diagnostics.
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java` - CUDA planner legality and diagnostics.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` - shared DAG lowering and `LOG_SOFTMAX` expansion.

### Tests And Reports
- `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java` - coverage matrix tests.
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` - Metal legality/lowering tests.
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` - CUDA legality/lowering tests.
- `src/test/java/PreparedExecutionBuildTest.java` - prepare/build required-mode behavior and CPU parity tests.
- `src/test/java/CompiledGraphTraceTest.java` - prepare trace evidence tests.
- `src/test/java/GpuCoverageSummaryTest.java` - coverage summary tests.
- `src/test/java/BenchmarkSessionTest.java` - benchmark report rendering tests.
</canonical_refs>

<specifics>
## Specific Ideas

- Start by making the matrix and tests Phase 17-specific before changing backend legality strings.
- Do not claim `LAYER_NORM`, `RMS_NORM`, `SUM`, or loss-adjacent native compute unless lowering, legality, trace evidence, and CPU parity tests prove it.
- Prefer exact rejection evidence for the first pass if backend-native arithmetic would require a broader kernel or vendor-library strategy.
- Keep `LOG_SOFTMAX` support green while making loss-adjacent operations explicitly rejected when target dtype or region semantics are unsupported.
</specifics>

<deferred>
## Deferred Ideas

- cuBLAS, cuDNN, MPSGraph, or backend-native library routing for reductions/norms/losses.
- Universal reduction kernels for every axis/rank/dtype combination.
- Multi-op region execution across normalization boundaries; that belongs to Phase 19.
- Region-internal fusion of elementwise and epilogue chains; that belongs to Phase 18.
</deferred>

---

*Phase: 17-normalization-reduction-and-loss-adjacent-lowering*
*Context gathered: 2026-05-01 via auto plan-phase*
