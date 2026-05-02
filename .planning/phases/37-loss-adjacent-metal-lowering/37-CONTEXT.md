# Phase 37: Loss-Adjacent Metal Lowering - Context

**Gathered:** 2026-05-02
**Status:** Ready for planning
**Source:** Roadmap + current codebase inspection

## Phase Boundary

Phase 37 owns Metal loss-adjacent coverage for dense and index-target loss flows. It must not hide CPU exits behind broad GPU support claims.

The phase should deliver:

- scoped dense `NLL_LOSS` / `CROSS_ENTROPY_LOSS` Metal lowering or explicit stable rejection where the formula/reduction contract is not representable,
- index-target CE/NLL semantics gates for `INT32` target bounds, ignore-index, class weights, and denominator behavior,
- training/backward trace gates that reduce avoidable CPU boundaries without claiming unsupported scatter/index-gradient behavior,
- coverage/report separation between supported dense loss-adjacent flows and unsupported index-target variants.

## Locked Decisions

- Public `Tensor` remains logical; loss residency belongs in compile/prepare/execute state.
- Use existing GPU region lowering patterns instead of adding a public GPU tensor API.
- Dense loss support may be lowered through existing primitives such as `LOG_SOFTMAX`, `MUL`, `SUM`, `MEAN`, `NEG`, and reductions if CPU parity is proven.
- Index-target support must respect Phase 36: `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` remain stable `UNSUPPORTED_DUPLICATE_INDEX` blockers.
- Forward index-target CE can be considered separately from backward/index-gradient support, but reports must say exactly what was supported.
- Ignore-index and class-weight semantics are correctness gates, not performance polish.
- Unsupported variants must stay visible in planner diagnostics, prepare traces, and benchmark reports.

## Canonical References

- `.planning/ROADMAP.md` - Phase 37 goal, dependencies, success criteria, and wave outline.
- `.planning/REQUIREMENTS.md` - `METALLOSS-01`, `METALLOSS-02`, `METALLOSS-03`.
- `.planning/phases/36-scatter-and-index-gradient-semantics/36-VERIFICATION.md` - scatter/index-gradient blockers that Phase 37 must honor.
- `docs/gpu-lowering-coverage.md` - current loss-adjacent matrix and visible blocker contract.
- `docs/metal-backend.md` - Metal planner/lowering/runtime contract.
- `src/main/java/tensor/ops/loss/TensorLossOps.java` - public loss graph construction and backward behavior.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` - source-of-truth loss-adjacent matrix rows.
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` - Metal planner unsupported reasons.
- `src/main/java/tuning/benchmark/report/GpuHotPathCoverageTargets.java` - hot-path gates and report expectations.

## Deferred

- Full native index-target loss training support is deferred unless Phase 37 proves all target, ignore-index, class-weight, denominator, and gradient scatter semantics.
- Custom Metal kernels for loss are not required unless MPSGraph/lowered DAG support cannot meet the scoped correctness contract.

---

*Phase: 37-loss-adjacent-metal-lowering*
*Context gathered: 2026-05-02*
