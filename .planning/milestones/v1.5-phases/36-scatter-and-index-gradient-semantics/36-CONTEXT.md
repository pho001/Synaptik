# Phase 36: Scatter And Index Gradient Semantics - Context

**Gathered:** 2026-05-02
**Status:** Ready for planning
**Source:** Auto context from roadmap, requirements, Phase 32 verification, and current codebase

<domain>
## Phase Boundary

Phase 36 covers the remaining index write/gradient surface left open by Phase 32:

- `SCATTER_ADD`
- `GATHER_GRAD`
- `TAKE_ALONG_AXIS_GRAD`

The phase must either support these paths on Metal with CPU parity and trace evidence, or keep them explicitly rejected with stable reason codes. It must not blur Phase 32 forward `GATHER` / `TAKE_ALONG_AXIS` support with index-gradient support.
</domain>

<decisions>
## Implementation Decisions

### Locked By Roadmap
- Duplicate-index accumulation order/tolerance semantics must be explicit before planner admission.
- `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` must either execute on Metal or reject with stable duplicate/bounds reasons.
- CPU parity must cover duplicate indices, repeated indices, out-of-range handling, and gradient scatter.
- Supported scatter/index-gradient flows must preserve device residency.
- Reports must distinguish forward index support from index-gradient support.

### Existing Context
- Phase 32 already supports Metal forward `GATHER` and `TAKE_ALONG_AXIS` for dense `FLOAT32` value/output tensors with dense static in-bounds `INT32` indices.
- `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` are currently matrix-listed as unsupported with `UNSUPPORTED_DUPLICATE_INDEX`.
- CPU kernels implement accumulation by iterating logical index order and using additive updates into the destination.

### the agent's Discretion
- Choose native MPSGraph routing, a custom Metal kernel, or a stable rejection path per operation after semantics and backend proof.
- Prefer a narrow supported subset over a broad unproven claim.
- CUDA remains explicit capability-gated unless implemented separately.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope
- `.planning/ROADMAP.md` - Phase 36 goal, dependencies, success criteria, and wave outline.
- `.planning/REQUIREMENTS.md` - `METALSCATTER-01`, `METALSCATTER-02`, `METALSCATTER-03`.
- `.planning/phases/32-int32-index-tensor-and-gather-take-path/32-VERIFICATION.md` - completed forward index support and explicit non-goals.

### Existing Code
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` - current index-gradient matrix rows and reason codes.
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` - Metal index legality checks and forward gather/take admission.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` - shared DAG lowerer.
- `src/main/java/backend/cpu/kernels/index/IndexReadWriteBackend.java` - CPU accumulation semantics.
- `src/main/java/tensor/ops/index/TensorIndexOps.java` - public index/backward graph construction.
- `src/main/native/apple/synaptik_apple_mps_stub.m` - Metal native graph execution shim.

### Tests And Reports
- `src/test/java/ScatterAddExecutionTest.java`
- `src/test/java/GatherExecutionTest.java`
- `src/test/java/TakeAlongAxisExecutionTest.java`
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java`
- `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java`
- `src/test/java/GpuHotPathCoverageTargetsTest.java`
- `src/test/java/GpuCoverageSummaryTest.java`
</canonical_refs>

<specifics>
## Specific Ideas

- Add duplicate-index fixtures where several source positions map to the same destination and verify deterministic CPU-compatible sums.
- Add bounds fixtures for negative/out-of-range indices and keep CPU exception parity.
- Treat `GATHER_GRAD` as scatter-add into a zero-shaped input-gradient tensor.
- Treat `TAKE_ALONG_AXIS_GRAD` as rank-preserving scatter-add into the input-gradient tensor.
- Add trace/report fields or reason strings that separate `forwardIndexSupported=true` from `indexGradientSupported=true`.
</specifics>

<deferred>
## Deferred Ideas

- Generic INT32 compute/output remains out of scope.
- CUDA native index-gradient execution remains out of scope.
- Loss-adjacent index-target CE/NLL remains Phase 37.
- Broad non-dense/strided index-gradient compute remains out of scope unless layout repair makes a dense Metal path explicit.
</deferred>

---

*Phase: 36-scatter-and-index-gradient-semantics*
*Context gathered: 2026-05-02 via auto context*
