# Phase 32: INT32 Index Tensor And Gather Take Path - Context

**Gathered:** 2026-05-02
**Status:** Ready for planning
**Source:** Roadmap/requirements context plus live codebase inspection

<domain>
## Phase Boundary

Phase 32 adds Metal support for `INT32` index tensors as native buffer inputs and admits scoped forward `GATHER` and `TAKE_ALONG_AXIS` execution for legal dense cases.

The phase is not a full scatter/index-training phase. `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` remain Phase 36 unless a small rejection/gate change is needed to keep diagnostics stable.
</domain>

<decisions>
## Implementation Decisions

### Locked Scope

- `INT32` residency is index-input support, not generic INT32 arithmetic or INT32 output compute.
- Metal may accept `INT32` external inputs only in legal index roles for supported index operations.
- Forward `GATHER` and `TAKE_ALONG_AXIS` are the only native index compute targets for this phase.
- Legal initial execution scope should be dense inputs, dense index tensors, rank/axis combinations proven by CPU parity tests, and explicit bounds behavior.
- CUDA must remain capability-gated/unsupported for native INT32 index compute unless a CUDA implementation is explicitly added later.
- Unsupported layouts, index dtypes, ranks, axes, native symbol gaps, and bounds cases must reject or fallback with stable visible reasons.

### the agent's Discretion

- Choose exact MPSGraph primitive path or native shim helper after checking local SDK/native compile behavior.
- Decide whether scoped BF16 value gather/take is admitted in Phase 32 or deferred behind a stable unsupported dtype reason.
- Add a representative `gather_take_small` or similar coverage target if existing coverage targets do not expose the new path cleanly.
</decisions>

<canonical_refs>
## Canonical References

### Planning

- `.planning/PROJECT.md` — milestone goals and accelerator architecture constraints.
- `.planning/ROADMAP.md` — Phase 32 goal, dependencies, success criteria, and planned waves.
- `.planning/REQUIREMENTS.md` — `METALINTIDX-01`, `METALINTIDX-02`, `METALINTIDX-03`.
- `.planning/phases/31-bool-producing-metal-compute/31-VERIFICATION.md` — prior dtype-role closure pattern and CUDA gating precedent.

### Code

- `src/main/java/tensor/ops/index/TensorIndexOps.java` — public gather/take/scatter graph construction and CPU semantic shape rules.
- `src/main/java/backend/cpu/kernels/index/` — CPU correctness oracle for gather/take and gradients.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` — current index rows and stable unsupported reasons.
- `src/main/java/backend/accelerator/dag/AcceleratorDagNodeType.java` — shared native DAG ABI operation codes.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` — shared DAG lowering and dtype residency evidence.
- `src/main/java/backend/metal/MetalMpsCapabilities.java` — Metal dtype and role-specific capability truth.
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` — Metal planner legality and stable rejection details.
- `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` — Java FFM bridge dtype validation and buffer execution contract.
- `src/main/native/apple/synaptik_apple_mps_stub.m` — native MPSGraph shim.

### Tests And Reports

- `src/test/java/GatherExecutionTest.java` — CPU gather behavior and shape/bounds expectations.
- `src/test/java/PreparedExecutionBuildTest.java` — partition/prepare evidence patterns for GPU index rejection and admission.
- `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` — native Metal parity patterns.
- `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java` — matrix contract tests.
- `src/test/java/GpuHotPathCoverageTargetsTest.java` and `src/test/java/GpuCoverageRegressionGateTest.java` — hard coverage-gate pattern.
</canonical_refs>

<specifics>
## Specific Ideas

- Add DAG nodes such as `GATHER` and `TAKE_ALONG_AXIS` with axis metadata encoded in the existing scalar/metadata path.
- Treat `INT32` index tensors similarly to BOOL predicate tensors: role-legal external inputs, not broad compute/output.
- Keep gather/take outputs dtype-preserving for the value tensor; do not add INT32 output publication unless the operation actually produces an INT32 output.
- Native parity should assert representative axis 0/axis 1/rank 2 and rank 3 cases, plus visible rejection for non-dense or unsupported ranks.
</specifics>

<deferred>
## Deferred Ideas

- Native `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` duplicate-index semantics are Phase 36.
- Loss-adjacent index-target CE/NLL is Phase 37.
- Custom Metal kernels and router selection remain Phase 39 unless MPSGraph cannot support the minimal forward gather/take path and a tiny fallback kernel is explicitly justified.
</deferred>

---

*Phase: 32-int32-index-tensor-and-gather-take-path*
*Context gathered: 2026-05-02*
