# Phase 34: Masked And Causal SDPA - Context

**Gathered:** 2026-05-02
**Status:** Ready for implementation planning
**Source:** Roadmap/requirements context plus live codebase inspection

<domain>
## Phase Boundary

Phase 34 admits verified masked and causal Metal forward SDPA without weakening Synaptik public mask semantics. Public attention masks are `BOOL` tensors; causal attention is currently represented by Tensor front-end mask construction and becomes an effective `BOOL` mask on the `SCALED_DOT_PRODUCT_ATTENTION` operation. Metal may use MPSGraph or a lowered primitive DAG internally, but the planner must not treat a backend floating/additive mask convention as equivalent until CPU parity and trace evidence prove the conversion.

This phase is Metal-first. Shared accelerator DAG contracts may be extended for mask metadata, but CUDA must keep visible capability-gated rejection until CUDA has its own native execution proof.
</domain>

<decisions>
## Implementation Decisions

### Locked Scope

- Public `Tensor` remains logical; mask residency and generated causal masks belong in compile/prepare/execute runtime state.
- Supported direct Metal SDPA remains scoped to dense `FLOAT32` query/key/value/output rank-3 or rank-4 tensors unless this phase explicitly proves a wider dtype.
- Public `BOOL` mask semantics are authoritative: `true` means an attention score is allowed, `false` means it is excluded before softmax.
- Causal semantics must match `AttentionOptions.causalDefaults()`: query position `i` can attend only to key positions `j <= i`, with any external mask combined by logical AND.
- Additive/floating mask support is allowed only as an internal lowering representation, with explicit conversion from public BOOL/causal semantics to backend score bias behavior.
- Broadcast and layout handling must consume Phase 33 layout router results; unsupported mask layout/dtype cases must reject with stable reason codes instead of hidden CPU materialization.
- Direct masked SDPA support must be visible in lowering manifests, trace/debug fields, and hot-path coverage gates.
- `GPU_CUDA` direct masked/causal SDPA remains `CAPABILITY_MISSING` or equivalent visible rejection.

### Agent Discretion

- Choose whether direct masked SDPA is represented by a new DAG node flag/mask input contract or by lowering to existing primitives (`MATMUL`, `WHERE`/additive fill, `SOFTMAX`, `MATMUL`) inside one Metal region.
- Choose whether causal-only SDPA generates the causal mask in Java planning metadata, native MPSGraph, or a lowered region-internal primitive, as long as no CPU materialization boundary is introduced for supported cases.
- Choose whether the first executable path supports external BOOL mask, causal-only mask, or both in the same wave, as long as unsupported combinations remain explicit.
</decisions>

<canonical_refs>
## Canonical References

### Planning

- `.planning/PROJECT.md` - v1.5 architecture constraints and Metal-first/CUDA-gated rule.
- `.planning/ROADMAP.md` - Phase 34 goal, success criteria, and dependencies.
- `.planning/REQUIREMENTS.md` - `METALSDPAMASK-01`, `METALSDPAMASK-02`, `METALSDPAMASK-03`.
- `.planning/phases/31-bool-producing-metal-compute/31-VERIFICATION.md` - BOOL mask residency foundation.
- `.planning/phases/33-gpu-layout-router-and-strided-materialization/33-VERIFICATION.md` - layout repair foundation used by mask tensors.

### Code

- `src/main/java/tensor/ops/linalg/TensorAttentionOps.java` - public attention construction, causal mask creation, and effective BOOL mask expansion.
- `src/main/java/tensor/options/AttentionOptions.java` - causal and scale semantics.
- `src/main/java/operations/linalg/scaledDotProductAttention.java` - current direct SDPA descriptor with `scale` and `hasMask`.
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` - current direct masked SDPA rejection and rank/dtype/layout gates.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` - direct SDPA and generic attention-pattern lowering.
- `src/main/java/backend/accelerator/dag/AcceleratorDagNodeType.java` and `AcceleratorDagNode.java` - shared DAG ABI that currently represents unmasked `SDPA`.
- `src/main/java/backend/metal/MetalMpsCapabilities.java` - role-sensitive dtype/input legality and SDPA external input restrictions.
- `src/main/native/apple/synaptik_apple_mps_stub.m` - current native SDPA primitive DAG (`Q * K^T`, scale, softmax, `* V`).
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` - prepare/execute gating, trace evidence, and forward-SDPA safety checks.

### Tests And Reports

- `src/test/java/AttentionExecutionTest.java` - CPU/public parity for unmasked, causal, and external BOOL mask SDPA.
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` - current Metal direct masked SDPA rejection test.
- `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` - native bridge SDPA parity tests.
- `src/test/java/PreparedExecutionBuildTest.java` - selected Metal region and fallback evidence tests.
- `src/test/java/GpuCoverageRegressionGateTest.java`, `GpuHotPathCoverageTargetsTest.java`, and `GpuCoverageSummaryTest.java` - coverage gate/report contracts.
- `src/main/java/tuning/workload/StandardWorkloads.java` and `CalibrationWorkloads.java` - transformer and masked-attention workload sources.
</canonical_refs>

<current_state>
## Current Codebase Facts

- Direct unmasked Metal forward SDPA is supported for dense `FLOAT32` rank-3/rank-4 inputs and lowers to a single `SDPA` DAG node with scale bits.
- `MetalPartitionSupport.sdpaUnsupportedReason(...)` rejects any `scaledDotProductAttention` with `attention.hasMask()` using `UNSUPPORTED_MASK_SEMANTICS`.
- `TensorAttentionOps` creates causal masks as `BOOL`, combines external and causal masks through logical AND, expands the effective mask to scores shape, and then records `hasMask=true` on the direct SDPA operation.
- The native Objective-C shim implements ABI node type `26` as unmasked scores, optional scale, softmax, and value matmul; it rejects `input3 != nil`.
- Existing generic `matmul -> scale -> where -> softmax -> matmul` pattern lowering deliberately returns `null` when a `WHERE` mask is present, to avoid confusing BOOL masks with native floating masks.
- Phase 31 made Metal BOOL-producing masks and `WHERE` mask chains device-resident; Phase 33 made dense and broadcast layout repair visible through the layout router.
</current_state>

<deferred>
## Deferred Ideas

- CUDA masked/causal SDPA execution parity is out of scope for v1.5 Metal-first Phase 34.
- BF16 masked SDPA is out of scope unless FLOAT32 support proves the same backend path safely and the dtype capability model is widened deliberately.
- Backward masked SDPA completeness belongs to Phase 38 unless a small forward-adjacent trace field is needed now.
- A public additive-mask Tensor API is not required in this phase; additive/floating masks may be an internal lowering representation only.
</deferred>

---

*Phase: 34-masked-and-causal-sdpa*
*Context gathered: 2026-05-02*
