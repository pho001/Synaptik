# Phase 33: GPU Layout Router And Strided Materialization - Context

**Gathered:** 2026-05-02
**Status:** Ready for implementation planning
**Source:** Roadmap/requirements context plus live codebase inspection

<domain>
## Phase Boundary

Phase 33 closes avoidable Metal CPU exits caused by layout repair. It does not make every operation strided-native. The target is a clear router contract that can keep legal view/layout flows on Metal by choosing metadata-only propagation, GPU-side dense materialization, scoped broadcast repair, or explicit rejection before hidden tensor-array or CPU fallback.

This phase feeds masked/causal SDPA, conv/pool, scatter/index gradients, and loss phases. Those later phases should be able to consume dense bindings produced by the layout router instead of solving layout repair independently.
</domain>

<decisions>
## Implementation Decisions

### Locked Scope

- Public `Tensor` remains logical; layout repair must stay in compile/prepare/execute runtime state.
- Existing metadata-only view behavior for `RESHAPE`, `PERMUTE`, `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `NOOP`, and `SELECT` must remain visible and backend-neutral.
- `CONTIGUOUS` and reshape-from-non-dense should continue to use GPU-side dense materialization where the bridge supports it.
- Broadcast zero-stride support must be explicit: either GPU-side materialization for scoped dense outputs or stable rejection with a broadcast-specific reason code.
- Selected strided compute admission is allowed only when the consumer operation can prove it consumes the layout directly; otherwise the router should produce a dense binding or reject.
- `GPU_CUDA` must keep existing capability-gated behavior. Shared contracts may be extended, but CUDA must not inherit Metal support claims accidentally.
- REQUIRE mode must fail on missing layout support before tensor-array replay or CPU fallback hides the boundary.

### Agent Discretion

- Choose whether the router is a new type or a conservative expansion of `AcceleratorLayoutTransformPlanner`, as long as the public contract distinguishes route classes and reason codes.
- Decide whether BF16/BOOL layout materialization is admitted in this phase or remains explicit rejection after checking native kernel cost/risk.
- Add a new representative workload name or extend existing layout workloads if that gives clearer coverage gates.
</decisions>

<canonical_refs>
## Canonical References

### Planning

- `.planning/PROJECT.md` — v1.5 architecture constraints and Metal-first/CUDA-gated rule.
- `.planning/ROADMAP.md` — Phase 33 goal, success criteria, and dependencies.
- `.planning/REQUIREMENTS.md` — `METALLAYOUT-01`, `METALLAYOUT-02`, `METALLAYOUT-03`.
- `.planning/phases/30-bf16-metal-compute-and-output/30-VERIFICATION.md` — BF16 dtype scope that layout repair must not break.
- `.planning/phases/31-bool-producing-metal-compute/31-VERIFICATION.md` — BOOL mask residency scope.
- `.planning/phases/32-int32-index-tensor-and-gather-take-path/32-VERIFICATION.md` — index path layout constraints and dense-index assumptions.

### Code

- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlanner.java` — current metadata-only vs dense materialization classifier.
- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformKind.java` — current route kind vocabulary.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java` — stable reason code registry.
- `src/main/java/graph/execution/DeviceLayoutViewPropagator.java` — runtime layout transform execution and materializer invocation.
- `src/main/java/graph/execution/PreparedExecution.java` — trace attributes such as `gpuLayoutMaterializationCount`.
- `src/main/java/backend/metal/buffer/MetalDeviceLayoutMaterializer.java` — Metal dense layout materializer, currently FLOAT32-only.
- `src/main/java/backend/metal/buffer/MetalLayoutPolicy.java` — Metal input/output layout policy and broadcast rejection.
- `src/main/java/backend/metal/buffer/MetalBufferAllocator.java` — output allocation and dtype/layout readback.
- `src/main/java/backend/metal/bridge/MetalMpsGraphBridge.java` and `MetalMpsFfmBridge.java` — bridge layout materialization SPI.
- `src/main/native/apple/synaptik_apple_mps_stub.m` — current `synaptik_layout_contiguous_f32` kernel.
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` — operation legality that currently rejects several non-dense consumers.

### Tests And Reports

- `src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java` — existing layout flow, broadcast fallback, and device residency tests.
- `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` — native layout materialization parity.
- `src/test/java/PreparedExecutionBuildTest.java` — layout fallback cost and REQUIRE-mode patterns.
- `src/test/java/GpuCoverageRegressionGateTest.java` and `GpuHotPathCoverageTargetsTest.java` — hard gate patterns.
- `src/main/java/tuning/workload/CalibrationWorkloads.java` — existing materialization-style workload builders.
- `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java` — layout materialization evidence aggregation.
</canonical_refs>

<current_state>
## Current Codebase Facts

- Layout ABI v2 can describe shape, strides, storage offset, rank, dtype, logical element count, and physical span.
- Metadata-only view propagation already keeps layout views device-owned in supported chains.
- Dense GPU layout materialization exists for Metal through a FLOAT32 native kernel that materializes strided/permuted source layout into a dense destination buffer.
- `MetalDeviceLayoutMaterializer` only admits `FLOAT32` dense targets today.
- `MetalLayoutPolicy` rejects `BROADCAST_ZERO_STRIDE_VIEW` for inputs and outputs, so expand/broadcast repair can still force visible fallback.
- Tensor-array fallback still requires CPU-readable contiguous tensors, so missing layout support can become a CPU materialization boundary unless REQUIRE/gates catch it.
- Coverage summary already counts GPU layout materialization, but representative hard gates for non-contiguous hot paths are not yet Phase 33-specific.
</current_state>

<deferred>
## Deferred Ideas

- Universal strided-native execution for every Metal primitive is out of scope. Start with materialize-to-dense and only admit direct strided consumers when proven.
- CUDA implementation parity is deferred; Phase 33 may keep CUDA layout routing as visible unsupported/capability-gated evidence.
- Full zero-copy output alias closure is Phase 39 unless a small trace field is needed to keep layout gates honest.
</deferred>

---

*Phase: 33-gpu-layout-router-and-strided-materialization*
*Context gathered: 2026-05-02*
