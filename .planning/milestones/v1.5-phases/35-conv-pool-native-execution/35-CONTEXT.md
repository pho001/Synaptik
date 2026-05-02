# Phase 35: Conv Pool Native Execution - Context

**Gathered:** 2026-05-02
**Status:** Ready for implementation planning
**Source:** Roadmap/requirements context plus live codebase inspection

<domain>
## Phase Boundary

Phase 35 adds selected Metal native/lowered forward execution for conv/pool. The first supported scope should be deliberately narrow and evidence-driven: dense rank-4 NCHW `FLOAT32` forward `CONV2D`, `MAX_POOL2D`, and `AVG_POOL2D` variants whose stride, padding, dilation/group/count semantics are proven against CPU. Unsupported variants must reject with stable reasons instead of silently shortening GPU regions or falling back through tensor-array replay.

Backward conv/pool is not the completion target for this phase. It must remain visible as Phase 38 training/backward scope unless a small diagnostic update is needed to avoid overclaiming forward support.
</domain>

<decisions>
## Implementation Decisions

### Locked Scope

- Public `Tensor` remains logical; native conv/pool residency belongs in compile/prepare/execute runtime state.
- v1.5 remains Metal-first. Shared DAG contracts may be extended, but CUDA conv/pool stays capability-gated until implemented separately.
- Initial native scope is `FLOAT32`, dense rank-4 NCHW.
- `CONV2D`/`CONV2D_GEMM` support must lock stride, symmetric padding, dilation, groups, optional bias, weight shape, and output shape semantics before admission.
- `MAX_POOL2D` and `AVG_POOL2D` support must lock pooling window geometry, padding, stride, max tie behavior, and avg-pool divisor semantics.
- Unsupported conv/pool backward ops remain explicit `CAPABILITY_MISSING`/training-scope blockers.
- Coverage gates must prove native Metal path evidence for supported conv/pool targets and visible blockers for unsupported variants.

### Agent Discretion

- Choose MPSGraph direct conv/pool primitives, a custom Metal kernel, or a lowered primitive DAG, as long as CPU parity and trace evidence exist.
- Choose whether Wave 35-02 implements only groups=1/dilation=1/bias optional Conv2D first, then keeps grouped/dilated variants rejected.
- Choose whether pooling support starts with no-padding square-kernel cases or includes padded cases if native semantics are easy to prove.
</decisions>

<canonical_refs>
## Canonical References

### Planning

- `.planning/PROJECT.md` - v1.5 architecture constraints and Metal-first/CUDA-gated rule.
- `.planning/ROADMAP.md` - Phase 35 goal, success criteria, and dependencies.
- `.planning/REQUIREMENTS.md` - `METALCONVPOOL-01`, `METALCONVPOOL-02`, `METALCONVPOOL-03`.
- `.planning/phases/29-metal-dtype-abi-and-capability-truth/29-VERIFICATION.md` - dtype capability truth.
- `.planning/phases/33-gpu-layout-router-and-strided-materialization/33-VERIFICATION.md` - dense/layout repair foundation.

### Code

- `src/main/java/tensor/ops/conv/TensorConvOps.java` - public NCHW Conv2D construction and backward publication.
- `src/main/java/tensor/ops/pool/TensorPoolOps.java` - public max/avg pool construction and backward publication.
- `src/main/java/tensor/options/Conv2dOptions.java` - stride/padding/dilation/groups contract.
- `src/main/java/tensor/options/Pool2dOptions.java` - kernel/stride/padding/countIncludePad contract.
- `src/main/java/operations/nn/conv/*.java` - Conv2D operation descriptors.
- `src/main/java/operations/nn/pool/*.java` - Pool operation descriptors.
- `src/main/java/backend/cpu/kernels/nn/Conv2dDirectBackend.java` and `Conv2dGemmBackend.java` - CPU correctness behavior.
- `src/main/java/backend/cpu/kernels/nn/Pool2dDirectBackend.java` - CPU pooling tie/divisor behavior.
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` - Metal operation legality.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` - shared DAG construction.
- `src/main/native/apple/synaptik_apple_mps_stub.m` - native MPSGraph switch.

### Tests And Reports

- `src/test/java/Conv2dExecutionTest.java` and `Conv2dLoweringRuleTest.java` - CPU/public Conv2D behavior and GEMM lowering.
- `src/test/java/Pool2dExecutionTest.java` - CPU/public pool behavior.
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` - Metal planner/lowering tests.
- `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` - native bridge parity tests.
- `src/test/java/GpuHotPathCoverageTargetsTest.java`, `GpuCoverageRegressionGateTest.java`, and `GpuCoverageSummaryTest.java` - coverage gates.
- `src/main/java/tuning/workload/StandardWorkloads.java` - conv/pool hot-path workloads.
</canonical_refs>

<current_state>
## Current Codebase Facts

- The GPU lowering coverage matrix lists every targeted conv/pool op as unsupported for Metal and CUDA with `CAPABILITY_MISSING`.
- `MetalPartitionSupport` currently reaches the matrix unsupported reason for `CONV2D`, `CONV2D_GEMM`, `MAX_POOL2D`, and `AVG_POOL2D`.
- Standard workload coverage already has `conv2d_resnet_3x3` and `max_pool2d_small`, but both are visible blockers instead of native gates.
- CPU Conv2D supports direct and GEMM-lowered execution; public rewrite can produce `CONV2D_GEMM`.
- CPU pooling supports max and avg forward/backward. Max pooling records argmax workspace for backward; avg pooling supports `countIncludePad`.
- Existing native DAG has no conv/pool node type yet.
</current_state>

<deferred>
## Deferred Ideas

- Conv/pool backward native execution belongs to Phase 38 unless a small matrix/diagnostic update is needed now.
- CUDA conv/pool parity is deferred.
- Universal grouped/dilated/padded/strided conv and all pooling variants can be staged after the first proven forward scope.
- A vendor router between MPSGraph, custom Metal kernels, and CPU is Phase 39 unless needed for a safe minimum path.
</deferred>

---

*Phase: 35-conv-pool-native-execution*
*Context gathered: 2026-05-02*
