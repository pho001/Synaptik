# Phase 34 Verification: Masked And Causal SDPA

**Phase:** 34 Masked And Causal SDPA
**Status:** Complete
**Verified:** 2026-05-02

## Requirement Mapping

### METALSDPAMASK-01

Metal direct SDPA semantics now distinguish unmasked, external BOOL mask, causal effective mask, external+causal effective mask, unsupported dtype/layout/rank, scale, and dense mask legality before admission.

Evidence:
- `MetalSdpaMaskSemantics`
- `MetalPartitionSupport.sdpaUnsupportedReason(...)`
- `MetalRegionLowererTest`
- `GpuTargetSemanticsContract`

### METALSDPAMASK-02

Supported masked and causal SDPA paths execute through the Metal MPSGraph native SDPA DAG. SDPA input 3 carries the dense effective BOOL mask, and the native shim applies `select(mask, scores, -1.0e9)` before softmax with CPU-compatible mask polarity.

Evidence:
- `MetalMpsCapabilities.externalInputRoleDecision(...)`
- `AcceleratorSubgraphLowerer` generic SDPA DAG input refs
- `synaptik_apple_mps_stub.m` node type `26`
- `MetalMpsFfmBridgeTest` masked/causal/external+causal parity fixtures
- `PreparedExecutionBuildTest`

### METALSDPAMASK-03

Coverage gates now include `masked_sdpa_small` with `METALSDPAMASK` and native Metal expectations. Existing `transformer_block_hot_path` remains causal transformer coverage. Unsupported CUDA and unsupported layout cases remain visible.

Evidence:
- `MaskedSdpaWorkloadSpec`
- `StandardWorkloads.defaultCatalog()`
- `GpuHotPathCoverageTargets`
- `GpuHotPathCoverageTargetsTest`
- `docs/gpu-lowering-coverage.md`
- `docs/metal-backend.md`

## Verification Commands

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest --tests AttentionExecutionTest
./gradlew test --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests GpuTargetSemanticsContractTest
./gradlew metalTest
./gradlew classes
git diff --check
```

All commands passed.

## Residual Risk

- Broadcast/non-dense SDPA masks still reject until the layout router is connected to SDPA mask input repair.
- CUDA direct masked/causal SDPA remains capability-gated.
- Backward masked SDPA completeness remains Phase 38 training/backward scope.

