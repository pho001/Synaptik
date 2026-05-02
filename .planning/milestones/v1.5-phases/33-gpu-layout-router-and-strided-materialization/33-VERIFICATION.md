# Phase 33 Verification: GPU Layout Router And Strided Materialization

**Status:** Complete
**Date:** 2026-05-02

## Requirement Mapping

- `METALLAYOUT-01`: Completed by router route kinds/reason codes for metadata-only views, dense materialization, broadcast materialization, strided-native unsupported routing, and stable rejection reasons.
- `METALLAYOUT-02`: Completed by Metal FLOAT32 dense/broadcast GPU-side materialization, physical-span binding validation, prepared execution trace integration, and native parity tests.
- `METALLAYOUT-03`: Completed by `layout_broadcast_repair_small`, hot-path expectations, and regression gate checks for layout materialization evidence and `CPU_CONSUMER` rejection.

## Evidence

- Code:
  - `AcceleratorLayoutTransformPlanner`
  - `AcceleratorLayoutTransformKind`
  - `AcceleratorBufferReasonCode`
  - `MetalDeviceLayoutMaterializer`
  - `MetalBufferBinding`
  - `PreparedExecution`
  - `GpuCoverageRegressionGate`
  - `LayoutRepairWorkloadSpec`
- Tests:
  - `AcceleratorLayoutTransformPlannerTest`
  - `MetalBufferBindingTest`
  - `MetalDeviceLayoutMaterializerTest`
  - `MetalMpsFfmBridgeTest`
  - `MetalLayoutAwareDeviceFlowTest`
  - `GpuCoverageRegressionGateTest`
  - `GpuHotPathCoverageTargetsTest`
  - `StandardWorkloadsTest`
- Docs:
  - `docs/metal-backend.md`
  - `docs/gpu-lowering-coverage.md`
  - `docs/troubleshooting.md`

## Residual Limitations

- Metal layout materialization remains scoped to FLOAT32.
- Direct `STRIDED_NATIVE_COMPUTE` is named but unsupported until specific consumer primitives prove direct strided execution.
- CUDA remains capability-gated for the new Phase 33 hot-path target.
