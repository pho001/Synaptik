# Plan 39-02 Summary: Custom Kernel Integration Point And MPSGraph Routing

**Status:** Complete
**Completed:** 2026-05-02 14:54 CEST
**Requirement Coverage:** METALROUTER-01

## What Changed

- Added the backend-internal custom Metal kernel SPI under `src/main/java/backend/metal/kernel/`:
  - `MetalCustomKernelBridge`
  - `MetalCustomKernelExecutable`
  - `MetalCustomKernelCapabilities`
  - `UnavailableMetalCustomKernelBridge`
- Added `MetalCustomKernelRouteAdapter` to convert custom-kernel capability/executable state into router evidence.
- Kept `CUSTOM_KERNEL` rejected by default with stable `CUSTOM_KERNEL_UNAVAILABLE` evidence.
- Extended `MetalRouteDecision` and Metal trace attrs with rejected route reason codes and reason text.
- Kept existing supported Metal execution on `MPS_GRAPH`; buffer-binding execution remains unchanged.

## Important Semantics

- The custom-kernel route is an integration seam only. It does not claim native execution until a concrete kernel bridge and parity tests exist.
- `CUSTOM_KERNEL_UNAVAILABLE` is now machine-readable route evidence, not only free-form detail text.
- The seam consumes lowered Metal region metadata and does not depend on public Tensor device-residency APIs.

## Verification

Passed:

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest
./gradlew test --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest
./gradlew classes
git diff --check
```
