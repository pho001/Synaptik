# 32-01 Summary: INT32 ABI Residency And Planner Legality

**Completed:** 2026-05-02
**Requirements:** METALINTIDX-01

## Delivered

- Allowed Metal `INT32` tensors as external/index input dtype metadata without enabling generic `INT32` compute or output support.
- Added role-specific Metal capability checks so `GATHER` and `TAKE_ALONG_AXIS` can accept `INT32` at input index 1 while rejecting `INT32` in non-index roles.
- Extended Metal buffer allocation and buffer-binding validation to carry `INT32` input buffers with correct byte sizing.
- Updated dtype residency policy so Metal can represent internal/input `INT32` residency evidence while compute/output remains rejected.
- Added tests for Metal capability truth, dtype residency policy, INT32 input buffer allocation, and bridge binding validation.

## Verification

```bash
./gradlew test --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest
git diff --check
```

## Notes

- `GATHER` and `TAKE_ALONG_AXIS` lowering/native execution remains Phase 32-02.
- `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` remain Phase 36 scope.
- Local profile artifacts under `profiles/platform/...` remain unstaged.
