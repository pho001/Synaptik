---
phase: 44
type: research
status: complete
requirements:
  - METALKERNEL-01
  - METALKERNEL-02
  - METALKERNEL-03
---

# Phase 44 Research: Custom Metal Kernel Execution Route

## Phase Goal

Turn the current custom Metal kernel seam from a visible unavailable route into a real scoped native execution route.

## Current Evidence

- Phase 39 added route/copy reporting and the backend-internal custom-kernel SPI.
- `MetalExecutionRouter` currently considers `CUSTOM_KERNEL` but always rejects it unless a custom executable is available.
- `MetalCustomKernelBridge`, `MetalCustomKernelCapabilities`, and `MetalCustomKernelExecutable` exist as Java-side route metadata.
- `PreparedMetalExecutable` prepares the custom executable but execution always uses MPSGraph buffer binding, tensor-array bridge, or CPU fallback.
- Native Metal code already builds runtime Metal compute pipelines inside `src/main/native/apple/synaptik_apple_mps_stub.m` for layout materialization; this is the natural place to add a first scoped custom kernel.

## Planning Direction

Phase 44 should implement the smallest truthful custom-kernel route:

1. Define a narrow custom-kernel candidate subset.
2. Add Java route metadata that proves the candidate is selected only for that subset.
3. Add native bridge execution for one simple buffer-bound operation family.
4. Execute that route through `PreparedMetalExecutable` without public device tensor APIs.
5. Keep MPSGraph as default for broader supported graph primitives.

The safest first candidate is a dense `FLOAT32` unary elementwise kernel such as `RELU` or a single-node elementwise chain. It has a simple buffer contract, deterministic CPU parity, no layout/view ambiguity when restricted to dense inputs/outputs, and no dependency on MPSGraph output-copy behavior.

## Verification Targets

- Unit tests for route selection and rejection:
  - custom bridge unavailable -> MPSGraph selected or fallback remains visible;
  - custom executable available for scoped candidate -> `CUSTOM_KERNEL` selected;
  - unsupported dtype/layout/multi-op candidate -> custom route rejected.
- Native Metal capability-gated test proving the custom kernel writes expected output through buffer bindings.
- Trace/report tests proving `metalExecutionRoute=CUSTOM_KERNEL`, rejected alternatives, and no tensor-array/CPU replay for the supported path.
- Docs explaining custom-kernel scope and fallback behavior.

