---
phase: 001-accelerator-buffer-layout-abi
reviewed: 2026-04-29T19:35:28Z
depth: standard
files_reviewed: 25
files_reviewed_list:
  - docs/compute-flow.md
  - src/main/java/backend/accelerator/buffer/AcceleratorBufferAccessMode.java
  - src/main/java/backend/accelerator/buffer/AcceleratorBufferInputDecision.java
  - src/main/java/backend/accelerator/buffer/AcceleratorBufferLayout.java
  - src/main/java/backend/accelerator/buffer/AcceleratorBufferLayoutClass.java
  - src/main/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifier.java
  - src/main/java/backend/accelerator/buffer/AcceleratorBufferOutputDecision.java
  - src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java
  - src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java
  - src/main/java/backend/cuda/bridge/CudaGraphBridge.java
  - src/main/java/backend/cuda/exec/PreparedCudaExecutable.java
  - src/main/java/backend/memory/DeviceBufferBinding.java
  - src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java
  - src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java
  - src/main/java/backend/metal/buffer/MetalBufferAllocator.java
  - src/main/java/backend/metal/buffer/MetalBufferBinding.java
  - src/main/java/backend/metal/buffer/MetalDeviceToCpuMaterializer.java
  - src/main/java/backend/metal/exec/PreparedMetalExecutable.java
  - src/test/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifierTest.java
  - src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java
  - src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java
  - src/test/java/backend/metal/buffer/MetalBufferAllocatorTest.java
  - src/test/java/backend/metal/buffer/MetalBufferBindingTest.java
  - src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java
  - src/test/java/graph/execution/ExecutionStateResidencyTest.java
findings:
  critical: 0
  warning: 4
  info: 1
  total: 5
status: issues_found
---

# Phase 001: Code Review Report

**Reviewed:** 2026-04-29T19:35:28Z
**Depth:** standard
**Files Reviewed:** 25
**Status:** issues_found

## Summary

Reviewed the backend-neutral accelerator buffer layout ABI, Metal buffer binding path, CUDA buffer policy taxonomy, and focused tests. The implementation is directionally clean and the scoped Gradle tests pass, but several ABI boundary checks still trust caller-provided metadata too much. The highest-risk gaps are under-validated layout byte lengths, CUDA `REQUIRE` mode not enforcing a real buffer execution path when a bridge advertises support, and Metal native buffer execution accepting bindings without checking node identity.

Verification run:

```bash
./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests graph.execution.ExecutionStateResidencyTest
```

Result: `BUILD SUCCESSFUL` with tasks up to date.

## Warnings

### WR-01: Direct Layout Construction Can Understate Native Buffer Size

**File:** `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayout.java:21`

**Issue:** The compact constructor validates only that `logicalByteLength` is non-negative. Because the record constructor is public, callers can create `new AcceleratorBufferLayout(DataType.FLOAT32, shape, strides, 0, 1024, 4, DENSE_CONTIGUOUS)`. `MetalBufferBinding.available()` and `MetalBufferAllocator.createOutputBinding()` then trust the understated byte length, so a binding can be accepted or allocated too small for the logical element count.

**Fix:** Enforce the derived byte length in the constructor, or remove the constructor-supplied byte length from the ABI and always compute it.

```java
long expectedByteLength = byteLength(dataType, logicalElementCount);
if (logicalByteLength != expectedByteLength) {
    throw new IllegalArgumentException(
            "logicalByteLength " + logicalByteLength
                    + " does not match dtype/element count byte length " + expectedByteLength);
}
```

### WR-02: CUDA REQUIRE Mode Can Still Execute Tensor-List Path

**File:** `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java:84`

**Issue:** `REQUIRE` mode throws only when `!bridge.supportsBufferBindings()`. If a CUDA bridge later returns `true`, this executable still has no buffer request/resolve/execute implementation; it records `UNAVAILABLE`/`NOT_EVALUATED` but proceeds into `bridge.execute(...)` on the tensor-list path. That silently violates the meaning of `AcceleratorBufferBindingMode.REQUIRE`.

**Fix:** Until CUDA has a real buffer execution implementation, fail `REQUIRE` unconditionally in `PreparedCudaExecutable`, or add the full buffer decision and execution path before allowing `supportsBufferBindings()==true`.

```java
if (backendConfig.buffer().bindingMode() == AcceleratorBufferBindingMode.REQUIRE) {
    lastAcceleratorBufferDecision = new AcceleratorBufferDecision(
            ComputeBackend.GPU_CUDA,
            backendConfig.buffer().bindingMode(),
            AcceleratorBufferExecutionPath.UNAVAILABLE,
            false,
            true,
            AcceleratorBufferReasonCode.REQUIRED_BUFFER_EXECUTION_UNAVAILABLE,
            "CUDA prepared executable does not implement buffer binding execution",
            List.of(),
            List.of());
    throw new IllegalStateException("Accelerator buffer path is required for GPU_CUDA but unavailable: "
            + lastAcceleratorBufferDecision.reasonCode() + ": " + lastAcceleratorBufferDecision.reason());
}
```

Also add a test where a fake CUDA bridge overrides `supportsBufferBindings()` to return `true` and asserts tensor-list execution is not entered under `REQUIRE`.

### WR-03: Metal Buffer Bridge Does Not Validate Binding Node IDs

**File:** `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java:571`

**Issue:** `validateBufferBindings` checks counts, availability, dtype, and access, but it does not check that `externalInputs.get(i).nodeId()` matches `executable.externalInputNodeIds().get(i)` or that output binding node ids match `executable.outputNodeIds()`. A direct bridge caller can pass the right number of buffers in the wrong semantic order, and the native graph will consume them without a Java-side ABI failure.

**Fix:** Validate node identity at the same boundary that validates dtype and access.

```java
int expectedInputNodeId = executable.externalInputNodeIds().get(i);
if (binding.nodeId() != expectedInputNodeId) {
    throw new UnsupportedOperationException("Metal buffer input " + i
            + " nodeId " + binding.nodeId()
            + " does not match executable nodeId " + expectedInputNodeId + ".");
}
```

Apply the same check for outputs using `executable.outputNodeIds()`.

### WR-04: Request ABI Still Allows Missing DType Lists

**File:** `src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java:26`

**Issue:** The request now strictly requires layout list sizes to match node id list sizes, but dtype lists still default to empty and consumers treat missing entries as `null` at `MetalAcceleratorBufferBinder.java:127` and `MetalAcceleratorBufferBinder.java:188`. That leaves a compatibility shim in the ABI: malformed requests can skip executable dtype compatibility checks while still passing layout validation.

**Fix:** Make dtype lists as strict as layout lists for any non-empty node id list.

```java
if (externalInputDataTypes.size() != externalInputNodeIds.size()) {
    throw new IllegalArgumentException("externalInputDataTypes size must match externalInputNodeIds size");
}
if (outputDataTypes.size() != outputNodeIds.size()) {
    throw new IllegalArgumentException("outputDataTypes size must match outputNodeIds size");
}
```

## Info

### IN-01: Layout Diagnostic Documentation Mentions Shape But Runtime Reason Omits It

**File:** `docs/compute-flow.md:1886`

**Issue:** The docs say layout fallback diagnostics include `layoutClass`, `shape`, `strides`, and `storageOffset`, but `MetalAcceleratorBufferBinder.unsupportedBufferLayoutReason` emits only `layoutClass`, `storageOffset`, and `strides`.

**Fix:** Either add `shape=` to the runtime diagnostic string or narrow the docs to the fields currently emitted.

---

_Reviewed: 2026-04-29T19:35:28Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
