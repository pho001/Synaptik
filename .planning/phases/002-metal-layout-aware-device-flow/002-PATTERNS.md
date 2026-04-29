# Phase 2: Metal Layout-Aware Device Flow - Pattern Map

**Mapped:** 2026-04-29
**Files analyzed:** 13 likely modified files
**Analogs found:** 13 / 13

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` | service | request-response | same file | exact |
| `src/main/java/backend/metal/buffer/MetalBufferAllocator.java` | service | native-buffer I/O | same file | exact |
| `src/main/java/backend/metal/buffer/MetalDeviceToCpuMaterializer.java` | service | native-buffer I/O | same file | exact |
| `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` | service | request-response | same file | exact |
| `src/main/java/backend/metal/bridge/MetalMpsGraphBridge.java` | interface | native capability | same file | exact |
| `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` | service | FFM/native bridge | same file | exact |
| `src/main/native/apple/synaptik_apple_mps_stub.m` | native bridge | Metal execution | same file | exact |
| `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` | policy | planner legality | same file | exact |
| `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` | lowerer | graph-to-DAG | same file | exact |
| `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` | test | request-response | same file | exact |
| `src/test/java/backend/metal/buffer/MetalBufferAllocatorTest.java` | test | native-buffer I/O | same file | exact |
| `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` | test | FFM/native bridge | same file | exact |
| `src/test/java/backend/metal/MetalBufferTraceSmokeTest.java` | test | traced execution | same file | exact |

## Pattern Assignments

### `MetalAcceleratorBufferBinder.java` (service, request-response)

**Existing decision flow to preserve:**

```java
List<AcceleratorBufferInputDecision> inputDecisions = inputDecisions(request, inputs, config, context);
AcceleratorBufferInputDecision rejectedInput = inputDecisions.stream()
        .filter(input -> !input.accepted())
        .findFirst()
        .orElse(null);
if (rejectedInput != null) {
    return decision(request, config, fallbackPath(mode), false,
            rejectedInput.reasonCode(), rejectedInput.reason(), inputDecisions, List.of());
}
```

**Apply:** Keep input preflight before output allocation. Replace the dense-only `unsupportedBufferLayoutReason(...)` with a Metal layout policy that returns explicit accept/transform/reject decisions. Do not allocate outputs if an input preflight failed.

### `MetalBufferAllocator.java` (service, native-buffer I/O)

**Existing allocation pattern:**

```java
MetalBufferHandle handle = nativeAccess.createBuffer(layout.logicalByteLength(), STORAGE_MODE_SHARED, MemorySegment.NULL, 0L);
return new MetalBufferBinding(nodeId, layout, handle, MetalBufferAccess.READ_WRITE);
```

**Apply:** Keep allocator run-scoped and resource-owned by execution. If Phase 2 adds device-contiguous transforms, use new explicit methods with names that state whether the returned binding is dense physical storage or a logical layout view. Do not overload existing dense methods with ambiguous behavior.

### `MetalDeviceToCpuMaterializer.java` (service, native-buffer I/O)

**Existing support gate:**

```java
return metalBinding.available()
        && metalBinding.layout().dataType() == DataType.FLOAT32
        && target.getDataType() == DataType.FLOAT32
        && Arrays.equals(metalBinding.layout().shape(), targetLayout.shape())
        && Arrays.equals(metalBinding.layout().strides(), targetLayout.strides())
        && metalBinding.layout().storageOffset() == targetLayout.storageOffset()
        && metalBinding.layout().logicalElementCount() == targetLayout.logicalElementCount();
```

**Apply:** Keep `supports(...)` truthful. If readback cannot scatter into a non-contiguous target, return false. If scatter support is added, test it with graph output and public data access reasons.

### `PreparedMetalExecutable.java` (service, request-response)

**Existing buffer-path selection pattern:**

```java
if (decision.path() == AcceleratorBufferExecutionPath.BUFFER_BINDING) {
    try {
        AcceleratorBufferBindings<MetalBufferBinding> bindings = bufferBinder.resolve(request, resolvedInputs, decision, context);
        lastExecutionStats = bridge.executeBuffers(bridgeContext, bridgeExecutable, bindings.inputs(), bindings.outputs());
        if (!lastExecutionStats.usedCpuFallback()) {
            markBufferOutputsCurrent(context, bindings.outputs());
        }
    } catch (RuntimeException ex) {
        ...
    }
    return;
}
```

**Apply:** Only mark device outputs current after native buffer execution succeeds. If a layout transform or materializer cannot support the output, fail preflight or fall back before publishing a device-owned binding.

### `MetalMpsFfmBridge.java` (service, FFM/native bridge)

**Existing optional-symbol pattern:**

```java
MethodHandle executePartitionBuffersFn = optionalHandle(
        linker,
        lookup,
        "synaptik_apple_mps_execute_partition_f32_buffers",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS)
);
```

**Apply:** Any native layout-aware ABI should be added as a new optional symbol or explicit capability, not by changing the semantics of the existing buffer execution symbol silently. Tests should verify missing symbols fall back or throw under `REQUIRE`.

### `synaptik_apple_mps_stub.m` (native bridge, Metal execution)

**Existing shape-only MPSGraph tensor data pattern:**

```objective-c
MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:box.buffer
                                                                   shape:shape
                                                                dataType:dataType];
```

**Apply:** Do not pretend this path supports arbitrary strides or storage offsets. If logical views are represented natively, the ABI must pass metadata and tests must prove MPSGraph consumes it correctly. Otherwise use dense output buffers and Java-side policy to decide legal transforms.

### `PreparedMetalExecutableBufferBindingTest.java` (test, request-response)

**Existing fake-bridge pattern:**

```java
assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
assertEquals("using native buffer bindings", executable.lastBufferBindingDecision());
assertTrue(fixture.state().cpuMaterializationTraces().isEmpty());
```

**Apply:** Extend this suite first. Fake bridge tests should assert execution path, reason code, buffer allocation count, residency, and CPU materialization traces for every new layout policy case.

### `MetalBufferTraceSmokeTest.java` (test, traced execution)

**Existing trace assertion pattern:**

```java
assertEquals("BUFFER_BINDING", attrs.get("metalExecutionPath"));
assertEquals("DEVICE_OWNED", attrs.get("storageResidency"));
assertEquals(false, attrs.get("storageCpuCurrent"));
assertEquals(true, attrs.get("storageDeviceCurrent"));
```

**Apply:** Add layout-aware attributes only if they are actually emitted by trace metadata. Prefer assertions on existing `metalExecutionPath`, `storageResidency`, and CPU materialization traces unless Phase 2 explicitly adds new trace fields.

## Constraints For Planning

- Keep public `Tensor` API logical.
- Keep shared accelerator abstractions backend-neutral; Metal-specific policy belongs under `backend.metal`.
- Do not change CPU hot-path kernels to satisfy Metal layout behavior.
- Keep fallback visible in `AcceleratorBufferDecision`, bridge stats, and trace metadata.
- Do not commit generated native binaries, local benchmark profiles, or `.planning/tmp/`.
