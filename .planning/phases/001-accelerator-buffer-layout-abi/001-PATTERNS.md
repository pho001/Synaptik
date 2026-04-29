# Phase 1: Accelerator Buffer Layout ABI - Pattern Map

**Mapped:** 2026-04-29
**Files analyzed:** 18 likely new/modified files
**Analogs found:** 18 / 18

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayout.java` | model | transform | `src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java` + `src/main/java/tensor/TensorMetadata.java` | role-match |
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayoutClass.java` | model | transform | `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java` | role-match |
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifier.java` | utility | transform | `src/main/java/tensor/TensorMetadata.java` + `src/test/java/backend/cpu/kernels/layout/StridedLayoutPlanningTest.java` | data-flow-match |
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java` | model | request-response | `src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java` | exact |
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferInputDecision.java` | model | request-response | `src/main/java/backend/accelerator/buffer/AcceleratorBufferInputDecision.java` | exact |
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferOutputDecision.java` | model | request-response | `src/main/java/backend/accelerator/buffer/AcceleratorBufferOutputDecision.java` | exact |
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferDecision.java` | model | request-response | `src/main/java/backend/accelerator/buffer/AcceleratorBufferDecision.java` | exact |
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java` | config | request-response | `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java` | exact |
| `src/main/java/backend/memory/DeviceBufferBinding.java` | model | request-response | `src/main/java/backend/memory/DeviceBufferBinding.java` | exact |
| `src/main/java/backend/metal/buffer/MetalBufferBinding.java` | model | native-buffer I/O | `src/main/java/backend/metal/buffer/MetalBufferBinding.java` | exact |
| `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` | service | request-response | `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` | exact |
| `src/main/java/backend/metal/buffer/MetalBufferAllocator.java` | service | native-buffer I/O | `src/main/java/backend/metal/buffer/MetalBufferAllocator.java` | exact |
| `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` | service | request-response | `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` | exact |
| `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` | service | request-response | `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` | exact |
| `src/test/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifierTest.java` | test | transform | `src/test/java/backend/cpu/kernels/layout/StridedLayoutPlanningTest.java` | role-match |
| `src/test/java/graph/execution/ExecutionStateResidencyTest.java` | test | request-response | `src/test/java/graph/execution/ExecutionStateResidencyTest.java` | exact |
| `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` | test | request-response | `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` | exact |
| `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` | test | request-response | `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` | exact |

## Pattern Assignments

### `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayout.java` (model, transform)

**Analog:** `src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java` and `src/main/java/tensor/TensorMetadata.java`

**Record validation and defensive-copy pattern** (`AcceleratorBufferRequest.java` lines 12-28):
```java
public record AcceleratorBufferRequest(
        ComputeBackend backend,
        long estimatedWork,
        List<Integer> externalInputNodeIds,
        List<DataType> externalInputDataTypes,
        List<Integer> outputNodeIds,
        List<DataType> outputDataTypes,
        boolean runsBackwardPass
) {
    public AcceleratorBufferRequest {
        Objects.requireNonNull(backend, "backend cannot be null");
        estimatedWork = Math.max(0L, estimatedWork);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        externalInputDataTypes = List.copyOf(externalInputDataTypes == null ? List.of() : externalInputDataTypes);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        outputDataTypes = List.copyOf(outputDataTypes == null ? List.of() : outputDataTypes);
    }
}
```

**Array ownership pattern** (`TensorMetadata.java` lines 39-60, 63-80):
```java
public TensorMetadata(int[] shape, int[] strides, int storageOffset, String label, boolean requiresGrad, DataType dataType) {
    int[] normalizedShape = normalizeShape(shape);
    int[] normalizedStrides;
    if (strides == null) {
        throw new IllegalArgumentException("Strides cannot be null.");
    }
    if (strides.length != normalizedShape.length) {
        throw new IllegalArgumentException("Strides length must match shape length.");
    }
    normalizedStrides = strides.clone();

    this.shape = normalizedShape;
    this.strides = normalizedStrides;
    this.storageOffset = normalizeStorageOffset(storageOffset);
    this.contiguous = computeContiguous(this.shape, this.strides);
    this.dataType = dataType == null ? DEFAULT_DATA_TYPE : dataType;
}

public int[] getShape() { return shape.clone(); }
public int[] getStrides() { return strides.clone(); }
public int getStorageOffset() { return storageOffset; }
```

**Apply:** Build the layout record as immutable runtime metadata. Clone `shape` and `strides` in the compact constructor, return clones from accessors, require `DataType`, normalize byte/element counts with explicit validation, and keep native handles out of this record.

---

### `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayoutClass.java` (model, transform)

**Analog:** `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java`

**Stable enum pattern** (lines 3-27):
```java
/**
 * Stable reason codes for accelerator buffer decisions.
 */
public enum AcceleratorBufferReasonCode {
    NOT_EVALUATED,
    BUFFER_BINDINGS_DISABLED,
    BRIDGE_UNAVAILABLE,
    BELOW_MINIMUM_WORK,
    BUFFER_ALLOCATOR_UNAVAILABLE,
    BACKEND_BUFFER_NOT_IMPLEMENTED,
    INPUT_NOT_CPU_CURRENT,
    INPUT_NOT_CONTIGUOUS,
    OUTPUT_LAYOUT_UNSUPPORTED,
    OUTPUT_DTYPE_UNSUPPORTED,
    BUFFER_BINDING_AVAILABLE,
    TENSOR_ARRAY_SELECTED,
    CPU_FALLBACK_SELECTED
}
```

**Apply:** Add layout classes as append-only, stable enum names. Use explicit classes for ABI-03: dense contiguous, zero-offset view, non-zero-offset view, permuted/strided view, broadcast/zero-stride view, and unsupported. Do not rename existing reason codes consumed by tests.

---

### `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifier.java` (utility, transform)

**Analog:** `src/main/java/tensor/TensorMetadata.java` and `src/main/java/tensor/ops/layout/TensorLayoutOps.java`

**Authoritative layout fact pattern** (`TensorMetadata.java` lines 127-145, 148-157):
```java
public boolean isContiguous() {
    return contiguous;
}

public boolean hasZeroStride() {
    for (int stride : strides) {
        if (stride == 0) {
            return true;
        }
    }
    return false;
}

public boolean isBroadcastView() {
    return hasZeroStride();
}

public boolean hasStorageOffset() {
    return storageOffset != 0;
}

private static boolean computeContiguous(int[] shape, int[] strides) {
    int expectedStride = 1;
    for (int i = shape.length - 1; i >= 0; i--) {
        if (strides[i] != expectedStride) {
            return false;
        }
        expectedStride *= shape[i];
    }
    return true;
}
```

**View creation examples to classify** (`TensorLayoutOps.java` lines 55-68, 87-99, 118-128):
```java
Tensor out = input.isContiguous()
        ? TensorPrimitiveBuilder.unaryView(
                input,
                newShape,
                TensorMetadata.computeStrides(newShape),
                input.getStorageOffsetUnsafe(),
                op,
                "reshape",
                input.getDataType()
        )
        : TensorPrimitiveBuilder.unary(input, newShape, op, "reshape", input.getDataType());

int[] targetStrides = LayoutSupport.buildExpandedStrides(input.getShapeUnsafe(), input.getStridesUnsafe(), targetShape);

int[] inStrides = input.getStrides();
for (int i = 0; i < rank; i++) {
    outStrides[i] = inStrides[normalizedAxes[i]];
}
```

**Apply:** Keep classification pure and backend-neutral. Consume dtype, shape, strides, storage offset, and logical element count; return a layout record. Distinguish zero-stride broadcast before generic strided/permuted, and distinguish contiguous-with-offset from dense contiguous.

---

### `src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java` (model, request-response)

**Analog:** current `AcceleratorBufferRequest.java`

**Existing shape to extend** (lines 12-28):
```java
public record AcceleratorBufferRequest(
        ComputeBackend backend,
        long estimatedWork,
        List<Integer> externalInputNodeIds,
        List<DataType> externalInputDataTypes,
        List<Integer> outputNodeIds,
        List<DataType> outputDataTypes,
        boolean runsBackwardPass
) {
    public AcceleratorBufferRequest {
        Objects.requireNonNull(backend, "backend cannot be null");
        estimatedWork = Math.max(0L, estimatedWork);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        externalInputDataTypes = List.copyOf(externalInputDataTypes == null ? List.of() : externalInputDataTypes);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        outputDataTypes = List.copyOf(outputDataTypes == null ? List.of() : outputDataTypes);
    }
}
```

**Apply:** Add `List<AcceleratorBufferLayout>` fields for external inputs and outputs beside existing ids/dtypes. Preserve `List.copyOf(null ? List.of() : value)` semantics. If keeping dtype lists for compatibility, enforce list-size consistency only where current callers can satisfy it.

---

### `src/main/java/backend/accelerator/buffer/AcceleratorBufferInputDecision.java` and `AcceleratorBufferOutputDecision.java` (model, request-response)

**Analog:** current decision records

**Input decision defaulting pattern** (`AcceleratorBufferInputDecision.java` lines 12-22):
```java
public record AcceleratorBufferInputDecision(
        int nodeId,
        boolean preparedInputUsed,
        boolean accepted,
        AcceleratorBufferReasonCode reasonCode,
        String reason
) {
    public AcceleratorBufferInputDecision {
        reasonCode = reasonCode == null ? AcceleratorBufferReasonCode.NOT_EVALUATED : reasonCode;
        reason = reason == null ? "" : reason;
    }
}
```

**Output decision defaulting pattern** (`AcceleratorBufferOutputDecision.java` lines 11-20):
```java
public record AcceleratorBufferOutputDecision(
        int nodeId,
        boolean accepted,
        AcceleratorBufferReasonCode reasonCode,
        String reason
) {
    public AcceleratorBufferOutputDecision {
        reasonCode = reasonCode == null ? AcceleratorBufferReasonCode.NOT_EVALUATED : reasonCode;
        reason = reason == null ? "" : reason;
    }
}
```

**Apply:** Add an optional `AcceleratorBufferLayout layout` to per-input/per-output decisions if planners need traceable layout detail. Preserve null-safe reason defaults and stable codes.

---

### `src/main/java/backend/memory/DeviceBufferBinding.java` (model, request-response)

**Analog:** current `DeviceBufferBinding.java`

**Interface contract pattern** (lines 3-45):
```java
/**
 * Backend-neutral descriptor for a runtime tensor value that has a device-visible buffer.
 *
 * <p>This interface is intentionally smaller than backend-specific buffer handles. Execution state
 * only needs to know which compiled node the binding represents, which backend owns it, whether it
 * is usable, and how many logical bytes it covers. Metal, CUDA, or another accelerator can carry
 * richer native-handle details in their own implementation classes.</p>
 */
public interface DeviceBufferBinding {
    int nodeId();

    String backendId();

    long logicalByteLength();

    boolean available();

    String describe();
}
```

**Runtime validation pattern** (`ExecutionState.java` lines 527-536):
```java
private void validateDeviceBufferBinding(int nodeId, DeviceBufferBinding binding) {
    Objects.requireNonNull(binding, "binding cannot be null");
    if (binding.nodeId() != nodeId) {
        throw new IllegalArgumentException("Device buffer binding nodeId=" + binding.nodeId()
                + " does not match requested nodeId=" + nodeId);
    }
    if (!binding.available()) {
        throw new IllegalArgumentException("Device buffer binding is not available: " + binding.describe());
    }
    residencyForNodeId(nodeId);
}
```

**Apply:** Extend the interface with shared layout/access/native identity methods only. Keep backend-specific handle types out of the interface. Provide either interface methods that implementations must satisfy or default bridging methods only if all existing fake bindings can safely derive them.

---

### `src/main/java/backend/metal/buffer/MetalBufferBinding.java` (model, native-buffer I/O)

**Analog:** current `MetalBufferBinding.java`

**Backend-specific binding pattern** (lines 25-39, 46-57, 65-100):
```java
public record MetalBufferBinding(
        int nodeId,
        DataType dataType,
        int[] shape,
        long elementCount,
        MetalBufferHandle handle,
        MetalBufferAccess access
) implements DeviceBufferBinding {
    public MetalBufferBinding {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        shape = shape == null ? new int[0] : shape.clone();
        elementCount = Math.max(0L, elementCount);
        Objects.requireNonNull(handle, "handle cannot be null");
        access = access == null ? MetalBufferAccess.READ : access;
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    public long logicalByteLength() {
        return elementCount * elementByteSize(dataType);
    }

    @Override
    public String backendId() {
        return ComputeBackend.GPU_METAL.name();
    }

    public boolean bufferCoversLogicalPayload() {
        return handle.available() && handle.byteLength() >= logicalByteLength();
    }

    @Override
    public boolean available() {
        return bufferCoversLogicalPayload();
    }

    public String describe() {
        return "nodeId=" + nodeId
                + ", dtype=" + dataType
                + ", shape=" + Arrays.toString(shape)
                + ", access=" + access
                + ", bytes=" + logicalByteLength()
                + ", handleBytes=" + handle.byteLength();
    }
}
```

**Handle ownership pattern** (`MetalBufferHandle.java` lines 20-40, 52-54):
```java
public record MetalBufferHandle(
        MemorySegment nativeHandle,
        long byteLength,
        String storageMode,
        String owner,
        boolean ownsHandle
) {
    public MetalBufferHandle {
        nativeHandle = nativeHandle == null ? MemorySegment.NULL : nativeHandle;
        byteLength = Math.max(0L, byteLength);
        storageMode = storageMode == null ? "" : storageMode;
        owner = owner == null ? "" : owner;
    }

    public boolean available() {
        return !nativeHandle.equals(MemorySegment.NULL) && byteLength > 0L;
    }

    public boolean hostShared() {
        return "shared".equalsIgnoreCase(storageMode);
    }
}
```

**Apply:** Replace duplicated `dataType`, `shape`, and `elementCount` fields with or delegate them to shared layout metadata. Keep `MetalBufferHandle` and `MetalBufferAccess` backend-owned. Update `describe()` to include layout class, strides, storage offset, logical bytes, and native identity/handle byte length.

---

### `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` (service, request-response)

**Analog:** current `MetalAcceleratorBufferBinder.java`

**Decision pipeline pattern** (lines 41-92):
```java
public AcceleratorBufferDecision decide(
        AcceleratorBufferRequest request,
        ResolvedAcceleratorInputs inputs,
        AcceleratorBufferConfig bufferConfig,
        ExecutionContext context
) {
    AcceleratorBufferConfig config = bufferConfig == null ? AcceleratorBufferConfig.defaults() : bufferConfig;
    AcceleratorBufferBindingMode mode = config.bindingMode();
    if (mode == AcceleratorBufferBindingMode.OFF) {
        return decision(request, config, AcceleratorBufferExecutionPath.TENSOR_ARRAY, false,
                AcceleratorBufferReasonCode.BUFFER_BINDINGS_DISABLED, "buffer bindings disabled", List.of(), List.of());
    }
    if (!bridge.supportsBufferBindings()) {
        return decision(request, config, fallbackPath(mode), false,
                AcceleratorBufferReasonCode.BACKEND_BUFFER_NOT_IMPLEMENTED,
                "bridge does not support buffer bindings", List.of(), List.of());
    }

    List<AcceleratorBufferInputDecision> inputDecisions = inputDecisions(request, inputs, config, context);
    AcceleratorBufferInputDecision rejectedInput = inputDecisions.stream()
            .filter(input -> !input.accepted())
            .findFirst()
            .orElse(null);
    if (rejectedInput != null) {
        return decision(request, config, fallbackPath(mode), false,
                rejectedInput.reasonCode(), rejectedInput.reason(), inputDecisions, List.of());
    }

    List<AcceleratorBufferOutputDecision> outputDecisions = outputDecisions(request, context);
    AcceleratorBufferOutputDecision rejectedOutput = outputDecisions.stream()
            .filter(output -> !output.accepted())
            .findFirst()
            .orElse(null);
    if (rejectedOutput != null) {
        return decision(request, config, fallbackPath(mode), false,
                rejectedOutput.reasonCode(), rejectedOutput.reason(), inputDecisions, outputDecisions);
    }

    return decision(request, config, AcceleratorBufferExecutionPath.BUFFER_BINDING, true,
            AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
            "using native buffer bindings", inputDecisions, outputDecisions);
}
```

**Current layout rejection to replace** (lines 146-150, 185-189, 348-365):
```java
String layoutReason = unsupportedBufferInputLayoutReason(tensor);
if (!layoutReason.isBlank()) {
    out.add(new AcceleratorBufferInputDecision(nodeId, prepared, false,
            AcceleratorBufferReasonCode.INPUT_NOT_CONTIGUOUS,
            "external input nodeId=" + nodeId + " input tensor layout is not " + layoutReason));
    continue;
}

String layoutReason = unsupportedBufferOutputLayoutReason(tensor);
if (!layoutReason.isBlank()) {
    out.add(new AcceleratorBufferOutputDecision(nodeId, false,
            AcceleratorBufferReasonCode.OUTPUT_LAYOUT_UNSUPPORTED,
            "output nodeId=" + nodeId + " output tensor layout is not " + layoutReason));
    continue;
}

private static String unsupportedBufferInputLayoutReason(Tensor tensor) {
    if (tensor == null) {
        return "available";
    }
    if (!tensor.isContiguous() || tensor.hasStorageOffset()) {
        return "contiguous/zero-offset";
    }
    return "";
}
```

**Binding compatibility pattern** (lines 312-338):
```java
private static String incompatibleBindingReason(
        Tensor tensor,
        MetalBufferBinding metalBinding,
        MetalBufferAccess requiredAccess,
        DataType expectedDataType
) {
    if (!metalBinding.available()) {
        return "binding is unavailable: " + metalBinding.describe();
    }
    if (tensor != null && metalBinding.dataType() != tensor.getDataType()) {
        return "binding dtype " + metalBinding.dataType() + " does not match tensor dtype " + tensor.getDataType();
    }
    if (tensor != null && !Arrays.equals(metalBinding.shape(), tensor.getShape())) {
        return "binding shape " + Arrays.toString(metalBinding.shape())
                + " does not match tensor shape " + Arrays.toString(tensor.getShape());
    }
    if (!accessCompatible(metalBinding.access(), requiredAccess)) {
        return "binding access " + metalBinding.access() + " is incompatible with required " + requiredAccess;
    }
    return "";
}
```

**Apply:** Keep the staged preflight shape. Generate layouts before backend capability decisions. Replace generic contiguous/zero-offset reason strings with layout class and stable ABI-04 reason codes. Compare existing `MetalBufferBinding.layout()` against runtime tensor/request layout rather than comparing shape/count fields independently.

---

### `src/main/java/backend/metal/buffer/MetalBufferAllocator.java` (service, native-buffer I/O)

**Analog:** current `MetalBufferAllocator.java`

**Allocator availability and native seam pattern** (lines 24-55):
```java
public final class MetalBufferAllocator {
    public interface NativeAccess {
        MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes);

        void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength);

        void destroyBuffer(MetalBufferHandle handle);
    }

    public static MetalBufferAllocator available(NativeAccess nativeAccess) {
        return new MetalBufferAllocator(true, "", Objects.requireNonNull(nativeAccess, "nativeAccess cannot be null"));
    }

    public static MetalBufferAllocator unavailable(String reason) {
        return new MetalBufferAllocator(false, reason, null);
    }
}
```

**Input/output binding construction pattern** (lines 92-107, 144-156, 222-240):
```java
public MetalBufferBinding createInputBinding(int nodeId, Tensor tensor) {
    ensureAvailable();
    validateCommonInput(tensor);
    if (tensor.getDataType() != DataType.FLOAT32) {
        throw new UnsupportedOperationException("Metal buffer inputs support FLOAT32 data buffers only; got " + tensor.getDataType());
    }
    float[] data = tensor.getFloat32Data();
    try (Arena arena = Arena.ofConfined()) {
        MemorySegment initialData = arena.allocateFrom(JAVA_FLOAT, data);
        long bytes = (long) data.length * Float.BYTES;
        MetalBufferHandle handle = nativeAccess.createBuffer(bytes, STORAGE_MODE_SHARED, initialData, bytes);
        return binding(nodeId, tensor, handle, MetalBufferAccess.READ);
    }
}

public MetalBufferBinding createOutputBinding(int nodeId, DataType dtype, int[] shape, long elementCount) {
    ensureAvailable();
    long bytes = Math.multiplyExact(elementCount, (long) Float.BYTES);
    MetalBufferHandle handle = nativeAccess.createBuffer(bytes, STORAGE_MODE_SHARED, MemorySegment.NULL, 0L);
    return new MetalBufferBinding(nodeId, dtype, safeShape, elementCount, handle, MetalBufferAccess.READ_WRITE);
}

private static MetalBufferBinding binding(int nodeId, Tensor tensor, MetalBufferHandle handle, MetalBufferAccess access) {
    return new MetalBufferBinding(
            nodeId,
            tensor.getDataType(),
            tensor.getShape(),
            tensor.getFlatDataSize(),
            handle,
            access
    );
}
```

**Apply:** Route binding creation through `AcceleratorBufferLayout.fromTensor(...)` or classifier output. Keep current conservative input validation for native payload upload until Phase 2. Use layout logical byte length for native allocation size; use checked arithmetic as the existing output path does.

---

### `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` (service, request-response)

**Analog:** current `PreparedMetalExecutable.java`

**Buffer-first execution pattern** (lines 120-145):
```java
ResolvedAcceleratorInputs resolvedInputs = AcceleratorPreparedInputResolver.resolve(
        cpuFallbackSteps,
        bridgeExecutable.externalInputNodeIds(),
        context
);
AcceleratorBufferRequest request = bufferRequest(context);
AcceleratorBufferDecision decision = bufferBinder.decide(
        request,
        resolvedInputs,
        backendConfig.buffer(),
        context
);
publishDecision(decision);
requireBufferOrThrow(decision);

if (decision.path() == AcceleratorBufferExecutionPath.BUFFER_BINDING) {
    AcceleratorBufferBindings<MetalBufferBinding> bindings = bufferBinder.resolve(request, resolvedInputs, decision, context);
    lastExecutionStats = bridge.executeBuffers(
            bridgeContext,
            bridgeExecutable,
            bindings.inputs(),
            bindings.outputs()
    );
    if (!lastExecutionStats.usedCpuFallback()) {
        markBufferOutputsCurrent(context, bindings.outputs());
    }
}
```

**Request construction pattern to extend** (lines 232-241):
```java
private AcceleratorBufferRequest bufferRequest(ExecutionContext context) {
    return new AcceleratorBufferRequest(
            ComputeBackend.GPU_METAL,
            plan.estimatedWork(),
            bridgeExecutable.externalInputNodeIds(),
            bridgeExecutable.externalInputDataTypes(),
            bridgeExecutable.outputNodeIds(),
            bridgeExecutable.outputDataTypes(),
            context != null && context.runsBackwardPass()
    );
}
```

**Output promotion pattern** (lines 345-377):
```java
private static void markBufferOutputsCurrent(ExecutionContext context, List<MetalBufferBinding> outputBindings) {
    if (context == null || outputBindings == null || outputBindings.isEmpty()) {
        return;
    }
    for (MetalBufferBinding binding : outputBindings) {
        MetalBufferBinding activeBinding = readableAfterWrite(binding);
        context.attachDeviceBufferBinding(
                activeBinding.nodeId(),
                activeBinding,
                residencyForOutputBinding(activeBinding),
                "metal buffer binding output"
        );
    }
}

private static StorageResidency residencyForOutputBinding(MetalBufferBinding binding) {
    return StorageResidency.DEVICE_OWNED;
}
```

**Apply:** Build input/output layout lists in `bufferRequest(...)` using runtime tensors from `ExecutionContext`. Preserve buffer-first execution, conservative tensor-array fallback, `publishDecision(...)`, and only promote reserved outputs after successful native buffer execution.

---

### `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` (service, request-response)

**Analog:** current `PreparedCudaExecutable.java`

**CUDA required-buffer policy pattern** (lines 83-99):
```java
if (backendConfig.buffer().bindingMode() == AcceleratorBufferBindingMode.REQUIRE
        && !bridge.supportsBufferBindings()) {
    lastAcceleratorBufferDecision = new AcceleratorBufferDecision(
            ComputeBackend.GPU_CUDA,
            backendConfig.buffer().bindingMode(),
            AcceleratorBufferExecutionPath.UNAVAILABLE,
            false,
            true,
            AcceleratorBufferReasonCode.BACKEND_BUFFER_NOT_IMPLEMENTED,
            "CUDA bridge does not support buffer bindings",
            List.of(),
            List.of()
    );
    throw new IllegalStateException("Accelerator buffer path is required for GPU_CUDA but unavailable: "
            + lastAcceleratorBufferDecision.reasonCode() + ": " + lastAcceleratorBufferDecision.reason());
}
```

**CUDA seam pattern** (`CudaGraphBridge.java` lines 48-56):
```java
/**
 * Returns whether this CUDA bridge can execute through explicit native buffer bindings.
 *
 * <p>The default is {@code false}. Future CUDA native-buffer implementations should override this
 * only after they own a concrete device pointer/graph-buffer lifetime contract.</p>
 */
default boolean supportsBufferBindings() {
    return false;
}
```

**Apply:** Keep CUDA native buffers unavailable in Phase 1. Change only the shared reason code/details needed for ABI-04, such as a required-but-unavailable code, while preserving explicit `REQUIRE` failure before tensor-list execution.

---

### `src/test/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifierTest.java` (test, transform)

**Analog:** `src/test/java/backend/cpu/kernels/layout/StridedLayoutPlanningTest.java`

**Layout fixture/assertion pattern** (lines 21-37, 100-115):
```java
@Test
public void resolvesSingleNonContiguousBinaryInputToSelectiveMaterialization() {
    Tensor left = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, new int[]{1, 2}, null, "left_noncontig", DataType.FLOAT64);
    Tensor right = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "right_contig", DataType.FLOAT64);
    Tensor out = new Tensor(new int[]{2, 2}, List.of(left, right), new add(), "out", DataType.FLOAT64);

    StridedLayoutDecision decision = StridedPathEligibility.resolve(
            new add(),
            List.of(left, right),
            out,
            DataType.FLOAT64,
            planner
    );

    assertEquals(StridedLayoutDecision.MATERIALIZE_INPUT_0, decision);
}

@Test
public void rowBroadcastStrideClassDoesNotUseCheapMaterializationThreshold() {
    Tensor left = new Tensor(new double[]{1, 2, 0, 0}, new int[]{2, 2}, new int[]{1, 0}, null, "left_row_broadcast", DataType.FLOAT64);

    StridedLayoutDecision decision = StridedPathEligibility.resolve(...);

    assertEquals(StridedLayoutDecision.KEEP_STRIDED, decision);
}
```

**Apply:** Use small direct tensors and assert exact enum classes. Cover dense contiguous, zero-offset reshape/view, non-zero-offset view, permuted/strided view, broadcast/zero-stride view, invalid shape/stride mismatch, dtype, logical byte length, and defensive copies.

---

### Existing test fake binding updates

**Analogs:** `ExecutionStateResidencyTest.java`, `PreparedMetalExecutableBufferBindingTest.java`, `MetalBufferBindingTest.java`, `MetalBufferAllocatorTest.java`

**Fake `DeviceBufferBinding` pattern to update** (`ExecutionStateResidencyTest.java` lines 280-289):
```java
private record FakeDeviceBufferBinding(
        int nodeId,
        String backendId,
        long logicalByteLength,
        boolean available
) implements DeviceBufferBinding {
    @Override
    public String describe() {
        return "fake nodeId=" + nodeId + ", backend=" + backendId + ", bytes=" + logicalByteLength;
    }
}
```

**Non-Metal binding pattern to update** (`PreparedMetalExecutableBufferBindingTest.java` lines 645-659):
```java
private record NonMetalBinding(int nodeId, long logicalByteLength) implements DeviceBufferBinding {
    @Override
    public String backendId() {
        return "GPU_TEST";
    }

    @Override
    public boolean available() {
        return true;
    }

    @Override
    public String describe() {
        return "non-metal nodeId=" + nodeId;
    }
}
```

**Metal binding unit pattern** (`MetalBufferBindingTest.java` lines 13-29, 45-55):
```java
@Test
void bindingReportsLogicalPayloadCoverage() {
    MetalBufferHandle handle = new MetalBufferHandle(MemorySegment.ofAddress(1), 16, "shared", "test", false);
    MetalBufferBinding binding = new MetalBufferBinding(
            7,
            DataType.FLOAT32,
            new int[]{2, 2},
            4,
            handle,
            MetalBufferAccess.READ
    );

    assertTrue(binding.bufferCoversLogicalPayload());
    assertTrue(binding.available());
    assertTrue(binding.backendId().equals("GPU_METAL"));
    assertTrue(binding.describe().contains("nodeId=7"));
    assertTrue(binding.describe().contains("bytes=16"));
}
```

**Allocator rejection test pattern** (`MetalBufferAllocatorTest.java` lines 16-49):
```java
@Test
void readToCpuRejectsNonContiguousDestinationBeforeNativeRead() {
    AtomicInteger reads = new AtomicInteger();
    MetalBufferAllocator allocator = MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
        @Override
        public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
            reads.incrementAndGet();
        }
    });
    MetalBufferBinding binding = new MetalBufferBinding(...);
    Tensor base = new Tensor(new float[]{0f, 0f, 0f, 0f}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
    Tensor destination = base.permute(1, 0);

    assertThrows(
            UnsupportedOperationException.class,
            () -> allocator.readToCpu(binding, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS)
    );
    assertEquals(0, reads.get());
}
```

**CUDA required-buffer test pattern** (`PreparedCudaExecutableBufferPolicyTest.java` lines 27-45):
```java
@Test
void requireBufferModeFailsBecauseCudaBufferBindingsAreNotImplementedYet() {
    PreparedCudaExecutable executable = new PreparedCudaExecutable(
            dag(),
            LoweringFamily.CUDA_GRAPH_REGION,
            new FakeCudaBridge(),
            List.of(),
            AcceleratorBackendConfig.defaults().withBuffer(
                    new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
            )
    );

    IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(null));

    assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
    assertEquals(AcceleratorBufferBindingMode.REQUIRE, executable.lastAcceleratorBufferDecision().mode());
    assertEquals("BACKEND_BUFFER_NOT_IMPLEMENTED", executable.lastAcceleratorBufferDecision().reasonCode().name());
}
```

**Apply:** Update fake bindings to satisfy new interface methods with a simple dense FLOAT32 layout helper. Update Metal tests to construct `MetalBufferBinding` with shared layout metadata. Keep tests assertion-focused and use exact reason-code names where ABI stability matters.

## Shared Patterns

### Backend-Neutral Value Records

**Source:** `AcceleratorBufferRequest.java`, `AcceleratorBufferDecision.java`

**Apply to:** `AcceleratorBufferLayout`, request/decision updates
```java
Objects.requireNonNull(backend, "backend cannot be null");
estimatedWork = Math.max(0L, estimatedWork);
externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
reasonCode = reasonCode == null ? AcceleratorBufferReasonCode.NOT_EVALUATED : reasonCode;
reason = reason == null ? "" : reason;
```

### Layout Source Of Truth

**Source:** `TensorMetadata.java`, `Tensor.java`, `CompiledNode.java`

**Apply to:** classifier, request construction, Metal binding compatibility
```java
tensor.getShape();
tensor.getStrides();
tensor.getStorageOffsetUnsafe();
tensor.getDataType();
tensor.isContiguous();
tensor.hasStorageOffset();
tensor.getFlatDataSize();
```

Compiled snapshots already capture the same facts (`CompiledNode.java` lines 126-134):
```java
tensor.getShapeUnsafe(),
tensor.getStridesUnsafe(),
tensor.getStorageOffsetUnsafe(),
tensor.getDataType(),
tensor.isContiguous(),
tensor.hasStorageOffset(),
tensor.getFlatDataSize(),
```

### Runtime Residency Safety

**Source:** `ExecutionState.java`

**Apply to:** `DeviceBufferBinding`, Metal binding promotion, fake binding tests
```java
public void reserveDeviceBufferBinding(int nodeId, DeviceBufferBinding binding) {
    validateDeviceBufferBinding(nodeId, binding);
    reservedDeviceBufferBindingByNodeId.put(nodeId, binding);
}

public DeviceBufferBinding writableDeviceBufferBindingForNodeId(int nodeId) {
    residencyForNodeId(nodeId);
    DeviceBufferBinding reserved = reservedDeviceBufferBindingByNodeId.get(nodeId);
    return reserved == null ? deviceBufferBindingByNodeId.get(nodeId) : reserved;
}
```

### Conservative Metal Phase Boundary

**Source:** `MetalAcceleratorBufferBinder.java`, `PreparedMetalExecutable.java`

**Apply to:** layout reason codes and Metal binder behavior
```java
if (rejectedOutput != null) {
    return decision(request, config, fallbackPath(mode), false,
            rejectedOutput.reasonCode(), rejectedOutput.reason(), inputDecisions, outputDecisions);
}

private static void requireBufferOrThrow(AcceleratorBufferDecision decision) {
    if (decision != null && decision.required() && decision.path() != AcceleratorBufferExecutionPath.BUFFER_BINDING) {
        throw new IllegalStateException("Accelerator buffer path is required for "
                + decision.backend() + " but unavailable: "
                + decision.reasonCode() + ": " + decision.reason());
    }
}
```

### Native Handle Ownership

**Source:** `MetalBufferHandle.java`, `MetalBufferResource.java`

**Apply to:** Metal binding updates only
```java
public boolean available() {
    return !nativeHandle.equals(MemorySegment.NULL) && byteLength > 0L;
}

public void close() {
    if (closed) {
        return;
    }
    allocator.destroy(handle);
    closed = true;
}
```

## No Analog Found

No file lacks a usable analog. The only non-exact area is the new backend-neutral layout classifier; use `TensorMetadata` for layout facts and `StridedLayoutPlanningTest` for direct tensor classification test style.

## Metadata

**Analog search scope:** `src/main/java/backend/accelerator`, `src/main/java/backend/memory`, `src/main/java/backend/metal`, `src/main/java/backend/cuda`, `src/main/java/graph/execution`, `src/main/java/tensor`, `src/test/java/backend`, `src/test/java/graph`
**Files scanned:** 641 source/test paths from targeted `rg --files`, plus project codebase maps and phase artifacts
**Pattern extraction date:** 2026-04-29
