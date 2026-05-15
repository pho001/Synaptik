package backend.metal.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferLayoutClass;
import backend.accelerator.buffer.AcceleratorLayoutAbiV2Descriptor;
import backend.memory.CpuMaterializationReason;
import backend.memory.CpuMaterializationResult;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Run-scoped allocator and materializer for native Metal buffer bindings.
 *
 * <p>The allocator is created from the active Metal bridge context. It owns no global state: each allocated
 * {@link MetalBufferHandle} is returned to execution code, which registers the handle as a run resource and
 * eventually calls {@link #destroy(MetalBufferHandle)}. The allocator supports shared F32/BF16 compute buffers,
 * BOOL predicate buffers, INT32 index buffers, scoped INT64 index outputs, and dtype-matched CPU materialization.</p>
 */
public final class MetalBufferAllocator {
    /**
     * Native buffer calls supplied by the active bridge implementation.
     */
    public interface NativeAccess {
        MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes);

        void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength);

        void destroyBuffer(MetalBufferHandle handle);
    }

    private static final int STORAGE_MODE_SHARED = 1;
    private final boolean available;
    private final String unavailableReason;
    private final NativeAccess nativeAccess;

    private MetalBufferAllocator(boolean available, String unavailableReason, NativeAccess nativeAccess) {
        this.available = available;
        this.unavailableReason = unavailableReason == null ? "" : unavailableReason;
        this.nativeAccess = nativeAccess;
    }

    /**
     * Creates an available allocator backed by native bridge calls.
     *
     * @param nativeAccess native buffer API
     * @return allocator
     */
    public static MetalBufferAllocator available(NativeAccess nativeAccess) {
        return new MetalBufferAllocator(true, "", Objects.requireNonNull(nativeAccess, "nativeAccess cannot be null"));
    }

    /**
     * Creates an unavailable allocator that reports a stable reason.
     *
     * @param reason unavailable reason
     * @return unavailable allocator
     */
    public static MetalBufferAllocator unavailable(String reason) {
        return new MetalBufferAllocator(false, reason, null);
    }

    /**
     * Returns whether native Metal buffer allocation can be attempted.
     *
     * @return true when backed by available native calls
     */
    public boolean available() {
        return available;
    }

    /**
     * Returns why allocation is unavailable.
     *
     * @return diagnostic reason, or an empty string when available
     */
    public String unavailableReason() {
        return unavailableReason;
    }

    /**
     * Creates a shared numeric input binding initialized from the tensor's CPU storage.
     *
     * @param nodeId compiled node id represented by the tensor
     * @param tensor runtime tensor
     * @return initialized Metal input binding
     */
    public MetalBufferBinding createInputBinding(int nodeId, Tensor tensor) {
        ensureAvailable();
        validateCommonInput(tensor);
        return switch (tensor.getDataType()) {
            case FLOAT32 -> {
                float[] data = tensor.getFloat32Data();
                if (data == null) {
                    throw new UnsupportedOperationException("Metal FLOAT32 input tensor has no direct float[] storage.");
                }
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment initialData = arena.allocateFrom(JAVA_FLOAT, data);
                    long bytes = (long) data.length * Float.BYTES;
                    MetalBufferHandle handle = nativeAccess.createBuffer(bytes, STORAGE_MODE_SHARED, initialData, bytes);
                    yield binding(nodeId, tensor, handle, MetalBufferAccess.READ);
                }
            }
            case BFLOAT16 -> {
                short[] data = tensor.getBFloat16Data();
                if (data == null) {
                    throw new UnsupportedOperationException("Metal BFLOAT16 input tensor has no direct short[] storage.");
                }
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment initialData = arena.allocateFrom(JAVA_SHORT, data);
                    long bytes = (long) data.length * Short.BYTES;
                    MetalBufferHandle handle = nativeAccess.createBuffer(bytes, STORAGE_MODE_SHARED, initialData, bytes);
                    yield binding(nodeId, tensor, handle, MetalBufferAccess.READ);
                }
            }
            case INT32 -> {
                int[] data = tensor.getInt32Data();
                if (data == null) {
                    throw new UnsupportedOperationException("Metal INT32 index tensor has no direct int[] storage.");
                }
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment initialData = arena.allocateFrom(JAVA_INT, data);
                    long bytes = (long) data.length * Integer.BYTES;
                    MetalBufferHandle handle = nativeAccess.createBuffer(bytes, STORAGE_MODE_SHARED, initialData, bytes);
                    yield binding(nodeId, tensor, handle, MetalBufferAccess.READ);
                }
            }
            default -> throw new UnsupportedOperationException("Metal buffer inputs support FLOAT32/BFLOAT16/INT32 data buffers only; got " + tensor.getDataType());
        };
    }

    /**
     * Creates a shared BOOL predicate input binding initialized from the tensor's CPU storage.
     *
     * @param nodeId compiled node id represented by the predicate tensor
     * @param tensor runtime tensor
     * @return initialized Metal predicate binding
     */
    public MetalBufferBinding createPredicateInputBinding(int nodeId, Tensor tensor) {
        ensureAvailable();
        validateCommonInput(tensor);
        if (tensor.getDataType() != DataType.BOOL) {
            throw new UnsupportedOperationException("Metal predicate buffer inputs require BOOL tensors; got " + tensor.getDataType());
        }
        byte[] data = tensor.getBoolData();
        if (data == null) {
            throw new UnsupportedOperationException("Metal BOOL predicate tensor has no direct byte[] storage.");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment initialData = arena.allocateFrom(JAVA_BYTE, data);
            long bytes = data.length;
            MetalBufferHandle handle = nativeAccess.createBuffer(bytes, STORAGE_MODE_SHARED, initialData, bytes);
            return binding(nodeId, tensor, handle, MetalBufferAccess.READ);
        }
    }

    /**
     * Creates an unwritten shared output binding.
     *
     * @param nodeId output node id
     * @param layout output layout
     * @return reserved output binding
     */
    public MetalBufferBinding createOutputBinding(int nodeId, AcceleratorBufferLayout layout) {
        ensureAvailable();
        Objects.requireNonNull(layout, "layout cannot be null");
        if (layout.dataType() != DataType.FLOAT32
                && layout.dataType() != DataType.BFLOAT16
                && layout.dataType() != DataType.BOOL
                && layout.dataType() != DataType.INT32
                && layout.dataType() != DataType.INT64) {
            throw new UnsupportedOperationException("Metal buffer outputs support FLOAT32/BFLOAT16/BOOL/INT32/INT64 only in this phase; got " + layout.dataType());
        }
        if (layout.logicalElementCount() <= 0) {
            throw new IllegalArgumentException("Metal output elementCount must be positive.");
        }
        MetalLayoutPolicy.Decision layoutDecision = MetalLayoutPolicy.output(layout);
        if (!layoutDecision.accepted() || !supportsOutputAllocation(layout.layoutClass())) {
            throw new UnsupportedOperationException(
                    "Metal buffer output layout unsupported: " + layoutDecision.reason()
            );
        }
        MetalBufferHandle handle = nativeAccess.createBuffer(layout.logicalByteLength(), STORAGE_MODE_SHARED, MemorySegment.NULL, 0L);
        return new MetalBufferBinding(
                nodeId,
                layout,
                handle,
                MetalBufferAccess.READ_WRITE
        );
    }

    /**
     * Reads a shared Metal binding into CPU tensor storage.
     *
     * @param binding active Metal binding
     * @param destination destination runtime tensor
     * @param reason materialization reason
     * @return materialization diagnostics
     */
    public CpuMaterializationResult readToCpu(
            MetalBufferBinding binding,
            Tensor destination,
            CpuMaterializationReason reason
    ) {
        ensureAvailable();
        Objects.requireNonNull(binding, "binding cannot be null");
        Objects.requireNonNull(destination, "destination cannot be null");
        if (binding.layout().dataType() != destination.getDataType()
                || (binding.layout().dataType() != DataType.FLOAT32
                && binding.layout().dataType() != DataType.BFLOAT16
                && binding.layout().dataType() != DataType.BOOL
                && binding.layout().dataType() != DataType.INT32
                && binding.layout().dataType() != DataType.INT64)) {
            throw new UnsupportedOperationException(
                    "Metal materializer supports dtype-matched FLOAT32/BFLOAT16/BOOL/INT32/INT64 bindings only; binding="
                            + binding.layout().dataType() + ", destination=" + destination.getDataType()
            );
        }
        AcceleratorBufferLayout destinationLayout = AcceleratorBufferLayout.fromTensor(destination);
        if (!Arrays.equals(binding.layout().shape(), destinationLayout.shape())) {
            throw new IllegalArgumentException("Metal binding shape " + Arrays.toString(binding.layout().shape())
                    + " does not match destination shape " + Arrays.toString(destinationLayout.shape()) + ".");
        }
        boolean denseRepairedLogicalLayout = denseRepairedLogicalLayout(binding.layout(), destinationLayout);
        if (!denseRepairedLogicalLayout && !Arrays.equals(binding.layout().strides(), destinationLayout.strides())) {
            throw new IllegalArgumentException("Metal binding strides " + Arrays.toString(binding.layout().strides())
                    + " do not match destination strides " + Arrays.toString(destinationLayout.strides()) + ".");
        }
        if (!denseRepairedLogicalLayout && binding.layout().storageOffset() != destinationLayout.storageOffset()) {
            throw new IllegalArgumentException("Metal binding storageOffset " + binding.layout().storageOffset()
                    + " does not match destination storageOffset " + destinationLayout.storageOffset() + ".");
        }
        if (binding.layout().logicalElementCount() != destinationLayout.logicalElementCount()) {
            throw new IllegalArgumentException("Metal binding elementCount " + binding.layout().logicalElementCount()
                    + " does not match destination elementCount " + destinationLayout.logicalElementCount() + ".");
        }
        long start = System.nanoTime();
        if (binding.layout().layoutClass() == AcceleratorBufferLayoutClass.BROADCAST_ZERO_STRIDE_VIEW) {
            materializePhysicalViewStorage(binding, destination);
        } else if (binding.layout().dataType() == DataType.FLOAT32) {
            float[] data = destination.getFloat32Data();
            if (data == null) {
                throw new UnsupportedOperationException("Destination FLOAT32 tensor has no direct float[] storage.");
            }
            materializeFloat32(binding, destination, destinationLayout, data);
        } else if (binding.layout().dataType() == DataType.BFLOAT16) {
            short[] data = destination.getBFloat16Data();
            if (data == null) {
                throw new UnsupportedOperationException("Destination BFLOAT16 tensor has no direct short[] storage.");
            }
            materializeBFloat16(binding, destination, destinationLayout, data);
        } else if (binding.layout().dataType() == DataType.BOOL) {
            byte[] data = destination.getBoolData();
            if (data == null) {
                throw new UnsupportedOperationException("Destination BOOL tensor has no direct byte[] storage.");
            }
            materializeBool(binding, destination, destinationLayout, data);
        } else if (binding.layout().dataType() == DataType.INT32) {
            int[] data = destination.getInt32Data();
            if (data == null) {
                throw new UnsupportedOperationException("Destination INT32 tensor has no direct int[] storage.");
            }
            materializeInt32(binding, destination, destinationLayout, data);
        } else {
            long[] data = destination.getInt64Data();
            if (data == null) {
                throw new UnsupportedOperationException("Destination INT64 tensor has no direct long[] storage.");
            }
            materializeInt64(binding, destination, destinationLayout, data);
        }
        destination.markDataViewStale();
        return new CpuMaterializationResult(
                System.nanoTime() - start,
                "metal read_buffer materialized nodeId=" + binding.nodeId() + " reason=" + reason.label()
        );
    }

    /**
     * Destroys an owned Metal buffer handle.
     *
     * @param handle handle returned by this allocator
     */
    public void destroy(MetalBufferHandle handle) {
        ensureAvailable();
        if (handle != null && handle.ownsHandle() && handle.available()) {
            nativeAccess.destroyBuffer(handle);
        }
    }

    private void ensureAvailable() {
        if (!available) {
            throw new UnsupportedOperationException(unavailableReason.isBlank()
                    ? "Metal buffer allocator is unavailable."
                    : unavailableReason);
        }
    }

    private static void validateCommonInput(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        if (!tensor.isContiguous()) {
            throw new UnsupportedOperationException("Metal buffer input tensor is not contiguous.");
        }
        if (tensor.hasStorageOffset()) {
            throw new UnsupportedOperationException("Metal buffer input tensor has storage offset.");
        }
    }

    private static boolean supportsOutputAllocation(AcceleratorBufferLayoutClass layoutClass) {
        return switch (layoutClass) {
            case DENSE_CONTIGUOUS, ZERO_OFFSET_VIEW, NON_ZERO_OFFSET_VIEW, PERMUTED_OR_STRIDED_VIEW -> true;
            case BROADCAST_ZERO_STRIDE_VIEW, UNSUPPORTED -> false;
        };
    }

    private void materializePhysicalViewStorage(MetalBufferBinding binding, Tensor destination) {
        long physicalBytes = AcceleratorLayoutAbiV2Descriptor.physicalByteSpan(binding.layout());
        int physicalElements = checkedPhysicalElementCount(binding.layout(), physicalBytes);
        switch (binding.layout().dataType()) {
            case FLOAT32 -> {
                float[] data = destination.getFloat32Data();
                if (data == null) {
                    throw new UnsupportedOperationException("Destination FLOAT32 tensor has no direct float[] storage.");
                }
                requireStorageCapacity(data.length, physicalElements, binding);
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment nativeDestination = arena.allocate(JAVA_FLOAT, physicalElements);
                    nativeAccess.readBuffer(binding.handle(), nativeDestination, physicalBytes);
                    float[] physical = new float[physicalElements];
                    MemorySegment.ofArray(physical).copyFrom(nativeDestination.reinterpret(physicalBytes));
                    System.arraycopy(physical, 0, data, 0, physicalElements);
                }
            }
            case BFLOAT16 -> {
                short[] data = destination.getBFloat16Data();
                if (data == null) {
                    throw new UnsupportedOperationException("Destination BFLOAT16 tensor has no direct short[] storage.");
                }
                requireStorageCapacity(data.length, physicalElements, binding);
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment nativeDestination = arena.allocate(JAVA_SHORT, physicalElements);
                    nativeAccess.readBuffer(binding.handle(), nativeDestination, physicalBytes);
                    short[] physical = new short[physicalElements];
                    MemorySegment.ofArray(physical).copyFrom(nativeDestination.reinterpret(physicalBytes));
                    System.arraycopy(physical, 0, data, 0, physicalElements);
                }
            }
            case BOOL -> {
                byte[] data = destination.getBoolData();
                if (data == null) {
                    throw new UnsupportedOperationException("Destination BOOL tensor has no direct byte[] storage.");
                }
                requireStorageCapacity(data.length, physicalElements, binding);
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment nativeDestination = arena.allocate(JAVA_BYTE, physicalElements);
                    nativeAccess.readBuffer(binding.handle(), nativeDestination, physicalBytes);
                    byte[] physical = new byte[physicalElements];
                    MemorySegment.ofArray(physical).copyFrom(nativeDestination.reinterpret(physicalBytes));
                    System.arraycopy(physical, 0, data, 0, physicalElements);
                }
            }
            case INT32 -> {
                int[] data = destination.getInt32Data();
                if (data == null) {
                    throw new UnsupportedOperationException("Destination INT32 tensor has no direct int[] storage.");
                }
                requireStorageCapacity(data.length, physicalElements, binding);
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment nativeDestination = arena.allocate(JAVA_INT, physicalElements);
                    nativeAccess.readBuffer(binding.handle(), nativeDestination, physicalBytes);
                    int[] physical = new int[physicalElements];
                    MemorySegment.ofArray(physical).copyFrom(nativeDestination.reinterpret(physicalBytes));
                    System.arraycopy(physical, 0, data, 0, physicalElements);
                }
            }
            case INT64 -> {
                long[] data = destination.getInt64Data();
                if (data == null) {
                    throw new UnsupportedOperationException("Destination INT64 tensor has no direct long[] storage.");
                }
                requireStorageCapacity(data.length, physicalElements, binding);
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment nativeDestination = arena.allocate(JAVA_LONG, physicalElements);
                    nativeAccess.readBuffer(binding.handle(), nativeDestination, physicalBytes);
                    long[] physical = new long[physicalElements];
                    MemorySegment.ofArray(physical).copyFrom(nativeDestination.reinterpret(physicalBytes));
                    System.arraycopy(physical, 0, data, 0, physicalElements);
                }
            }
            case FLOAT64 -> throw new UnsupportedOperationException("Metal readback does not support FLOAT64 buffers.");
        }
    }

    private static int checkedPhysicalElementCount(AcceleratorBufferLayout layout, long physicalBytes) {
        int bytesPerElement = bytesPerElement(layout.dataType());
        if (physicalBytes % bytesPerElement != 0L) {
            throw new IllegalArgumentException("Physical byte span " + physicalBytes
                    + " is not divisible by dtype byte width " + bytesPerElement + ".");
        }
        long elements = physicalBytes / bytesPerElement;
        if (elements > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("Metal physical-view materialization supports at most "
                    + Integer.MAX_VALUE + " physical elements.");
        }
        return (int) elements;
    }

    private static int bytesPerElement(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> Double.BYTES;
            case FLOAT32 -> Float.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case INT32 -> Integer.BYTES;
            case INT64 -> Long.BYTES;
            case BOOL -> Byte.BYTES;
        };
    }

    private static void requireStorageCapacity(int actualLength, int requiredElements, MetalBufferBinding binding) {
        if (actualLength < requiredElements) {
            throw new IllegalArgumentException("Destination storage length " + actualLength
                    + " cannot hold Metal physical-view readback span " + requiredElements
                    + " for " + binding.describe() + ".");
        }
    }

    private static boolean denseRepairedLogicalLayout(
            AcceleratorBufferLayout bindingLayout,
            AcceleratorBufferLayout destinationLayout
    ) {
        return bindingLayout.layoutClass() == AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                && bindingLayout.storageOffset() == 0
                && Arrays.equals(bindingLayout.shape(), destinationLayout.shape())
                && bindingLayout.logicalElementCount() == destinationLayout.logicalElementCount();
    }

    private static int checkedLogicalElementCount(long logicalElementCount) {
        if (logicalElementCount > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException(
                    "Metal logical-view materialization supports at most "
                            + Integer.MAX_VALUE + " logical elements."
            );
        }
        return (int) logicalElementCount;
    }

    private void materializeFloat32(
            MetalBufferBinding binding,
            Tensor destination,
            AcceleratorBufferLayout destinationLayout,
            float[] data
    ) {
        if (destination.isContiguous() && !destination.hasStorageOffset()) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeDestination = arena.allocate(JAVA_FLOAT, data.length);
                nativeAccess.readBuffer(binding.handle(), nativeDestination, binding.logicalByteLength());
                MemorySegment.ofArray(data).copyFrom(nativeDestination.reinterpret(binding.logicalByteLength()));
            }
        } else {
            int logicalElements = checkedLogicalElementCount(binding.layout().logicalElementCount());
            float[] dense = new float[logicalElements];
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeDestination = arena.allocate(JAVA_FLOAT, logicalElements);
                nativeAccess.readBuffer(binding.handle(), nativeDestination, binding.logicalByteLength());
                MemorySegment.ofArray(dense).copyFrom(nativeDestination.reinterpret(binding.logicalByteLength()));
            }
            scatterDenseLogicalToDestination(
                    dense,
                    data,
                    destinationLayout.shape(),
                    destinationLayout.strides(),
                    destinationLayout.storageOffset()
            );
        }
    }

    private void materializeBFloat16(
            MetalBufferBinding binding,
            Tensor destination,
            AcceleratorBufferLayout destinationLayout,
            short[] data
    ) {
        if (destination.isContiguous() && !destination.hasStorageOffset()) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeDestination = arena.allocate(JAVA_SHORT, data.length);
                nativeAccess.readBuffer(binding.handle(), nativeDestination, binding.logicalByteLength());
                MemorySegment.ofArray(data).copyFrom(nativeDestination.reinterpret(binding.logicalByteLength()));
            }
        } else {
            int logicalElements = checkedLogicalElementCount(binding.layout().logicalElementCount());
            short[] dense = new short[logicalElements];
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeDestination = arena.allocate(JAVA_SHORT, logicalElements);
                nativeAccess.readBuffer(binding.handle(), nativeDestination, binding.logicalByteLength());
                MemorySegment.ofArray(dense).copyFrom(nativeDestination.reinterpret(binding.logicalByteLength()));
            }
            scatterDenseLogicalToDestination(
                    dense,
                    data,
                    destinationLayout.shape(),
                    destinationLayout.strides(),
                    destinationLayout.storageOffset()
            );
        }
    }

    private void materializeBool(
            MetalBufferBinding binding,
            Tensor destination,
            AcceleratorBufferLayout destinationLayout,
            byte[] data
    ) {
        if (destination.isContiguous() && !destination.hasStorageOffset()) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeDestination = arena.allocate(JAVA_BYTE, data.length);
                nativeAccess.readBuffer(binding.handle(), nativeDestination, binding.logicalByteLength());
                MemorySegment.ofArray(data).copyFrom(nativeDestination.reinterpret(binding.logicalByteLength()));
            }
        } else {
            int logicalElements = checkedLogicalElementCount(binding.layout().logicalElementCount());
            byte[] dense = new byte[logicalElements];
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeDestination = arena.allocate(JAVA_BYTE, logicalElements);
                nativeAccess.readBuffer(binding.handle(), nativeDestination, binding.logicalByteLength());
                MemorySegment.ofArray(dense).copyFrom(nativeDestination.reinterpret(binding.logicalByteLength()));
            }
            scatterDenseLogicalToDestination(
                    dense,
                    data,
                    destinationLayout.shape(),
                    destinationLayout.strides(),
                    destinationLayout.storageOffset()
            );
        }
    }

    private void materializeInt32(
            MetalBufferBinding binding,
            Tensor destination,
            AcceleratorBufferLayout destinationLayout,
            int[] data
    ) {
        if (destination.isContiguous() && !destination.hasStorageOffset()) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeDestination = arena.allocate(JAVA_INT, data.length);
                nativeAccess.readBuffer(binding.handle(), nativeDestination, binding.logicalByteLength());
                MemorySegment.ofArray(data).copyFrom(nativeDestination.reinterpret(binding.logicalByteLength()));
            }
        } else {
            int logicalElements = checkedLogicalElementCount(binding.layout().logicalElementCount());
            int[] dense = new int[logicalElements];
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeDestination = arena.allocate(JAVA_INT, logicalElements);
                nativeAccess.readBuffer(binding.handle(), nativeDestination, binding.logicalByteLength());
                MemorySegment.ofArray(dense).copyFrom(nativeDestination.reinterpret(binding.logicalByteLength()));
            }
            scatterDenseLogicalToDestination(
                    dense,
                    data,
                    destinationLayout.shape(),
                    destinationLayout.strides(),
                    destinationLayout.storageOffset()
            );
        }
    }

    private void materializeInt64(
            MetalBufferBinding binding,
            Tensor destination,
            AcceleratorBufferLayout destinationLayout,
            long[] data
    ) {
        if (destination.isContiguous() && !destination.hasStorageOffset()) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeDestination = arena.allocate(JAVA_LONG, data.length);
                nativeAccess.readBuffer(binding.handle(), nativeDestination, binding.logicalByteLength());
                MemorySegment.ofArray(data).copyFrom(nativeDestination.reinterpret(binding.logicalByteLength()));
            }
        } else {
            int logicalElements = checkedLogicalElementCount(binding.layout().logicalElementCount());
            long[] dense = new long[logicalElements];
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeDestination = arena.allocate(JAVA_LONG, logicalElements);
                nativeAccess.readBuffer(binding.handle(), nativeDestination, binding.logicalByteLength());
                MemorySegment.ofArray(dense).copyFrom(nativeDestination.reinterpret(binding.logicalByteLength()));
            }
            scatterDenseLogicalToDestination(
                    dense,
                    data,
                    destinationLayout.shape(),
                    destinationLayout.strides(),
                    destinationLayout.storageOffset()
            );
        }
    }

    private static void scatterDenseLogicalToDestination(
            float[] dense,
            float[] destination,
            int[] shape,
            int[] strides,
            int storageOffset
    ) {
        Objects.requireNonNull(dense, "dense cannot be null");
        Objects.requireNonNull(destination, "destination cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides must have the same length.");
        }
        if (shape.length > 4) {
            throw new UnsupportedOperationException("Metal logical-view materialization supports rank <= 4");
        }
        int expectedElements = checkedShapeElementCount(shape);
        if (dense.length != expectedElements) {
            throw new IllegalArgumentException(
                    "Dense logical readback length " + dense.length
                            + " does not match destination element count " + expectedElements + "."
            );
        }
        for (int linear = 0; linear < dense.length; linear++) {
            int storageIndex = storageOffset;
            int remaining = linear;
            for (int dim = shape.length - 1; dim >= 0; dim--) {
                int coordinate = remaining % shape[dim];
                remaining /= shape[dim];
                storageIndex += coordinate * strides[dim];
            }
            if (storageIndex < 0 || storageIndex >= destination.length) {
                throw new IndexOutOfBoundsException(
                        "Metal logical-view materialization storage index " + storageIndex
                                + " outside destination storage length " + destination.length + "."
                );
            }
            destination[storageIndex] = dense[linear];
        }
    }

    private static void scatterDenseLogicalToDestination(
            short[] dense,
            short[] destination,
            int[] shape,
            int[] strides,
            int storageOffset
    ) {
        Objects.requireNonNull(dense, "dense cannot be null");
        Objects.requireNonNull(destination, "destination cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides must have the same length.");
        }
        if (shape.length > 4) {
            throw new UnsupportedOperationException("Metal logical-view materialization supports rank <= 4");
        }
        int expectedElements = checkedShapeElementCount(shape);
        if (dense.length != expectedElements) {
            throw new IllegalArgumentException(
                    "Dense logical readback length " + dense.length
                            + " does not match destination element count " + expectedElements + "."
            );
        }
        for (int linear = 0; linear < dense.length; linear++) {
            int storageIndex = storageOffset;
            int remaining = linear;
            for (int dim = shape.length - 1; dim >= 0; dim--) {
                int coordinate = remaining % shape[dim];
                remaining /= shape[dim];
                storageIndex += coordinate * strides[dim];
            }
            if (storageIndex < 0 || storageIndex >= destination.length) {
                throw new IndexOutOfBoundsException(
                        "Metal logical-view materialization storage index " + storageIndex
                                + " outside destination storage length " + destination.length + "."
                );
            }
            destination[storageIndex] = dense[linear];
        }
    }

    private static void scatterDenseLogicalToDestination(
            byte[] dense,
            byte[] destination,
            int[] shape,
            int[] strides,
            int storageOffset
    ) {
        Objects.requireNonNull(dense, "dense cannot be null");
        Objects.requireNonNull(destination, "destination cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides must have the same length.");
        }
        if (shape.length > 4) {
            throw new UnsupportedOperationException("Metal logical-view materialization supports rank <= 4");
        }
        int expectedElements = checkedShapeElementCount(shape);
        if (dense.length != expectedElements) {
            throw new IllegalArgumentException(
                    "Dense logical readback length " + dense.length
                            + " does not match destination element count " + expectedElements + "."
            );
        }
        for (int linear = 0; linear < dense.length; linear++) {
            int storageIndex = storageOffset;
            int remaining = linear;
            for (int dim = shape.length - 1; dim >= 0; dim--) {
                int coordinate = remaining % shape[dim];
                remaining /= shape[dim];
                storageIndex += coordinate * strides[dim];
            }
            if (storageIndex < 0 || storageIndex >= destination.length) {
                throw new IndexOutOfBoundsException(
                        "Metal logical-view materialization storage index " + storageIndex
                                + " outside destination storage length " + destination.length + "."
                );
            }
            destination[storageIndex] = dense[linear];
        }
    }

    private static void scatterDenseLogicalToDestination(
            int[] dense,
            int[] destination,
            int[] shape,
            int[] strides,
            int storageOffset
    ) {
        Objects.requireNonNull(dense, "dense cannot be null");
        Objects.requireNonNull(destination, "destination cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides must have the same length.");
        }
        if (shape.length > 4) {
            throw new UnsupportedOperationException("Metal logical-view materialization supports rank <= 4");
        }
        int expectedElements = checkedShapeElementCount(shape);
        if (dense.length != expectedElements) {
            throw new IllegalArgumentException(
                    "Dense logical readback length " + dense.length
                            + " does not match destination element count " + expectedElements + "."
            );
        }
        for (int linear = 0; linear < dense.length; linear++) {
            int storageIndex = storageOffset;
            int remaining = linear;
            for (int dim = shape.length - 1; dim >= 0; dim--) {
                int coordinate = remaining % shape[dim];
                remaining /= shape[dim];
                storageIndex += coordinate * strides[dim];
            }
            if (storageIndex < 0 || storageIndex >= destination.length) {
                throw new IndexOutOfBoundsException(
                        "Metal logical-view materialization storage index " + storageIndex
                                + " outside destination storage length " + destination.length + "."
                );
            }
            destination[storageIndex] = dense[linear];
        }
    }

    private static void scatterDenseLogicalToDestination(
            long[] dense,
            long[] destination,
            int[] shape,
            int[] strides,
            int storageOffset
    ) {
        Objects.requireNonNull(dense, "dense cannot be null");
        Objects.requireNonNull(destination, "destination cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides must have the same length.");
        }
        if (shape.length > 4) {
            throw new UnsupportedOperationException("Metal logical-view materialization supports rank <= 4");
        }
        int expectedElements = checkedShapeElementCount(shape);
        if (dense.length != expectedElements) {
            throw new IllegalArgumentException(
                    "Dense logical readback length " + dense.length
                            + " does not match destination element count " + expectedElements + "."
            );
        }
        for (int linear = 0; linear < dense.length; linear++) {
            int storageIndex = storageOffset;
            int remaining = linear;
            for (int dim = shape.length - 1; dim >= 0; dim--) {
                int coordinate = remaining % shape[dim];
                remaining /= shape[dim];
                storageIndex += coordinate * strides[dim];
            }
            if (storageIndex < 0 || storageIndex >= destination.length) {
                throw new IndexOutOfBoundsException(
                        "Metal logical-view materialization storage index " + storageIndex
                                + " outside destination storage length " + destination.length + "."
                );
            }
            destination[storageIndex] = dense[linear];
        }
    }

    private static int checkedShapeElementCount(int[] shape) {
        long elements = 1L;
        for (int dimension : shape) {
            elements = Math.multiplyExact(elements, dimension);
            if (elements > Integer.MAX_VALUE) {
                throw new UnsupportedOperationException(
                        "Metal logical-view materialization supports at most "
                                + Integer.MAX_VALUE + " logical elements."
                );
            }
        }
        return (int) elements;
    }

    private static MetalBufferBinding binding(int nodeId, Tensor tensor, MetalBufferHandle handle, MetalBufferAccess access) {
        return new MetalBufferBinding(
                nodeId,
                AcceleratorBufferLayout.fromTensor(tensor),
                handle,
                access
        );
    }
}
