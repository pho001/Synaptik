package backend.metal.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferLayoutClass;
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

/**
 * Run-scoped allocator and materializer for native Metal buffer bindings.
 *
 * <p>The allocator is created from the active Metal bridge context. It owns no global state: each allocated
 * {@link MetalBufferHandle} is returned to execution code, which registers the handle as a run resource and
 * eventually calls {@link #destroy(MetalBufferHandle)}. The allocator supports only the narrow phase-46
 * contract: shared F32 compute buffers, BOOL predicate input buffers, and F32 CPU materialization.</p>
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
     * Creates a shared F32 input binding initialized from the tensor's CPU storage.
     *
     * @param nodeId compiled node id represented by the tensor
     * @param tensor runtime tensor
     * @return initialized Metal input binding
     */
    public MetalBufferBinding createInputBinding(int nodeId, Tensor tensor) {
        ensureAvailable();
        validateCommonInput(tensor);
        if (tensor.getDataType() != DataType.FLOAT32) {
            throw new UnsupportedOperationException("Metal buffer inputs support FLOAT32 data buffers only; got " + tensor.getDataType());
        }
        float[] data = tensor.getFloat32Data();
        if (data == null) {
            throw new UnsupportedOperationException("Metal FLOAT32 input tensor has no direct float[] storage.");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment initialData = arena.allocateFrom(JAVA_FLOAT, data);
            long bytes = (long) data.length * Float.BYTES;
            MetalBufferHandle handle = nativeAccess.createBuffer(bytes, STORAGE_MODE_SHARED, initialData, bytes);
            return binding(nodeId, tensor, handle, MetalBufferAccess.READ);
        }
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
     * Creates an unwritten shared F32 output binding.
     *
     * @param nodeId output node id
     * @param layout output layout
     * @return reserved output binding
     */
    public MetalBufferBinding createOutputBinding(int nodeId, AcceleratorBufferLayout layout) {
        ensureAvailable();
        Objects.requireNonNull(layout, "layout cannot be null");
        if (layout.dataType() != DataType.FLOAT32) {
            throw new UnsupportedOperationException("Metal buffer outputs support FLOAT32 only in this phase; got " + layout.dataType());
        }
        if (layout.logicalElementCount() <= 0) {
            throw new IllegalArgumentException("Metal output elementCount must be positive.");
        }
        if (layout.layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
            throw new UnsupportedOperationException(
                    "Metal buffer outputs require DENSE_CONTIGUOUS layout in this phase; got " + layout.layoutClass()
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
     * Reads a shared F32 Metal binding into CPU tensor storage.
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
        if (binding.layout().dataType() != DataType.FLOAT32 || destination.getDataType() != DataType.FLOAT32) {
            throw new UnsupportedOperationException(
                    "Metal materializer supports FLOAT32 bindings only; binding="
                            + binding.layout().dataType() + ", destination=" + destination.getDataType()
            );
        }
        AcceleratorBufferLayout destinationLayout = AcceleratorBufferLayout.fromTensor(destination);
        if (!Arrays.equals(binding.layout().shape(), destinationLayout.shape())) {
            throw new IllegalArgumentException("Metal binding shape " + Arrays.toString(binding.layout().shape())
                    + " does not match destination shape " + Arrays.toString(destinationLayout.shape()) + ".");
        }
        if (!Arrays.equals(binding.layout().strides(), destinationLayout.strides())) {
            throw new IllegalArgumentException("Metal binding strides " + Arrays.toString(binding.layout().strides())
                    + " do not match destination strides " + Arrays.toString(destinationLayout.strides()) + ".");
        }
        if (binding.layout().storageOffset() != destinationLayout.storageOffset()) {
            throw new IllegalArgumentException("Metal binding storageOffset " + binding.layout().storageOffset()
                    + " does not match destination storageOffset " + destinationLayout.storageOffset() + ".");
        }
        if (binding.layout().logicalElementCount() != destinationLayout.logicalElementCount()) {
            throw new IllegalArgumentException("Metal binding elementCount " + binding.layout().logicalElementCount()
                    + " does not match destination elementCount " + destinationLayout.logicalElementCount() + ".");
        }
        if (!destination.isContiguous() || destination.hasStorageOffset()) {
            throw new UnsupportedOperationException(
                    "Metal read_buffer materialization requires a contiguous, zero-offset destination tensor.");
        }
        float[] data = destination.getFloat32Data();
        if (data == null) {
            throw new UnsupportedOperationException("Destination FLOAT32 tensor has no direct float[] storage.");
        }
        long start = System.nanoTime();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment nativeDestination = arena.allocate(JAVA_FLOAT, data.length);
            nativeAccess.readBuffer(binding.handle(), nativeDestination, binding.logicalByteLength());
            MemorySegment.ofArray(data).copyFrom(nativeDestination.reinterpret(binding.logicalByteLength()));
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

    private static MetalBufferBinding binding(int nodeId, Tensor tensor, MetalBufferHandle handle, MetalBufferAccess access) {
        return new MetalBufferBinding(
                nodeId,
                AcceleratorBufferLayout.fromTensor(tensor),
                handle,
                access
        );
    }
}
