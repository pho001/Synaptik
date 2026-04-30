package backend.cuda.buffer;

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

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Run-scoped allocator and materializer for CUDA device-owned buffers.
 */
public final class CudaBufferAllocator {
    /**
     * Native CUDA buffer access supplied by the active bridge implementation.
     */
    public interface NativeAccess {
        CudaBufferHandle createBuffer(long byteLength, MemorySegment initialData, long initialDataBytes);

        void readBuffer(CudaBufferHandle handle, MemorySegment destination, long byteLength);

        void destroyBuffer(CudaBufferHandle handle);
    }

    private final boolean available;
    private final String unavailableReason;
    private final NativeAccess nativeAccess;

    private CudaBufferAllocator(boolean available, String unavailableReason, NativeAccess nativeAccess) {
        this.available = available;
        this.unavailableReason = unavailableReason == null ? "" : unavailableReason;
        this.nativeAccess = nativeAccess;
    }

    /**
     * Creates an allocator backed by native CUDA buffer calls.
     */
    public static CudaBufferAllocator available(NativeAccess nativeAccess) {
        return new CudaBufferAllocator(true, "", Objects.requireNonNull(nativeAccess, "nativeAccess cannot be null"));
    }

    /**
     * Creates an unavailable allocator with a stable reason.
     */
    public static CudaBufferAllocator unavailable(String reason) {
        return new CudaBufferAllocator(false, reason, null);
    }

    public boolean available() {
        return available;
    }

    public String unavailableReason() {
        return unavailableReason;
    }

    /**
     * Creates a CUDA input binding initialized from dense CPU FLOAT32 storage.
     */
    public CudaBufferBinding createInputBinding(int nodeId, Tensor tensor) {
        ensureAvailable();
        Objects.requireNonNull(tensor, "tensor cannot be null");
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.fromTensor(tensor);
        validateDenseFloat32(layout, "CUDA buffer inputs");
        float[] data = tensor.getFloat32Data();
        if (data == null) {
            throw new UnsupportedOperationException("CUDA FLOAT32 input tensor has no direct float[] storage.");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment initialData = arena.allocateFrom(JAVA_FLOAT, data);
            long bytes = (long) data.length * Float.BYTES;
            CudaBufferHandle handle = nativeAccess.createBuffer(bytes, initialData, bytes);
            return new CudaBufferBinding(nodeId, layout, handle, CudaBufferAccess.READ);
        }
    }

    /**
     * Creates an unwritten CUDA output binding for dense FLOAT32 output bytes.
     */
    public CudaBufferBinding createOutputBinding(int nodeId, AcceleratorBufferLayout layout) {
        ensureAvailable();
        validateDenseFloat32(layout, "CUDA buffer outputs");
        CudaBufferHandle handle = nativeAccess.createBuffer(layout.logicalByteLength(), MemorySegment.NULL, 0L);
        return new CudaBufferBinding(nodeId, layout, handle, CudaBufferAccess.READ_WRITE);
    }

    /**
     * Reads a CUDA device buffer into the destination tensor's CPU FLOAT32 storage.
     */
    public CpuMaterializationResult readToCpu(
            CudaBufferBinding binding,
            Tensor destination,
            CpuMaterializationReason reason
    ) {
        ensureAvailable();
        Objects.requireNonNull(binding, "binding cannot be null");
        Objects.requireNonNull(destination, "destination cannot be null");
        if (binding.layout().dataType() != DataType.FLOAT32 || destination.getDataType() != DataType.FLOAT32) {
            throw new UnsupportedOperationException("CUDA materializer supports FLOAT32 bindings only.");
        }
        AcceleratorBufferLayout destinationLayout = AcceleratorBufferLayout.fromTensor(destination);
        validateSameLayout(binding.layout(), destinationLayout);
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
                "cuda read_buffer materialized nodeId=" + binding.nodeId()
                        + " reason=" + (reason == null ? "" : reason.label())
        );
    }

    /**
     * Destroys an owned CUDA buffer handle.
     */
    public void destroy(CudaBufferHandle handle) {
        ensureAvailable();
        if (handle != null && handle.ownsHandle() && handle.available()) {
            nativeAccess.destroyBuffer(handle);
        }
    }

    private void ensureAvailable() {
        if (!available) {
            throw new UnsupportedOperationException(unavailableReason.isBlank()
                    ? "CUDA buffer allocator is unavailable."
                    : unavailableReason);
        }
    }

    private static void validateDenseFloat32(AcceleratorBufferLayout layout, String label) {
        Objects.requireNonNull(layout, "layout cannot be null");
        if (layout.dataType() != DataType.FLOAT32) {
            throw new UnsupportedOperationException(label + " support FLOAT32 only; got " + layout.dataType());
        }
        if (layout.layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
            throw new UnsupportedOperationException(label + " require DENSE_CONTIGUOUS layout; got " + layout.layoutClass());
        }
        if (layout.logicalElementCount() <= 0 || layout.logicalByteLength() <= 0) {
            throw new IllegalArgumentException(label + " require positive logical size.");
        }
    }

    private static void validateSameLayout(AcceleratorBufferLayout source, AcceleratorBufferLayout target) {
        if (!Arrays.equals(source.shape(), target.shape())) {
            throw new IllegalArgumentException("CUDA binding shape " + Arrays.toString(source.shape())
                    + " does not match destination shape " + Arrays.toString(target.shape()) + ".");
        }
        if (!Arrays.equals(source.strides(), target.strides())) {
            throw new IllegalArgumentException("CUDA binding strides " + Arrays.toString(source.strides())
                    + " do not match destination strides " + Arrays.toString(target.strides()) + ".");
        }
        if (source.storageOffset() != target.storageOffset()) {
            throw new IllegalArgumentException("CUDA binding storageOffset " + source.storageOffset()
                    + " does not match destination storageOffset " + target.storageOffset() + ".");
        }
        if (source.logicalElementCount() != target.logicalElementCount()) {
            throw new IllegalArgumentException("CUDA binding elementCount " + source.logicalElementCount()
                    + " does not match destination elementCount " + target.logicalElementCount() + ".");
        }
    }
}
