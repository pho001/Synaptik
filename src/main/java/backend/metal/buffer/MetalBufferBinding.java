package backend.metal.buffer;

import backend.ComputeBackend;
import backend.memory.DeviceBufferBinding;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Runtime binding between one compiled graph value and a Metal-compatible buffer.
 *
 * <p>This is the Java-side contract used by the native Metal buffer bridge.
 * It carries only execution facts: node id, dtype, shape, element count, buffer
 * handle, and access intent. Tensor materialization policy remains in execution
 * state; native code receives buffer bindings rather than semantic tensors.</p>
 *
 * @param nodeId compiled node id represented by the buffer
 * @param dataType tensor dtype stored in the buffer
 * @param shape logical tensor shape
 * @param elementCount logical element count
 * @param handle native buffer handle
 * @param access native access intent
 */
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

    /**
     * Returns a defensive copy of the logical shape.
     *
     * @return tensor shape
     */
    @Override
    public int[] shape() {
        return shape.clone();
    }

    /**
     * Returns the expected byte size implied by dtype and element count.
     *
     * @return logical payload byte size
     */
    public long logicalByteLength() {
        return elementCount * elementByteSize(dataType);
    }

    /**
     * Returns the backend id for Metal buffer bindings.
     *
     * @return {@code GPU_METAL}
     */
    @Override
    public String backendId() {
        return ComputeBackend.GPU_METAL.name();
    }

    /**
     * Returns whether the attached native buffer can hold the logical payload.
     *
     * @return true when the handle is available and large enough
     */
    public boolean bufferCoversLogicalPayload() {
        return handle.available() && handle.byteLength() >= logicalByteLength();
    }

    /**
     * Returns whether the binding can be used by Metal.
     *
     * @return true when the native handle is available and large enough
     */
    @Override
    public boolean available() {
        return bufferCoversLogicalPayload();
    }

    /**
     * Returns a concise diagnostic string for trace and failure messages.
     *
     * @return binding summary
     */
    public String describe() {
        return "nodeId=" + nodeId
                + ", dtype=" + dataType
                + ", shape=" + Arrays.toString(shape)
                + ", access=" + access
                + ", bytes=" + logicalByteLength()
                + ", handleBytes=" + handle.byteLength();
    }

    private static int elementByteSize(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> Double.BYTES;
            case FLOAT32 -> Float.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
            case INT32 -> Integer.BYTES;
        };
    }
}
