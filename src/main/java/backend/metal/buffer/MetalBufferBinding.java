package backend.metal.buffer;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferAccessMode;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.memory.DeviceBufferBinding;

import java.util.Objects;

/**
 * Runtime binding between one compiled graph value and a Metal-compatible buffer.
 *
 * <p>This is the Java-side contract used by the native Metal buffer bridge.
 * It carries only execution facts: node id, shared logical layout metadata,
 * buffer handle, and access intent. Tensor materialization policy remains in execution
 * state; native code receives buffer bindings rather than semantic tensors.</p>
 *
 * @param nodeId compiled node id represented by the buffer
 * @param layout backend-neutral logical tensor layout
 * @param handle native buffer handle
 * @param access native access intent
 */
public record MetalBufferBinding(
        int nodeId,
        AcceleratorBufferLayout layout,
        MetalBufferHandle handle,
        MetalBufferAccess access
) implements DeviceBufferBinding {
    public MetalBufferBinding {
        Objects.requireNonNull(layout, "layout cannot be null");
        Objects.requireNonNull(handle, "handle cannot be null");
        access = access == null ? MetalBufferAccess.READ : access;
    }

    /**
     * Returns the expected byte size implied by the shared layout.
     *
     * @return logical payload byte size
     */
    @Override
    public long logicalByteLength() {
        return layout.logicalByteLength();
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

    @Override
    public AcceleratorBufferAccessMode accessMode() {
        return switch (access) {
            case READ -> AcceleratorBufferAccessMode.READ;
            case WRITE -> AcceleratorBufferAccessMode.WRITE;
            case READ_WRITE -> AcceleratorBufferAccessMode.READ_WRITE;
        };
    }

    @Override
    public String nativeHandleIdentity() {
        return backendId()
                + ":owner=" + handle.owner()
                + ",storageMode=" + handle.storageMode()
                + ",bytes=" + handle.byteLength();
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
                + ", " + layout.describe()
                + ", access=" + access
                + ", handleBytes=" + handle.byteLength();
    }
}
