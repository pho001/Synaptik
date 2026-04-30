package backend.cuda.buffer;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferAccessMode;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.memory.DeviceBufferBinding;

import java.util.Objects;

/**
 * CUDA-specific device buffer binding for one compiled node.
 *
 * @param nodeId compiled node id represented by the buffer
 * @param layout logical layout covered by the native handle
 * @param handle CUDA native buffer handle
 * @param access access granted for this binding
 */
public record CudaBufferBinding(
        int nodeId,
        AcceleratorBufferLayout layout,
        CudaBufferHandle handle,
        CudaBufferAccess access
) implements DeviceBufferBinding {
    public CudaBufferBinding {
        Objects.requireNonNull(layout, "layout cannot be null");
        Objects.requireNonNull(handle, "handle cannot be null");
        access = access == null ? CudaBufferAccess.READ_WRITE : access;
    }

    @Override
    public String backendId() {
        return ComputeBackend.GPU_CUDA.name();
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
        return handle.identity();
    }

    @Override
    public boolean available() {
        return handle.available();
    }

    @Override
    public String describe() {
        return "CudaBufferBinding{"
                + "nodeId=" + nodeId
                + ", backend=" + backendId()
                + ", layoutClass=" + layout.layoutClass()
                + ", bytes=" + logicalByteLength()
                + ", access=" + access
                + ", handle=" + nativeHandleIdentity()
                + '}';
    }
}
