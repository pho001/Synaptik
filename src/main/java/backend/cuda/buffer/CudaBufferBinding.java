package backend.cuda.buffer;

import backend.contract.ComputeBackend;
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

    /**
     * Creates a logical view binding over an existing CUDA handle.
     *
     * <p>The returned binding borrows the source handle and does not register or own any native
     * resource. The allocation that produced {@code source} remains responsible for cleanup.</p>
     *
     * @param nodeId target compiled node id
     * @param layout target logical layout
     * @param source source binding whose handle is reused
     * @param access target access mode; defaults to source access when null
     * @return view binding over the same native handle
     */
    public static CudaBufferBinding viewOf(
            int nodeId,
            AcceleratorBufferLayout layout,
            CudaBufferBinding source,
            CudaBufferAccess access
    ) {
        Objects.requireNonNull(layout, "layout cannot be null");
        Objects.requireNonNull(source, "source cannot be null");
        if (!source.available()) {
            throw new IllegalArgumentException("source binding is unavailable: " + source.describe());
        }
        return new CudaBufferBinding(
                nodeId,
                layout,
                source.handle(),
                access == null ? source.access() : access
        );
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
    public CudaBufferBinding viewOf(
            int targetNodeId,
            AcceleratorBufferLayout targetLayout,
            AcceleratorBufferAccessMode accessMode
    ) {
        return viewOf(targetNodeId, targetLayout, this, cudaAccess(accessMode));
    }

    private static CudaBufferAccess cudaAccess(AcceleratorBufferAccessMode accessMode) {
        if (accessMode == null) {
            return null;
        }
        return switch (accessMode) {
            case READ -> CudaBufferAccess.READ;
            case WRITE -> CudaBufferAccess.WRITE;
            case READ_WRITE -> CudaBufferAccess.READ_WRITE;
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
