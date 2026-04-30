package backend.cuda.buffer;

import java.lang.foreign.MemorySegment;

/**
 * Backend-owned CUDA buffer handle and logical byte extent.
 *
 * @param handle native CUDA buffer handle
 * @param byteLength logical byte length represented by the handle
 * @param ownsHandle whether this run owns and should destroy the handle
 */
public record CudaBufferHandle(MemorySegment handle, long byteLength, boolean ownsHandle) {
    public CudaBufferHandle {
        handle = handle == null ? MemorySegment.NULL : handle;
        byteLength = Math.max(0L, byteLength);
    }

    /**
     * Returns whether this handle can be used for CUDA buffer execution.
     */
    public boolean available() {
        return !handle.equals(MemorySegment.NULL) && byteLength > 0L;
    }

    /**
     * Returns an opaque diagnostic identity for traces and messages.
     */
    public String identity() {
        if (!available()) {
            return "cuda:null";
        }
        return "cuda:0x" + Long.toHexString(handle.address());
    }
}
