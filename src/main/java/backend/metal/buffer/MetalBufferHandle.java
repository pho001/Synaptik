package backend.metal.buffer;

import java.lang.foreign.MemorySegment;

/**
 * Opaque handle to a Metal-compatible native buffer.
 *
 * <p>The handle is intentionally small and backend-owned. Java graph execution
 * code can reason about byte length, ownership, and storage mode without
 * depending on Objective-C classes. The native bridge is responsible for
 * interpreting {@link #nativeHandle()} as an {@code id<MTLBuffer>} or compatible
 * shim-owned handle.</p>
 *
 * @param nativeHandle native buffer handle, or {@link MemorySegment#NULL} when unavailable
 * @param byteLength byte length of the buffer
 * @param storageMode stable storage mode label reported by the allocator or bridge
 * @param owner backend component responsible for releasing the handle
 * @param ownsHandle whether this Java-side wrapper owns the native handle lifetime
 */
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

    /**
     * Returns whether this handle points to a non-null native buffer with non-zero capacity.
     *
     * @return true when the handle can be considered for native execution
     */
    public boolean available() {
        return !nativeHandle.equals(MemorySegment.NULL) && byteLength > 0L;
    }
}
