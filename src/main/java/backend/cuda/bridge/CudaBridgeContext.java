package backend.cuda.bridge;

import java.lang.foreign.MemorySegment;

/**
 * Native CUDA bridge context handle and availability state.
 *
 * @param available whether {@code handle} can be used for compile and execute calls
 * @param handle native context handle, or {@link MemorySegment#NULL}
 * @param reason unavailable reason when {@code available} is false
 */
public record CudaBridgeContext(
        boolean available,
        MemorySegment handle,
        String reason
) {
    public CudaBridgeContext {
        handle = handle == null ? MemorySegment.NULL : handle;
        reason = reason == null ? "" : reason;
    }

    /**
     * Creates an unavailable context carrying a diagnostic reason.
     */
    public static CudaBridgeContext unavailable(String reason) {
        return new CudaBridgeContext(false, MemorySegment.NULL, reason);
    }
}
