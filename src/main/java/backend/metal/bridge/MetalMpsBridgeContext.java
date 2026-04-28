package backend.metal.bridge;

import java.lang.foreign.MemorySegment;

/**
 * Native Metal MPS bridge context handle and availability state.
 *
 * @param available whether {@code handle} can be used for compile and execute calls
 * @param handle native context handle, or {@link MemorySegment#NULL}
 * @param reason unavailable reason when {@code available} is false
 */
public record MetalMpsBridgeContext(
        boolean available,
        MemorySegment handle,
        String reason
) {
    public MetalMpsBridgeContext {
        handle = handle == null ? MemorySegment.NULL : handle;
        reason = reason == null ? "" : reason;
    }

    /**
     * Creates an unavailable context carrying a diagnostic reason.
     */
    public static MetalMpsBridgeContext unavailable(String reason) {
        return new MetalMpsBridgeContext(false, MemorySegment.NULL, reason);
    }
}
