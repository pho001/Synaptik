package backend.metal.bridge;

import java.lang.foreign.MemorySegment;

public record MetalMpsBridgeContext(
        boolean available,
        MemorySegment handle,
        String reason
) {
    public MetalMpsBridgeContext {
        handle = handle == null ? MemorySegment.NULL : handle;
        reason = reason == null ? "" : reason;
    }

    public static MetalMpsBridgeContext unavailable(String reason) {
        return new MetalMpsBridgeContext(false, MemorySegment.NULL, reason);
    }
}
