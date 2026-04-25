package backend.apple.bridge;

import java.lang.foreign.MemorySegment;

public record AppleMpsBridgeContext(
        boolean available,
        MemorySegment handle,
        String reason
) {
    public AppleMpsBridgeContext {
        handle = handle == null ? MemorySegment.NULL : handle;
        reason = reason == null ? "" : reason;
    }

    public static AppleMpsBridgeContext unavailable(String reason) {
        return new AppleMpsBridgeContext(false, MemorySegment.NULL, reason);
    }
}
