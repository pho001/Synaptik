package backend.cuda.bridge;

import java.lang.foreign.MemorySegment;

public record CudaBridgeContext(
        boolean available,
        MemorySegment handle,
        String reason
) {
    public CudaBridgeContext {
        handle = handle == null ? MemorySegment.NULL : handle;
        reason = reason == null ? "" : reason;
    }

    public static CudaBridgeContext unavailable(String reason) {
        return new CudaBridgeContext(false, MemorySegment.NULL, reason);
    }
}
