package backend.apple.bridge;

import java.util.List;
import java.lang.foreign.MemorySegment;

public record AppleMpsBridgeExecutable(
        boolean available,
        MemorySegment handle,
        String reason,
        boolean cacheHit,
        List<Integer> externalInputNodeIds,
        int outputNodeId,
        int outputNodeIndex
) {
    public AppleMpsBridgeExecutable {
        handle = handle == null ? MemorySegment.NULL : handle;
        reason = reason == null ? "" : reason;
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
    }

    public static AppleMpsBridgeExecutable unavailable(String reason) {
        return new AppleMpsBridgeExecutable(false, MemorySegment.NULL, reason, false, List.of(), -1, -1);
    }
}
