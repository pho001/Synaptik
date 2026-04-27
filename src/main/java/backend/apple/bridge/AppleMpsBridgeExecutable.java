package backend.apple.bridge;

import java.util.List;
import java.lang.foreign.MemorySegment;

public record AppleMpsBridgeExecutable(
        boolean available,
        MemorySegment handle,
        String reason,
        boolean cacheHit,
        List<Integer> externalInputNodeIds,
        List<Integer> outputNodeIds,
        List<Integer> outputNodeIndexes
) {
    public AppleMpsBridgeExecutable {
        handle = handle == null ? MemorySegment.NULL : handle;
        reason = reason == null ? "" : reason;
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        outputNodeIndexes = List.copyOf(outputNodeIndexes == null ? List.of() : outputNodeIndexes);
    }

    public static AppleMpsBridgeExecutable unavailable(String reason) {
        return new AppleMpsBridgeExecutable(false, MemorySegment.NULL, reason, false, List.of(), List.of(), List.of());
    }
}
