package backend.metal.bridge;

import java.util.List;
import java.lang.foreign.MemorySegment;

public record MetalMpsBridgeExecutable(
        boolean available,
        MemorySegment handle,
        String reason,
        boolean cacheHit,
        List<Integer> externalInputNodeIds,
        List<Integer> outputNodeIds,
        List<Integer> outputNodeIndexes
) {
    public MetalMpsBridgeExecutable {
        handle = handle == null ? MemorySegment.NULL : handle;
        reason = reason == null ? "" : reason;
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        outputNodeIndexes = List.copyOf(outputNodeIndexes == null ? List.of() : outputNodeIndexes);
    }

    public static MetalMpsBridgeExecutable unavailable(String reason) {
        return new MetalMpsBridgeExecutable(false, MemorySegment.NULL, reason, false, List.of(), List.of(), List.of());
    }
}
