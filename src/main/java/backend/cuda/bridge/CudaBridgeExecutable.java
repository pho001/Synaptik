package backend.cuda.bridge;

import java.lang.foreign.MemorySegment;
import java.util.List;

public record CudaBridgeExecutable(
        boolean available,
        MemorySegment handle,
        String reason,
        boolean cacheHit,
        List<Integer> externalInputNodeIds,
        List<Integer> outputNodeIds
) {
    public CudaBridgeExecutable {
        handle = handle == null ? MemorySegment.NULL : handle;
        reason = reason == null ? "" : reason;
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
    }

    public static CudaBridgeExecutable unavailable(String reason) {
        return new CudaBridgeExecutable(false, MemorySegment.NULL, reason, false, List.of(), List.of());
    }
}
