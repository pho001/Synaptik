package backend.cuda.bridge;

import java.util.List;

public record CudaBridgeExecutable(
        boolean available,
        String reason,
        boolean cacheHit,
        List<Integer> externalInputNodeIds,
        int outputNodeId
) {
    public CudaBridgeExecutable {
        reason = reason == null ? "" : reason;
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
    }

    public static CudaBridgeExecutable unavailable(String reason) {
        return new CudaBridgeExecutable(false, reason, false, List.of(), -1);
    }
}
