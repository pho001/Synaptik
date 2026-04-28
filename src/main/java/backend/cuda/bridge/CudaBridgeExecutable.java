package backend.cuda.bridge;

import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Compiled native CUDA graph executable and runtime tensor mapping.
 *
 * @param available whether {@code handle} can be executed
 * @param handle native executable handle, or {@link MemorySegment#NULL}
 * @param reason unavailable reason when {@code available} is false
 * @param cacheHit whether compilation reused an existing native executable
 * @param externalInputNodeIds compiled-node ids expected as runtime inputs
 * @param outputNodeIds compiled-node ids written by execution
 */
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

    /**
     * Creates an unavailable executable carrying a diagnostic reason.
     */
    public static CudaBridgeExecutable unavailable(String reason) {
        return new CudaBridgeExecutable(false, MemorySegment.NULL, reason, false, List.of(), List.of());
    }
}
