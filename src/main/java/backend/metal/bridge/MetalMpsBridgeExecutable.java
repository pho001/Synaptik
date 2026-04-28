package backend.metal.bridge;

import java.util.List;
import java.lang.foreign.MemorySegment;

/**
 * Compiled native Metal MPS executable and runtime tensor mapping.
 *
 * @param available whether {@code handle} can be executed
 * @param handle native executable handle, or {@link MemorySegment#NULL}
 * @param reason unavailable reason when {@code available} is false
 * @param cacheHit whether compilation reused an existing native executable
 * @param externalInputNodeIds compiled-node ids expected as runtime inputs
 * @param outputNodeIds compiled-node ids written by execution
 * @param outputNodeIndexes output indices inside the lowered DAG
 */
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

    /**
     * Creates an unavailable executable carrying a diagnostic reason.
     */
    public static MetalMpsBridgeExecutable unavailable(String reason) {
        return new MetalMpsBridgeExecutable(false, MemorySegment.NULL, reason, false, List.of(), List.of(), List.of());
    }
}
