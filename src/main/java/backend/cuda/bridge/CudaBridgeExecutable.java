package backend.cuda.bridge;

import java.lang.foreign.MemorySegment;
import java.util.List;

import tensor.DataType;

/**
 * Compiled native CUDA graph executable and runtime tensor mapping.
 *
 * @param available whether {@code handle} can be executed
 * @param handle native executable handle, or {@link MemorySegment#NULL}
 * @param reason unavailable reason when {@code available} is false
 * @param cacheHit whether compilation reused an existing native executable
 * @param externalInputNodeIds compiled-node ids expected as runtime inputs
 * @param externalInputDataTypes data types expected for runtime inputs
 * @param outputNodeIds compiled-node ids written by execution
 * @param outputDataTypes data types produced by runtime outputs
 */
public record CudaBridgeExecutable(
        boolean available,
        MemorySegment handle,
        String reason,
        boolean cacheHit,
        List<Integer> externalInputNodeIds,
        List<DataType> externalInputDataTypes,
        List<Integer> outputNodeIds,
        List<DataType> outputDataTypes
) {
    public CudaBridgeExecutable {
        handle = handle == null ? MemorySegment.NULL : handle;
        reason = reason == null ? "" : reason;
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        externalInputDataTypes = List.copyOf(externalInputDataTypes == null ? List.of() : externalInputDataTypes);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        outputDataTypes = List.copyOf(outputDataTypes == null ? List.of() : outputDataTypes);
        if (!externalInputDataTypes.isEmpty() && externalInputDataTypes.size() != externalInputNodeIds.size()) {
            throw new IllegalArgumentException("externalInputDataTypes size must match externalInputNodeIds size when provided");
        }
        if (!outputDataTypes.isEmpty() && outputDataTypes.size() != outputNodeIds.size()) {
            throw new IllegalArgumentException("outputDataTypes size must match outputNodeIds size when provided");
        }
    }

    /**
     * Creates an executable with node ids only for older tests and unavailable fakes.
     */
    public CudaBridgeExecutable(
            boolean available,
            MemorySegment handle,
            String reason,
            boolean cacheHit,
            List<Integer> externalInputNodeIds,
            List<Integer> outputNodeIds
    ) {
        this(available, handle, reason, cacheHit, externalInputNodeIds, List.of(), outputNodeIds, List.of());
    }

    /**
     * Creates an unavailable executable carrying a diagnostic reason.
     */
    public static CudaBridgeExecutable unavailable(String reason) {
        return new CudaBridgeExecutable(false, MemorySegment.NULL, reason, false, List.of(), List.of(), List.of(), List.of());
    }
}
