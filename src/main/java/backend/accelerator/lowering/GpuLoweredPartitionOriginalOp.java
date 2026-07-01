package backend.accelerator.lowering;

import tensor.DataType;

import java.util.List;

/**
 * Original compiled operation metadata inside a GPU lowered-partition manifest.
 *
 * @param nodeId original compiled node id
 * @param opType original operation type
 * @param inputNodeIds original input node ids
 * @param outputNodeIds original output node ids represented by this op
 * @param dataType original operation output dtype
 * @param shape original operation output shape
 * @param loweredPrimitiveIds primitive ids produced from this operation
 * @param aggregatedReasons stable reasons aggregated from child primitives
 */
public record GpuLoweredPartitionOriginalOp(
        int nodeId,
        String opType,
        List<Integer> inputNodeIds,
        List<Integer> outputNodeIds,
        DataType dataType,
        List<Integer> shape,
        List<String> loweredPrimitiveIds,
        List<GpuLoweringUnsupportedReason> aggregatedReasons
) {
    public GpuLoweredPartitionOriginalOp {
        opType = opType == null ? "UNKNOWN" : opType;
        inputNodeIds = List.copyOf(inputNodeIds == null ? List.of() : inputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        dataType = dataType == null ? DataType.FLOAT32 : dataType;
        shape = List.copyOf(shape == null ? List.of() : shape);
        loweredPrimitiveIds = List.copyOf(loweredPrimitiveIds == null ? List.of() : loweredPrimitiveIds);
        aggregatedReasons = List.copyOf(aggregatedReasons == null ? List.of() : aggregatedReasons);
    }
}
