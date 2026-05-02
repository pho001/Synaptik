package backend.cuda.lowering;

import backend.cuda.CudaDTypeRolePolicy;
import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import operations.index.gather;
import operations.index.takeAlongAxis;
import tensor.DataType;

import java.util.Arrays;

/**
 * CUDA partition planner predicates that require operation-specific semantic checks.
 */
public final class CudaPartitionSupport {
    private CudaPartitionSupport() {
    }

    public static boolean isForwardIndexOp(Operation.OpType opType) {
        return opType == Operation.OpType.GATHER || opType == Operation.OpType.TAKE_ALONG_AXIS;
    }

    /**
     * Returns a stable reason for CUDA forward gather/take rejection.
     *
     * <p>Legal scoped inputs currently reach a final CAPABILITY_MISSING result because CUDA native forward
     * gather/take execution has not been implemented. Invalid inputs fail earlier with dtype/layout/rank/bounds detail.</p>
     */
    public static String indexUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA index op metadata is unavailable";
        }
        Operation.OpType opType = node.operation().opType();
        if (!isForwardIndexOp(opType)) {
            return "";
        }
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: forward " + opType + " nodes are not legal inside CUDA backward regions";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " requires planning context";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " requires value and INT32 index inputs";
        }
        CompiledNode value = context.compiledNode(node.inputIds().get(0));
        CompiledNode indices = context.compiledNode(node.inputIds().get(1));
        if (value == null || indices == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " inputs are unavailable";
        }
        if (value.dataType() != DataType.FLOAT32 || node.dataType() != DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_CUDA " + opType + " currently supports only FLOAT32 value/output tensors";
        }
        var indexDecision = CudaDTypeRolePolicy.indexInput(indices.dataType());
        if (!indexDecision.supported()) {
            return "UNSUPPORTED_DTYPE: " + indexDecision.detail();
        }
        if (!value.contiguous() || value.hasStorageOffset()
                || !indices.contiguous() || indices.hasStorageOffset()) {
            return "UNSUPPORTED_LAYOUT: GPU_CUDA " + opType + " inputs require dense value and INT32 index layouts";
        }
        int[] valueShape = value.shape();
        int[] indexShape = indices.shape();
        int[] outputShape = node.shape();
        if (valueShape.length < 1 || valueShape.length > 4
                || indexShape.length < 1 || indexShape.length > 4
                || outputShape.length < 1 || outputShape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " supports rank 1..4 tensors";
        }
        int axis = normalizedIndexAxis(node, valueShape.length);
        if (axis < 0 || axis >= valueShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " axis is outside value rank";
        }
        String boundsReason = indexBoundsUnsupportedReason(indices, valueShape[axis], opType);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        if (opType == Operation.OpType.GATHER) {
            int[] expected = reduceShape(valueShape, axis);
            if (!Arrays.equals(indexShape, expected) || !Arrays.equals(outputShape, expected)) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA GATHER index/output shape must equal value shape without gathered axis";
            }
        } else {
            if (indexShape.length != valueShape.length || !Arrays.equals(outputShape, indexShape)) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA TAKE_ALONG_AXIS index/output rank and shape must match";
            }
            for (int i = 0; i < valueShape.length; i++) {
                if (i != axis && indexShape[i] != valueShape[i]) {
                    return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA TAKE_ALONG_AXIS non-axis dimensions must match value input";
                }
            }
        }
        return "CAPABILITY_MISSING: operation " + opType
                + " is not supported by GPU_CUDA lowering family=INDEX_SCATTER_GATHER"
                + " status=unsupported note=CUDA forward " + opType
                + " native/lowered path is not implemented yet; target=gather_take_small";
    }

    private static String indexBoundsUnsupportedReason(CompiledNode indices, int axisSize, Operation.OpType opType) {
        if (axisSize <= 0) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opType + " axis size must be positive";
        }
        if (!indices.leaf()) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opType + " index bounds require a static INT32 leaf tensor";
        }
        int[] data;
        try {
            data = indices.semanticTensor().getInt32Data();
        } catch (RuntimeException ex) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opType + " index bounds require readable INT32 storage";
        }
        if (data == null) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opType + " index bounds require readable INT32 storage";
        }
        int logicalElements = indices.flatDataSize();
        if (logicalElements < 0 || logicalElements > data.length) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opType + " index bounds cannot be proven from storage";
        }
        for (int i = 0; i < logicalElements; i++) {
            int index = data[i];
            if (index < 0 || index >= axisSize) {
                return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opType + " index " + index
                        + " is outside axis size " + axisSize;
            }
        }
        return "";
    }

    private static int normalizedIndexAxis(CompiledNode node, int rank) {
        int axis = switch (node.operation().opType()) {
            case GATHER -> node.operation() instanceof gather op ? op.getDimension() : -1;
            case TAKE_ALONG_AXIS -> node.operation() instanceof takeAlongAxis op ? op.getDimension() : -1;
            default -> -1;
        };
        return axis < 0 ? axis + rank : axis;
    }

    private static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }
}
