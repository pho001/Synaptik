package backend.cuda.lowering;


import backend.cuda.CudaDTypeRolePolicy;
import graph.CompiledNode;
import graph.compile.planning.partition.PartitionPlanningContext;
import operations.Operation;
import operations.index.gather;
import operations.index.takeAlongAxis;
import operations.layout.fold2d;
import operations.layout.unfold2d;
import operations.layout.unfoldAxis;
import tensor.DataType;
import tensor.options.Window2dOptions;

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

    public static boolean isWindowLayoutOp(Operation.OpType opType) {
        return opType == Operation.OpType.UNFOLD_AXIS
                || opType == Operation.OpType.UNFOLD2D
                || opType == Operation.OpType.FOLD2D;
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

    public static String windowLayoutUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA window layout op metadata is unavailable";
        }
        Operation.OpType opType = node.operation().opType();
        if (!isWindowLayoutOp(opType)) {
            return "";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " requires planning context";
        }
        if (node.dataType() != DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_CUDA " + opType + " native window layout lowering currently supports FLOAT32";
        }
        if (node.inputIds().size() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " requires one input";
        }
        CompiledNode input = context.compiledNode(node.inputIds().getFirst());
        if (input == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " input is unavailable";
        }
        if (input.dataType() != DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_CUDA " + opType + " input/output dtype must be FLOAT32";
        }
        if (!input.contiguous() || input.hasStorageOffset()) {
            return "UNSUPPORTED_LAYOUT: GPU_CUDA " + opType + " input requires dense contiguous layout";
        }
        return switch (opType) {
            case UNFOLD_AXIS -> cudaUnfoldAxisReason(node, input);
            case UNFOLD2D -> cudaUnfold2dReason(node, input);
            case FOLD2D -> cudaFold2dReason(node, input);
            default -> "";
        };
    }

    private static String cudaUnfoldAxisReason(CompiledNode node, CompiledNode input) {
        if (!(node.operation() instanceof unfoldAxis op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA UNFOLD_AXIS descriptor is unavailable";
        }
        int[] inputShape = input.shape();
        int[] outputShape = node.shape();
        if (inputShape.length < 1 || inputShape.length > 3 || outputShape.length != inputShape.length + 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA UNFOLD_AXIS supports input rank 1..3 and output rank 2..4";
        }
        if (op.getAxis() < 0 || op.getAxis() >= inputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA UNFOLD_AXIS axis is outside input rank";
        }
        if (op.getAxis() > 15 || op.getSize() > 4095 || op.getStep() > 4095) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA UNFOLD_AXIS axis/size/step exceed native metadata encoding";
        }
        int windows = ((inputShape[op.getAxis()] - op.getSize()) / op.getStep()) + 1;
        if (op.getSize() <= 0 || op.getStep() <= 0 || windows <= 0
                || outputShape[op.getAxis()] != windows
                || outputShape[outputShape.length - 1] != op.getSize()) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA UNFOLD_AXIS output shape must match static sliding-window geometry";
        }
        for (int d = 0; d < inputShape.length; d++) {
            if (d != op.getAxis() && outputShape[d] != inputShape[d]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA UNFOLD_AXIS non-window dimensions must match input shape";
            }
        }
        return "";
    }

    private static String cudaUnfold2dReason(CompiledNode node, CompiledNode input) {
        if (!(node.operation() instanceof unfold2d op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA UNFOLD2D descriptor is unavailable";
        }
        int[] inputShape = input.shape();
        int[] outputShape = node.shape();
        if (inputShape.length != 4 || outputShape.length != 3) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA UNFOLD2D requires rank-4 NCHW input and rank-3 column output";
        }
        return cudaWindow2dReason("UNFOLD2D", op.getOptions(), inputShape, outputShape);
    }

    private static String cudaFold2dReason(CompiledNode node, CompiledNode input) {
        if (!(node.operation() instanceof fold2d op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA FOLD2D descriptor is unavailable";
        }
        int[] inputShape = input.shape();
        int[] outputShape = node.shape();
        if (inputShape.length != 3 || outputShape.length != 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA FOLD2D requires rank-3 columns and rank-4 NCHW output";
        }
        return cudaWindow2dReason("FOLD2D", op.getOptions(), outputShape, inputShape);
    }

    private static String cudaWindow2dReason(String opName, Window2dOptions options, int[] nchwShape, int[] columnShape) {
        if (options.dilationH() != 1 || options.dilationW() != 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " currently supports dilation=1 native lowering";
        }
        if (options.kernelH() < 1 || options.kernelH() > 15
                || options.kernelW() < 1 || options.kernelW() > 15
                || options.strideH() < 1 || options.strideH() > 15
                || options.strideW() < 1 || options.strideW() > 15
                || options.padH() < 0 || options.padH() > 15
                || options.padW() < 0 || options.padW() > 15) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " geometry exceeds native metadata encoding";
        }
        int outH = (nchwShape[2] + 2 * options.padH() - options.kernelH()) / options.strideH() + 1;
        int outW = (nchwShape[3] + 2 * options.padW() - options.kernelW()) / options.strideW() + 1;
        if (outH <= 0 || outW <= 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " output window geometry is empty";
        }
        int kernelArea = options.kernelH() * options.kernelW();
        if (columnShape[0] != nchwShape[0]
                || columnShape[1] != nchwShape[1] * kernelArea
                || columnShape[2] != outH * outW) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opName + " column shape must match NCHW window geometry";
        }
        return "";
    }

    private static String indexBoundsUnsupportedReason(CompiledNode indices, int axisSize, Operation.OpType opType) {
        if (axisSize <= 0) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opType + " axis size must be positive";
        }
        if (!indices.leaf()) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opType + " index bounds require a static INT32 leaf tensor";
        }
        int[] data = indices.staticDataSnapshot().int32Values();
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
