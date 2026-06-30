package backend.cuda.lowering;


import backend.cuda.CudaDTypeRolePolicy;
import graph.model.CompiledNode;
import graph.compile.planning.partition.PartitionPlanningContext;
import operations.Operation;
import operations.index.gatherGrad;
import operations.index.scatterAdd;
import operations.index.takeAlongAxisGrad;
import tensor.DataType;

import java.util.Arrays;

/**
 * CUDA semantics checks for index-write and index-gradient operations before stable rejection.
 */
final class CudaIndexWriteSemantics {
    private CudaIndexWriteSemantics() {
    }

    static boolean isHandled(Operation.OpType opType) {
        return opType == Operation.OpType.SCATTER_ADD
                || opType == Operation.OpType.GATHER_GRAD
                || opType == Operation.OpType.TAKE_ALONG_AXIS_GRAD;
    }

    static String unsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA index-write op metadata is unavailable";
        }
        Operation.OpType opType = node.operation().opType();
        String common = commonReason(opType, node, context);
        if (!common.isBlank()) {
            return common;
        }
        return switch (opType) {
            case SCATTER_ADD -> scatterAddReason(node, context);
            case GATHER_GRAD -> gatherGradReason(node, context);
            case TAKE_ALONG_AXIS_GRAD -> takeAlongAxisGradReason(node, context);
            default -> "";
        };
    }

    private static String scatterAddReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof scatterAdd op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SCATTER_ADD descriptor is unavailable";
        }
        if (node.inputIds().size() != 3) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SCATTER_ADD requires base, INT32 indices, and source inputs";
        }
        CompiledNode base = context.compiledNode(node.inputIds().get(0));
        CompiledNode indices = context.compiledNode(node.inputIds().get(1));
        CompiledNode src = context.compiledNode(node.inputIds().get(2));
        if (base == null || indices == null || src == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SCATTER_ADD inputs are unavailable";
        }
        String dtypeReason = requireFloat32ValueAndInt32Index("SCATTER_ADD", node, indices, base, src);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("SCATTER_ADD", base, indices, src);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = normalizeAxis(op.getDimension(), base.shape().length);
        int[] baseShape = base.shape();
        if (axis < 0 || axis >= baseShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SCATTER_ADD axis is outside base rank";
        }
        if (!Arrays.equals(node.shape(), baseShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SCATTER_ADD output shape must equal base shape";
        }
        int[] expected = reduceShape(baseShape, axis);
        if (!Arrays.equals(indices.shape(), expected) || !Arrays.equals(src.shape(), expected)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA SCATTER_ADD indices/source shape must equal base shape without scattered axis";
        }
        String boundsReason = staticBoundsReason("SCATTER_ADD", indices, baseShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return duplicateUnsupported("SCATTER_ADD");
    }

    private static String gatherGradReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof gatherGrad op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA GATHER_GRAD descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA GATHER_GRAD requires INT32 indices and output-gradient inputs";
        }
        CompiledNode indices = context.compiledNode(node.inputIds().get(0));
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(1));
        if (indices == null || outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA GATHER_GRAD inputs are unavailable";
        }
        String dtypeReason = requireFloat32ValueAndInt32Index("GATHER_GRAD", node, indices, outGrad);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("GATHER_GRAD", indices, outGrad);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = normalizeAxis(op.getDimension(), node.shape().length);
        int[] outputShape = node.shape();
        if (axis < 0 || axis >= outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA GATHER_GRAD axis is outside output rank";
        }
        int[] expected = reduceShape(outputShape, axis);
        if (!Arrays.equals(indices.shape(), expected) || !Arrays.equals(outGrad.shape(), expected)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA GATHER_GRAD indices/outGrad shape must equal output shape without gathered axis";
        }
        String boundsReason = staticBoundsReason("GATHER_GRAD", indices, outputShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return duplicateUnsupported("GATHER_GRAD");
    }

    private static String takeAlongAxisGradReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof takeAlongAxisGrad op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA TAKE_ALONG_AXIS_GRAD descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA TAKE_ALONG_AXIS_GRAD requires INT32 indices and output-gradient inputs";
        }
        CompiledNode indices = context.compiledNode(node.inputIds().get(0));
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(1));
        if (indices == null || outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA TAKE_ALONG_AXIS_GRAD inputs are unavailable";
        }
        String dtypeReason = requireFloat32ValueAndInt32Index("TAKE_ALONG_AXIS_GRAD", node, indices, outGrad);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("TAKE_ALONG_AXIS_GRAD", indices, outGrad);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = normalizeAxis(op.getDimension(), node.shape().length);
        int[] outputShape = node.shape();
        int[] gradShape = outGrad.shape();
        if (axis < 0 || axis >= outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA TAKE_ALONG_AXIS_GRAD axis is outside output rank";
        }
        if (!Arrays.equals(indices.shape(), gradShape) || gradShape.length != outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA TAKE_ALONG_AXIS_GRAD indices shape must equal outGrad shape and rank";
        }
        for (int i = 0; i < outputShape.length; i++) {
            if (i != axis && gradShape[i] != outputShape[i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA TAKE_ALONG_AXIS_GRAD non-axis dimensions must match output shape";
            }
        }
        String boundsReason = staticBoundsReason("TAKE_ALONG_AXIS_GRAD", indices, outputShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return duplicateUnsupported("TAKE_ALONG_AXIS_GRAD");
    }

    private static String commonReason(Operation.OpType opType, CompiledNode node, PartitionPlanningContext context) {
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: " + opType + " nodes are not legal inside nested CUDA backward regions";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " requires planning context";
        }
        int[] shape = node.shape();
        if (shape.length < 1 || shape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_CUDA " + opType + " supports rank 1..4 tensors";
        }
        return "";
    }

    private static String requireFloat32ValueAndInt32Index(
            String opName,
            CompiledNode output,
            CompiledNode indices,
            CompiledNode... values
    ) {
        if (output.dataType() != DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_CUDA " + opName + " currently supports only FLOAT32 output";
        }
        var indexDecision = CudaDTypeRolePolicy.indexInput(indices.dataType());
        if (!indexDecision.supported()) {
            return "UNSUPPORTED_DTYPE: " + indexDecision.detail();
        }
        for (CompiledNode value : values) {
            if (value != null && value.dataType() != DataType.FLOAT32) {
                return "UNSUPPORTED_DTYPE: GPU_CUDA " + opName + " currently supports only FLOAT32 value inputs";
            }
        }
        return "";
    }

    private static String requireDense(String opName, CompiledNode... inputs) {
        for (CompiledNode input : inputs) {
            if (input != null && (!input.contiguous() || input.hasStorageOffset())) {
                return "UNSUPPORTED_LAYOUT: GPU_CUDA " + opName + " inputs require dense layout";
            }
        }
        return "";
    }

    private static String staticBoundsReason(String opName, CompiledNode indices, int axisSize) {
        if (axisSize <= 0) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opName + " axis size must be positive";
        }
        if (!indices.leaf()) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opName + " index bounds require a static INT32 leaf tensor";
        }
        int[] data = indices.staticDataSnapshot().int32Values();
        if (data == null) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opName + " index bounds require readable INT32 storage";
        }
        int logicalElements = indices.flatDataSize();
        if (logicalElements < 0 || logicalElements > data.length) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opName + " index bounds cannot be proven from storage";
        }
        for (int i = 0; i < logicalElements; i++) {
            int index = data[i];
            if (index < 0 || index >= axisSize) {
                return "UNSUPPORTED_BOUNDS_CHECK: GPU_CUDA " + opName + " index " + index
                        + " is outside axis size " + axisSize;
            }
        }
        return "";
    }

    private static String duplicateUnsupported(String opName) {
        String detail = switch (opName) {
            case "TAKE_ALONG_AXIS_GRAD" -> "duplicate-index accumulation parity and rank-preserving static bounds checks are not proven";
            case "SCATTER_ADD" -> "duplicate-index accumulation order/tolerance and native write-add semantics are not proven";
            default -> "duplicate-index accumulation parity is not proven";
        };
        return "UNSUPPORTED_DUPLICATE_INDEX: operation " + opName
                + " is not supported by GPU_CUDA lowering; GPU_CUDA native " + detail
                + "; family=INDEX_SCATTER_GATHER target=scatter_index_gradient_small";
    }

    private static int normalizeAxis(int axis, int rank) {
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
