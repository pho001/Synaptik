package backend.metal.lowering;

import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import operations.index.gatherAxisGrad;
import operations.index.gatherGrad;
import operations.index.scatterAdd;
import operations.index.takeAlongAxisGrad;
import tensor.DataType;

import java.util.Arrays;

final class MetalIndexWriteSemantics {
    private MetalIndexWriteSemantics() {
    }

    static boolean isIndexWriteOrGradient(Operation.OpType opType) {
        return opType == Operation.OpType.SCATTER_ADD
                || opType == Operation.OpType.GATHER_GRAD
                || opType == Operation.OpType.GATHER_AXIS_GRAD
                || opType == Operation.OpType.TAKE_ALONG_AXIS_GRAD;
    }

    static String unsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        String common = commonReason(opType, node, context);
        if (!common.isBlank()) {
            return common;
        }
        return switch (opType) {
            case SCATTER_ADD -> scatterAddReason(node, context);
            case GATHER_GRAD -> gatherGradReason(node, context);
            case GATHER_AXIS_GRAD -> gatherAxisGradReason(node, context);
            case TAKE_ALONG_AXIS_GRAD -> takeAlongAxisGradReason(node, context);
            default -> "";
        };
    }

    private static String scatterAddReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof scatterAdd op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD descriptor is unavailable";
        }
        if (node.inputIds().size() != 3) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD requires base, INT32 indices, and source inputs";
        }
        CompiledNode base = context.compiledNode(node.inputIds().get(0));
        CompiledNode indices = context.compiledNode(node.inputIds().get(1));
        CompiledNode src = context.compiledNode(node.inputIds().get(2));
        if (base == null || indices == null || src == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD inputs are unavailable";
        }
        String dtypeReason = requireFloatingValueAndInt32Index("SCATTER_ADD", node, indices, base, src);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("SCATTER_ADD", base, indices, src);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = op.getDimension();
        int[] baseShape = base.shape();
        if (axis < 0 || axis >= baseShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD axis is outside base rank";
        }
        if (!Arrays.equals(node.shape(), baseShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD output shape must equal base shape";
        }
        int[] expected = reduceShape(baseShape, axis);
        if (!Arrays.equals(indices.shape(), expected) || !Arrays.equals(src.shape(), expected)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD indices/source shape must equal base shape without scattered axis";
        }
        String boundsReason = staticBoundsReason("SCATTER_ADD", indices, baseShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return "";
    }

    private static String gatherGradReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof gatherGrad op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_GRAD descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_GRAD requires INT32 indices and output-gradient inputs";
        }
        CompiledNode indices = context.compiledNode(node.inputIds().get(0));
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(1));
        if (indices == null || outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_GRAD inputs are unavailable";
        }
        String dtypeReason = requireFloatingValueAndInt32Index("GATHER_GRAD", node, indices, outGrad);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("GATHER_GRAD", indices, outGrad);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = op.getDimension();
        int[] outputShape = node.shape();
        if (axis < 0 || axis >= outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_GRAD axis is outside output rank";
        }
        int[] expected = reduceShape(outputShape, axis);
        if (!Arrays.equals(indices.shape(), expected) || !Arrays.equals(outGrad.shape(), expected)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_GRAD indices/outGrad shape must equal output shape without gathered axis";
        }
        String boundsReason = staticBoundsReason("GATHER_GRAD", indices, outputShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return "";
    }

    private static String gatherAxisGradReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof gatherAxisGrad op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD requires INT32 indices and output-gradient inputs";
        }
        CompiledNode indices = context.compiledNode(node.inputIds().get(0));
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(1));
        if (indices == null || outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD inputs are unavailable";
        }
        String dtypeReason = requireFloatingValueAndInt32Index("GATHER_AXIS_GRAD", node, indices, outGrad);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("GATHER_AXIS_GRAD", indices, outGrad);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = op.getAxis();
        int[] outputShape = node.shape();
        int[] gradShape = outGrad.shape();
        if (axis < 0 || axis >= outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD axis is outside output rank";
        }
        if (indices.shape().length != 1 || gradShape.length != outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD supports 1-D index tensors that preserve value rank";
        }
        for (int i = 0; i < outputShape.length; i++) {
            int expected = i == axis ? indices.shape()[0] : outputShape[i];
            if (gradShape[i] != expected) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD outGrad shape must equal output shape with gathered axis replaced by index length";
            }
        }
        String boundsReason = staticBoundsReason("GATHER_AXIS_GRAD", indices, outputShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return "";
    }

    private static String takeAlongAxisGradReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof takeAlongAxisGrad op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD requires INT32 indices and output-gradient inputs";
        }
        CompiledNode indices = context.compiledNode(node.inputIds().get(0));
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(1));
        if (indices == null || outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD inputs are unavailable";
        }
        String dtypeReason = requireFloatingValueAndInt32Index("TAKE_ALONG_AXIS_GRAD", node, indices, outGrad);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("TAKE_ALONG_AXIS_GRAD", indices, outGrad);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = op.getDimension();
        int[] outputShape = node.shape();
        int[] gradShape = outGrad.shape();
        if (axis < 0 || axis >= outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD axis is outside output rank";
        }
        if (!Arrays.equals(indices.shape(), gradShape) || gradShape.length != outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD indices shape must equal outGrad shape and rank";
        }
        for (int i = 0; i < outputShape.length; i++) {
            if (i != axis && gradShape[i] != outputShape[i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD non-axis dimensions must match output shape";
            }
        }
        String boundsReason = staticBoundsReason("TAKE_ALONG_AXIS_GRAD", indices, outputShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return "";
    }

    private static String commonReason(Operation.OpType opType, CompiledNode node, PartitionPlanningContext context) {
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: " + opType + " nodes are not legal inside nested Metal backward regions";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires planning context";
        }
        int[] shape = node.shape();
        if (shape.length < 1 || shape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " supports rank 1..4 tensors";
        }
        return "";
    }

    private static String requireFloatingValueAndInt32Index(
            String opName,
            CompiledNode output,
            CompiledNode indices,
            CompiledNode... values
    ) {
        if (!isMetalFloatingDType(output.dataType())) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " output requires FLOAT32/BFLOAT16";
        }
        if (indices.dataType() != DataType.INT32) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " index input requires INT32";
        }
        for (CompiledNode value : values) {
            if (value != null && value.dataType() != output.dataType()) {
                return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " value inputs must match FLOAT32/BFLOAT16 output dtype";
            }
        }
        return "";
    }

    private static boolean isMetalFloatingDType(DataType dtype) {
        return dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16;
    }

    private static String requireDense(String opName, CompiledNode... inputs) {
        for (CompiledNode input : inputs) {
            if (input != null && (!input.contiguous() || input.hasStorageOffset())) {
                return "UNSUPPORTED_LAYOUT: GPU_METAL " + opName + " inputs require dense layout";
            }
        }
        return "";
    }

    private static String staticBoundsReason(String opName, CompiledNode indices, int axisSize) {
        if (axisSize <= 0) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opName + " axis size must be positive";
        }
        if (!indices.leaf()) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opName + " index bounds require a static INT32 leaf tensor";
        }
        int[] data;
        try {
            data = indices.semanticTensor().getInt32Data();
        } catch (RuntimeException ex) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opName + " index bounds require readable INT32 storage";
        }
        if (data == null) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opName + " index bounds require readable INT32 storage";
        }
        for (int value : data) {
            if (value < 0 || value >= axisSize) {
                return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opName + " index " + value + " is outside axis size " + axisSize;
            }
        }
        return "";
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
