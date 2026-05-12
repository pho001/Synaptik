package backend.metal.lowering;

import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import operations.loss.crossEntropyLoss;
import operations.loss.crossEntropyLossIndices;
import operations.loss.crossEntropyLossIndicesGrad;
import operations.loss.nllLoss;
import tensor.DataType;
import tensor.Tensor;
import tensor.loss.LossReduction;

import java.util.Arrays;

final class MetalLossSemantics {
    private MetalLossSemantics() {
    }

    static boolean isDenseLoss(Operation.OpType opType) {
        return opType == Operation.OpType.NLL_LOSS
                || opType == Operation.OpType.CROSS_ENTROPY_LOSS
                || opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES
                || opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD;
    }

    static String unsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        if (opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES) {
            return indexLossUnsupportedReason(node, context);
        }
        if (opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD) {
            return indexLossGradUnsupportedReason(node, context);
        }
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: " + opType + " nodes are not legal inside nested Metal backward regions";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires planning context";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires scores/log-probabilities and dense targets";
        }
        int classAxis = classAxis(node);
        if (classAxis < 0) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " descriptor is unavailable";
        }
        CompiledNode scores = context.compiledNode(node.inputIds().get(0));
        CompiledNode targets = context.compiledNode(node.inputIds().get(1));
        if (scores == null || targets == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " inputs are unavailable";
        }
        if (!isMetalFloatingDType(node.dataType())
                || scores.dataType() != node.dataType()
                || targets.dataType() != node.dataType()) {
            return "UNSUPPORTED_DTYPE: GPU_METAL dense " + opType + " requires dtype-matched FLOAT32/BFLOAT16 output, scores/log-probabilities, and dense targets";
        }
        if (!scores.contiguous() || scores.hasStorageOffset()
                || !targets.contiguous() || targets.hasStorageOffset()) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL dense " + opType + " inputs require dense zero-offset layout";
        }
        int[] scoreShape = scores.shape();
        int[] targetShape = targets.shape();
        if (scoreShape.length < 1 || scoreShape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL dense " + opType + " supports rank 1..4 tensors";
        }
        if (classAxis >= scoreShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL dense " + opType + " class axis is outside input rank";
        }
        if (!Arrays.equals(scoreShape, targetShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL dense " + opType + " dense target shape must match input shape";
        }
        if (!Arrays.equals(node.shape(), new int[]{1})) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL dense " + opType + " currently locks mean-reduced scalar output shape [1]";
        }
        return "";
    }

    private static String indexLossUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: " + opType + " nodes are not legal inside nested Metal backward regions";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires planning context";
        }
        if (!(node.operation() instanceof crossEntropyLossIndices loss)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CROSS_ENTROPY_LOSS_INDICES descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CROSS_ENTROPY_LOSS_INDICES requires logits and INT32 target indices";
        }
        CompiledNode logits = context.compiledNode(node.inputIds().get(0));
        CompiledNode targets = context.compiledNode(node.inputIds().get(1));
        if (logits == null || targets == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CROSS_ENTROPY_LOSS_INDICES inputs are unavailable";
        }
        String common = indexLossCommonReason("CROSS_ENTROPY_LOSS_INDICES", node, logits, targets, loss.getClassDimension());
        if (!common.isBlank()) {
            return common;
        }
        int[] expectedOutputShape = loss.getReduction() == LossReduction.NONE
                ? reduceShape(logits.shape(), loss.getClassDimension())
                : new int[]{1};
        if (!Arrays.equals(node.shape(), expectedOutputShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CROSS_ENTROPY_LOSS_INDICES output shape does not match reduction contract";
        }
        if (loss.hasIgnoreIndex()
                && (loss.getIgnoreIndex() < Short.MIN_VALUE || loss.getIgnoreIndex() > Short.MAX_VALUE)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CROSS_ENTROPY_LOSS_INDICES ignoreIndex exceeds native DAG metadata range";
        }
        return targetBoundsReason(targets, logits.shape()[loss.getClassDimension()], loss.getIgnoreIndex());
    }

    private static String indexLossGradUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires planning context";
        }
        if (!(node.operation() instanceof crossEntropyLossIndicesGrad grad)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CROSS_ENTROPY_LOSS_INDICES_GRAD descriptor is unavailable";
        }
        if (node.inputIds().size() != 3) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CROSS_ENTROPY_LOSS_INDICES_GRAD requires logits, INT32 target indices, and sampleScale";
        }
        CompiledNode logits = context.compiledNode(node.inputIds().get(0));
        CompiledNode targets = context.compiledNode(node.inputIds().get(1));
        CompiledNode sampleScale = context.compiledNode(node.inputIds().get(2));
        if (logits == null || targets == null || sampleScale == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CROSS_ENTROPY_LOSS_INDICES_GRAD inputs are unavailable";
        }
        String common = indexLossCommonReason("CROSS_ENTROPY_LOSS_INDICES_GRAD", node, logits, targets, grad.getClassDimension());
        if (!common.isBlank()) {
            return common;
        }
        if (sampleScale.dataType() != node.dataType()) {
            return "UNSUPPORTED_DTYPE: GPU_METAL CROSS_ENTROPY_LOSS_INDICES_GRAD sampleScale must match FLOAT32/BFLOAT16 output dtype";
        }
        if ((!sampleScale.contiguous() || sampleScale.hasStorageOffset()) && !isGpuProducedLayoutValue(sampleScale)) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL CROSS_ENTROPY_LOSS_INDICES_GRAD sampleScale must be dense or produced by a GPU layout DAG op";
        }
        int[] reducedShape = reduceShape(logits.shape(), grad.getClassDimension());
        if (!Arrays.equals(sampleScale.shape(), reducedShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CROSS_ENTROPY_LOSS_INDICES_GRAD sampleScale shape must equal logits shape without class axis";
        }
        if (!Arrays.equals(node.shape(), logits.shape())) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL CROSS_ENTROPY_LOSS_INDICES_GRAD output shape must match logits shape";
        }
        return targetBoundsReason(targets, logits.shape()[grad.getClassDimension()], null);
    }

    private static boolean isGpuProducedLayoutValue(CompiledNode node) {
        if (node == null || node.operation() == null) {
            return false;
        }
        return switch (node.operation().opType()) {
            case EXPAND, EXPAND_DIMS, SQUEEZE, RESHAPE, CONTIGUOUS, NOOP -> true;
            default -> false;
        };
    }

    private static String indexLossCommonReason(
            String opName,
            CompiledNode node,
            CompiledNode logits,
            CompiledNode targets,
            int classAxis
    ) {
        if (!isMetalFloatingDType(node.dataType()) || logits.dataType() != node.dataType()) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " requires dtype-matched FLOAT32/BFLOAT16 logits/output";
        }
        if (targets.dataType() != DataType.INT32) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " requires INT32 target indices";
        }
        if (!logits.contiguous() || logits.hasStorageOffset()
                || !targets.contiguous() || targets.hasStorageOffset()) {
            return "UNSUPPORTED_LAYOUT: GPU_METAL " + opName + " inputs require dense zero-offset layout";
        }
        int[] logitsShape = logits.shape();
        if (logitsShape.length < 1 || logitsShape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " supports rank 1..4 logits";
        }
        if (classAxis < 0 || classAxis >= logitsShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " class axis is outside input rank";
        }
        int[] expectedTargets = reduceShape(logitsShape, classAxis);
        if (!Arrays.equals(targets.shape(), expectedTargets)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opName + " target shape must equal logits shape without class axis";
        }
        return "";
    }

    private static String targetBoundsReason(CompiledNode targets, int classCount, Integer ignoreIndex) {
        Tensor tensor = targets.semanticTensor();
        if (tensor == null || tensor.getDataType() != DataType.INT32 || tensor.getInt32Data() == null) {
            return "UNSUPPORTED_INDEX_SEMANTICS: GPU_METAL index-target loss requires statically inspectable INT32 targets before native execution";
        }
        int[] values = tensor.getInt32Data();
        int offset = tensor.getStorageOffsetUnsafe();
        int size = tensor.getFlatDataSize();
        for (int i = 0; i < size; i++) {
            int value = values[offset + i];
            if (ignoreIndex != null && value == ignoreIndex) {
                continue;
            }
            if (value < 0 || value >= classCount) {
                return "UNSUPPORTED_INDEX_SEMANTICS: GPU_METAL index-target loss target out of range: " + value + " for classes=" + classCount;
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

    private static int classAxis(CompiledNode node) {
        Operation operation = node.operation();
        return switch (operation.opType()) {
            case NLL_LOSS -> operation instanceof nllLoss op ? op.getClassDimension() : -1;
            case CROSS_ENTROPY_LOSS -> operation instanceof crossEntropyLoss op ? op.getClassDimension() : -1;
            default -> -1;
        };
    }

    private static boolean isMetalFloatingDType(DataType dtype) {
        return dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16;
    }
}
