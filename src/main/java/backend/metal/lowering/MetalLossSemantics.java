package backend.metal.lowering;

import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import operations.loss.crossEntropyLoss;
import operations.loss.nllLoss;
import tensor.DataType;

import java.util.Arrays;

final class MetalLossSemantics {
    private MetalLossSemantics() {
    }

    static boolean isDenseLoss(Operation.OpType opType) {
        return opType == Operation.OpType.NLL_LOSS
                || opType == Operation.OpType.CROSS_ENTROPY_LOSS;
    }

    static String unsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
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
        if (node.dataType() != DataType.FLOAT32
                || scores.dataType() != DataType.FLOAT32
                || targets.dataType() != DataType.FLOAT32) {
            return "UNSUPPORTED_DTYPE: GPU_METAL dense " + opType + " Phase 37 contract is scoped to FLOAT32 output, scores/log-probabilities, and dense targets";
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
        return "DAG_PRIMITIVE_UNSUPPORTED: GPU_METAL dense " + opType
                + " contract locked for FLOAT32 dense rank 1..4 mean-reduced class-axis loss; "
                + "native/lowered execution is pending Phase 37-02; target=loss_dense_small target=transformer_block_hot_path";
    }

    private static int classAxis(CompiledNode node) {
        Operation operation = node.operation();
        return switch (operation.opType()) {
            case NLL_LOSS -> operation instanceof nllLoss op ? op.getClassDimension() : -1;
            case CROSS_ENTROPY_LOSS -> operation instanceof crossEntropyLoss op ? op.getClassDimension() : -1;
            default -> -1;
        };
    }
}
