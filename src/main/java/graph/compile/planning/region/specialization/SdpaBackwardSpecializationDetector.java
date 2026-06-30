package graph.compile.planning.region.specialization;

import graph.model.CompiledNode;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.compile.planning.value.GraphValueRef;
import operations.Operation;
import operations.elementwise.unary.mulScalar;
import operations.layout.permute;
import operations.reduction.sum;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Finds isolated canonical scaled-dot-product-attention backward primitive DAG candidates.
 */
final class SdpaBackwardSpecializationDetector {
    private static final int UNKNOWN_NODE_ID = -1;

    private SdpaBackwardSpecializationDetector() {
    }

    static List<RegionSpecializationCandidate> findCandidates(
            Partition partition,
            RegionOptimizationContext context
    ) {
        if (partition == null || context == null || partition.orderedNodeIds().isEmpty()) {
            return List.of();
        }
        Set<Integer> partitionNodeIds = Set.copyOf(partition.orderedNodeIds());
        LinkedHashSet<RegionSpecializationCandidate> out = new LinkedHashSet<>();
        for (int terminalNodeId : partition.orderedNodeIds()) {
            if (!partition.outputValueRefs().contains(GraphValueRef.node(terminalNodeId))) {
                continue;
            }
            RegionSpecializationCandidate candidate = matchValueGradient(
                    terminalNodeId,
                    partition,
                    partitionNodeIds,
                    context
            );
            if (candidate == null) {
                candidate = matchQueryOrKeyGradient(terminalNodeId, partition, partitionNodeIds, context);
            }
            if (candidate != null) {
                out.add(candidate);
            }
        }
        return List.copyOf(out);
    }

    private static RegionSpecializationCandidate matchValueGradient(
            int terminalNodeId,
            Partition partition,
            Set<Integer> partitionNodeIds,
            RegionOptimizationContext context
    ) {
        CompiledNode terminal = context.compiledNode(terminalNodeId);
        if (opType(terminal) != Operation.OpType.MATMUL || terminal.inputIds().size() != 2) {
            return null;
        }
        int weightsTransposeNodeId = terminal.inputIds().get(0);
        int outGradNodeId = terminal.inputIds().get(1);
        int weightsNodeId = transposedInputNodeId(weightsTransposeNodeId, context);
        if (weightsNodeId < 0) {
            return null;
        }
        if (opType(context.compiledNode(weightsNodeId)) != Operation.OpType.DIV) {
            return null;
        }
        ArrayList<Integer> candidateNodeIds = new ArrayList<>();
        candidateNodeIds.add(weightsTransposeNodeId);
        candidateNodeIds.add(terminalNodeId);
        if (!partitionNodeIds.containsAll(candidateNodeIds)
                || hasUnsafeIntermediateBoundary(candidateNodeIds, terminalNodeId, partition)) {
            return null;
        }
        ForwardMetadata forward = forwardMetadata(weightsNodeId, context);
        SdpaBackwardSpecializationPayload payload = new SdpaBackwardSpecializationPayload(
                SdpaBackwardOutputKind.VALUE,
                forward.scale(),
                forward.hasMask(),
                weightsNodeId,
                outGradNodeId,
                forward.queryNodeId(),
                forward.keyNodeId(),
                UNKNOWN_NODE_ID,
                forward.maskNodeId()
        );
        return candidate(
                partition,
                candidateNodeIds,
                terminalNodeId,
                payload,
                "sdpa-backward:value,weights=" + weightsNodeId
                        + ",outGrad=" + outGradNodeId
                        + ",weightsT=" + weightsTransposeNodeId
        );
    }

    private static RegionSpecializationCandidate matchQueryOrKeyGradient(
            int terminalNodeId,
            Partition partition,
            Set<Integer> partitionNodeIds,
            RegionOptimizationContext context
    ) {
        CompiledNode terminal = context.compiledNode(terminalNodeId);
        if (opType(terminal) != Operation.OpType.MATMUL || terminal.inputIds().size() != 2) {
            return null;
        }
        int lhsNodeId = terminal.inputIds().get(0);
        int rhsNodeId = terminal.inputIds().get(1);
        int dScoresNodeId = lhsNodeId;
        int dScoresTransposeNodeId = UNKNOWN_NODE_ID;
        SdpaBackwardOutputKind outputKind = SdpaBackwardOutputKind.QUERY;
        int queryNodeId = UNKNOWN_NODE_ID;
        int keyNodeId = rhsNodeId;
        if (opType(context.compiledNode(lhsNodeId)) == Operation.OpType.PERMUTE) {
            dScoresTransposeNodeId = lhsNodeId;
            dScoresNodeId = transposedInputNodeId(lhsNodeId, context);
            if (dScoresNodeId < 0) {
                return null;
            }
            outputKind = SdpaBackwardOutputKind.KEY;
            queryNodeId = rhsNodeId;
            keyNodeId = UNKNOWN_NODE_ID;
        }

        ScoreChain scoreChain = matchScoreChain(dScoresNodeId, context);
        if (scoreChain == null) {
            return null;
        }
        ArrayList<Integer> candidateNodeIds = new ArrayList<>();
        candidateNodeIds.add(scoreChain.valueTransposeNodeId());
        candidateNodeIds.add(scoreChain.dWeightsNodeId());
        candidateNodeIds.add(scoreChain.weightsMulNodeId());
        candidateNodeIds.add(scoreChain.dotNodeId());
        candidateNodeIds.add(scoreChain.subtractNodeId());
        candidateNodeIds.add(scoreChain.dScoresCoreNodeId());
        if (scoreChain.whereNodeId() >= 0) {
            candidateNodeIds.add(scoreChain.whereNodeId());
        }
        if (scoreChain.scaleNodeId() >= 0) {
            candidateNodeIds.add(scoreChain.scaleNodeId());
        }
        if (dScoresTransposeNodeId >= 0) {
            candidateNodeIds.add(dScoresTransposeNodeId);
        }
        candidateNodeIds.add(terminalNodeId);
        if (!partitionNodeIds.containsAll(candidateNodeIds)
                || hasUnsafeIntermediateBoundary(candidateNodeIds, terminalNodeId, partition)) {
            return null;
        }

        if (outputKind == SdpaBackwardOutputKind.QUERY) {
            ForwardMetadata forward = forwardMetadata(scoreChain.weightsNodeId(), context);
            if (forward.keyNodeId() >= 0 && forward.keyNodeId() != keyNodeId) {
                return null;
            }
            queryNodeId = forward.queryNodeId();
        } else {
            ForwardMetadata forward = forwardMetadata(scoreChain.weightsNodeId(), context);
            if (forward.queryNodeId() >= 0 && forward.queryNodeId() != queryNodeId) {
                return null;
            }
            keyNodeId = forward.keyNodeId();
        }

        SdpaBackwardSpecializationPayload payload = new SdpaBackwardSpecializationPayload(
                outputKind,
                scoreChain.scale(),
                scoreChain.hasMask(),
                scoreChain.weightsNodeId(),
                scoreChain.outGradNodeId(),
                queryNodeId,
                keyNodeId,
                scoreChain.valueNodeId(),
                scoreChain.maskNodeId()
        );
        return candidate(
                partition,
                candidateNodeIds,
                terminalNodeId,
                payload,
                "sdpa-backward:" + outputKind.name().toLowerCase(java.util.Locale.ROOT)
                        + ",weights=" + scoreChain.weightsNodeId()
                        + ",outGrad=" + scoreChain.outGradNodeId()
                        + ",value=" + scoreChain.valueNodeId()
                        + ",scale=" + scoreChain.scale()
                        + ",hasMask=" + scoreChain.hasMask()
        );
    }

    private static ScoreChain matchScoreChain(int dScoresNodeId, RegionOptimizationContext context) {
        int cursor = dScoresNodeId;
        int scaleNodeId = UNKNOWN_NODE_ID;
        double scale = 1.0d;
        CompiledNode cursorNode = context.compiledNode(cursor);
        if (opType(cursorNode) == Operation.OpType.MUL_SCALAR && cursorNode.inputIds().size() == 1) {
            if (!(cursorNode.operation() instanceof mulScalar scaleOp) || !(scaleOp.getScalar() > 0.0d)) {
                return null;
            }
            scaleNodeId = cursor;
            scale = scaleOp.getScalar();
            cursor = cursorNode.inputIds().getFirst();
            cursorNode = context.compiledNode(cursor);
        }

        int whereNodeId = UNKNOWN_NODE_ID;
        int maskNodeId = UNKNOWN_NODE_ID;
        boolean hasMask = false;
        if (opType(cursorNode) == Operation.OpType.WHERE && cursorNode.inputIds().size() == 3) {
            whereNodeId = cursor;
            maskNodeId = cursorNode.inputIds().get(0);
            cursor = cursorNode.inputIds().get(1);
            cursorNode = context.compiledNode(cursor);
            hasMask = true;
        }

        CompiledNode dScoresCore = cursorNode;
        if (opType(dScoresCore) != Operation.OpType.MUL || dScoresCore.inputIds().size() != 2) {
            return null;
        }
        MulInputs dScoresInputs = splitMulInputs(dScoresCore, context, Operation.OpType.SUB);
        if (dScoresInputs == null) {
            return null;
        }
        int weightsNodeId = dScoresInputs.otherNodeId();
        int subtractNodeId = dScoresInputs.targetNodeId();
        CompiledNode subtract = context.compiledNode(subtractNodeId);
        if (opType(subtract) != Operation.OpType.SUB || subtract.inputIds().size() != 2) {
            return null;
        }
        int dWeightsNodeId = subtract.inputIds().get(0);
        int dotNodeId = subtract.inputIds().get(1);
        CompiledNode dot = context.compiledNode(dotNodeId);
        if (opType(dot) != Operation.OpType.SUM
                || dot.inputIds().size() != 1
                || !(dot.operation() instanceof sum sumOp)
                || !sumOp.keepDims()) {
            return null;
        }
        CompiledNode dWeights = context.compiledNode(dWeightsNodeId);
        if (opType(dWeights) != Operation.OpType.MATMUL || dWeights.inputIds().size() != 2) {
            return null;
        }
        int outGradNodeId = dWeights.inputIds().get(0);
        int valueTransposeNodeId = dWeights.inputIds().get(1);
        int valueNodeId = transposedInputNodeId(valueTransposeNodeId, context);
        if (valueNodeId < 0) {
            return null;
        }
        CompiledNode weightsMul = context.compiledNode(dot.inputIds().getFirst());
        if (opType(weightsMul) != Operation.OpType.MUL || weightsMul.inputIds().size() != 2) {
            return null;
        }
        if (!sameInputs(weightsMul, dWeightsNodeId, weightsNodeId)) {
            return null;
        }
        int axis = context.compiledNode(weightsNodeId) == null ? -1 : context.compiledNode(weightsNodeId).shape().length - 1;
        if (axis < 0 || sumOp.getDimension() != axis) {
            return null;
        }
        return new ScoreChain(
                weightsNodeId,
                outGradNodeId,
                valueNodeId,
                maskNodeId,
                valueTransposeNodeId,
                dWeightsNodeId,
                weightsMul.id(),
                dotNodeId,
                subtractNodeId,
                dScoresCore.id(),
                whereNodeId,
                scaleNodeId,
                scale,
                hasMask
        );
    }

    private static RegionSpecializationCandidate candidate(
            Partition partition,
            List<Integer> candidateNodeIds,
            int terminalNodeId,
            SdpaBackwardSpecializationPayload payload,
            String summary
    ) {
        LinkedHashSet<GraphValueRef> inputs = new LinkedHashSet<>();
        inputs.add(GraphValueRef.node(payload.weightsNodeId()));
        inputs.add(GraphValueRef.node(payload.outGradNodeId()));
        switch (payload.outputKind()) {
            case QUERY -> {
                if (payload.keyNodeId() >= 0) {
                    inputs.add(GraphValueRef.node(payload.keyNodeId()));
                }
                if (payload.valueNodeId() >= 0) {
                    inputs.add(GraphValueRef.node(payload.valueNodeId()));
                }
            }
            case KEY -> {
                if (payload.queryNodeId() >= 0) {
                    inputs.add(GraphValueRef.node(payload.queryNodeId()));
                }
                if (payload.valueNodeId() >= 0) {
                    inputs.add(GraphValueRef.node(payload.valueNodeId()));
                }
            }
            case VALUE -> {
            }
        }
        if (payload.hasMask() && payload.outputKind() != SdpaBackwardOutputKind.VALUE) {
            inputs.add(GraphValueRef.node(payload.maskNodeId()));
        }
        List<Integer> ordered = partition.orderedNodeIds().stream()
                .filter(candidateNodeIds::contains)
                .toList();
        return new RegionSpecializationCandidate(
                RegionSpecializationKind.SDPA_BACKWARD,
                ordered,
                List.copyOf(inputs),
                GraphValueRef.node(terminalNodeId),
                terminalNodeId,
                summary,
                payload
        );
    }

    private static ForwardMetadata forwardMetadata(int weightsNodeId, RegionOptimizationContext context) {
        CompiledNode weights = context.compiledNode(weightsNodeId);
        if (opType(weights) != Operation.OpType.DIV || weights.inputIds().isEmpty()) {
            return ForwardMetadata.unknown();
        }
        CompiledNode exp = context.compiledNode(weights.inputIds().getFirst());
        if (opType(exp) != Operation.OpType.EXP || exp.inputIds().size() != 1) {
            return ForwardMetadata.unknown();
        }
        CompiledNode shifted = context.compiledNode(exp.inputIds().getFirst());
        if (opType(shifted) != Operation.OpType.SUB || shifted.inputIds().isEmpty()) {
            return ForwardMetadata.unknown();
        }
        int logitsNodeId = shifted.inputIds().getFirst();
        CompiledNode logits = context.compiledNode(logitsNodeId);
        boolean hasMask = false;
        int maskNodeId = UNKNOWN_NODE_ID;
        int scaledNodeId = logitsNodeId;
        if (opType(logits) == Operation.OpType.WHERE && logits.inputIds().size() == 3) {
            hasMask = true;
            maskNodeId = logits.inputIds().get(0);
            scaledNodeId = logits.inputIds().get(1);
        }
        double scale = 1.0d;
        CompiledNode scaled = context.compiledNode(scaledNodeId);
        int scoresNodeId = scaledNodeId;
        if (opType(scaled) == Operation.OpType.MUL_SCALAR && scaled.inputIds().size() == 1
                && scaled.operation() instanceof mulScalar scalar) {
            scale = scalar.getScalar();
            scoresNodeId = scaled.inputIds().getFirst();
        }
        CompiledNode scores = context.compiledNode(scoresNodeId);
        if (opType(scores) != Operation.OpType.MATMUL || scores.inputIds().size() != 2) {
            return new ForwardMetadata(scale, hasMask, UNKNOWN_NODE_ID, UNKNOWN_NODE_ID, maskNodeId);
        }
        int queryNodeId = scores.inputIds().get(0);
        int keyNodeId = transposedInputNodeId(scores.inputIds().get(1), context);
        return new ForwardMetadata(scale, hasMask, queryNodeId, keyNodeId, maskNodeId);
    }

    private static int transposedInputNodeId(int nodeId, RegionOptimizationContext context) {
        CompiledNode node = context.compiledNode(nodeId);
        if (opType(node) != Operation.OpType.PERMUTE
                || node.inputIds().size() != 1
                || !(node.operation() instanceof permute permuteOp)
                || !isLastTwoAxesTranspose(permuteOp.getAxes())) {
            return UNKNOWN_NODE_ID;
        }
        return node.inputIds().getFirst();
    }

    private static boolean isLastTwoAxesTranspose(int[] axes) {
        if (axes == null || axes.length < 2) {
            return false;
        }
        for (int i = 0; i < axes.length - 2; i++) {
            if (axes[i] != i) {
                return false;
            }
        }
        return axes[axes.length - 2] == axes.length - 1
                && axes[axes.length - 1] == axes.length - 2;
    }

    private static MulInputs splitMulInputs(
            CompiledNode node,
            RegionOptimizationContext context,
            Operation.OpType targetType
    ) {
        int first = node.inputIds().get(0);
        int second = node.inputIds().get(1);
        if (opType(context.compiledNode(first)) == targetType && opType(context.compiledNode(second)) != targetType) {
            return new MulInputs(first, second);
        }
        if (opType(context.compiledNode(second)) == targetType && opType(context.compiledNode(first)) != targetType) {
            return new MulInputs(second, first);
        }
        return null;
    }

    private static boolean sameInputs(CompiledNode node, int firstExpected, int secondExpected) {
        return node.inputIds().size() == 2
                && ((node.inputIds().get(0) == firstExpected && node.inputIds().get(1) == secondExpected)
                || (node.inputIds().get(0) == secondExpected && node.inputIds().get(1) == firstExpected));
    }

    private static boolean hasUnsafeIntermediateBoundary(
            List<Integer> candidateNodeIds,
            int terminalNodeId,
            Partition partition
    ) {
        for (int nodeId : candidateNodeIds) {
            GraphValueRef ref = GraphValueRef.node(nodeId);
            if (nodeId != terminalNodeId && (partition.outputValueRefs().contains(ref)
                    || partition.requiredMaterializedValueRefs().contains(ref))) {
                return true;
            }
        }
        return false;
    }

    private static Operation.OpType opType(CompiledNode node) {
        return node == null || node.operation() == null ? null : node.operation().opType();
    }

    private record ForwardMetadata(
            double scale,
            boolean hasMask,
            int queryNodeId,
            int keyNodeId,
            int maskNodeId
    ) {
        static ForwardMetadata unknown() {
            return new ForwardMetadata(1.0d, false, UNKNOWN_NODE_ID, UNKNOWN_NODE_ID, UNKNOWN_NODE_ID);
        }
    }

    private record ScoreChain(
            int weightsNodeId,
            int outGradNodeId,
            int valueNodeId,
            int maskNodeId,
            int valueTransposeNodeId,
            int dWeightsNodeId,
            int weightsMulNodeId,
            int dotNodeId,
            int subtractNodeId,
            int dScoresCoreNodeId,
            int whereNodeId,
            int scaleNodeId,
            double scale,
            boolean hasMask
    ) {
    }

    private record MulInputs(
            int targetNodeId,
            int otherNodeId
    ) {
    }
}
