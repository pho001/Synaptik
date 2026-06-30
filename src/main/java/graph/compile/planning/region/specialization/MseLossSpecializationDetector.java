package graph.compile.planning.region.specialization;

import graph.model.CompiledNode;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.compile.planning.value.GraphValueRef;
import operations.Operation;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Finds first-pass MSE loss specialization candidates in optimized regions.
 */
final class MseLossSpecializationDetector {
    private MseLossSpecializationDetector() {
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
            RegionSpecializationCandidate candidate = matchTerminalReduction(
                    terminalNodeId,
                    partitionNodeIds,
                    partition,
                    context
            );
            if (candidate != null) {
                out.add(candidate);
            }
        }
        return List.copyOf(out);
    }

    private static RegionSpecializationCandidate matchTerminalReduction(
            int terminalNodeId,
            Set<Integer> partitionNodeIds,
            Partition partition,
            RegionOptimizationContext context
    ) {
        ArrayList<Integer> reductions = new ArrayList<>();
        int cursor = terminalNodeId;
        while (true) {
            CompiledNode node = context.compiledNode(cursor);
            if (!isReduction(node) || node.inputIds().size() != 1) {
                return null;
            }
            reductions.add(cursor);
            int inputNodeId = node.inputIds().getFirst();
            CompiledNode input = context.compiledNode(inputNodeId);
            if (isReduction(input)) {
                cursor = inputNodeId;
                continue;
            }
            RegionSpecializationCandidate candidate = matchSquare(
                    inputNodeId,
                    reductions.reversed(),
                    partitionNodeIds,
                    partition,
                    context
            );
            return candidate;
        }
    }

    private static RegionSpecializationCandidate matchSquare(
            int squareNodeId,
            List<Integer> reductionsInOrder,
            Set<Integer> partitionNodeIds,
            Partition partition,
            RegionOptimizationContext context
    ) {
        CompiledNode square = context.compiledNode(squareNodeId);
        if (opType(square) != Operation.OpType.MUL || square.inputIds().size() != 2) {
            return null;
        }
        if (!hasHomogeneousReductions(reductionsInOrder, context)) {
            return null;
        }
        int diffNodeId = square.inputIds().get(0);
        if (square.inputIds().get(1) != diffNodeId) {
            return null;
        }
        CompiledNode diff = context.compiledNode(diffNodeId);
        if (opType(diff) != Operation.OpType.SUB || diff.inputIds().size() != 2) {
            return null;
        }
        ArrayList<Integer> candidateNodeIds = new ArrayList<>();
        candidateNodeIds.add(diffNodeId);
        candidateNodeIds.add(squareNodeId);
        candidateNodeIds.addAll(reductionsInOrder);
        if (!partitionNodeIds.containsAll(candidateNodeIds)) {
            return null;
        }
        for (int nodeId : candidateNodeIds) {
            GraphValueRef ref = GraphValueRef.node(nodeId);
            boolean terminal = nodeId == reductionsInOrder.getLast();
            if (!terminal && (partition.outputValueRefs().contains(ref)
                    || partition.requiredMaterializedValueRefs().contains(ref))) {
                return null;
            }
        }
        LinkedHashSet<GraphValueRef> inputs = new LinkedHashSet<>();
        Set<Integer> candidateNodeSet = Set.copyOf(candidateNodeIds);
        for (int nodeId : candidateNodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                return null;
            }
            for (int inputId : node.inputIds()) {
                if (!candidateNodeSet.contains(inputId)) {
                    inputs.add(GraphValueRef.node(inputId));
                }
            }
        }
        int terminalNodeId = reductionsInOrder.getLast();
        return new RegionSpecializationCandidate(
                RegionSpecializationKind.MSE_LOSS,
                candidateNodeIds,
                List.copyOf(inputs),
                GraphValueRef.node(terminalNodeId),
                terminalNodeId,
                "mse-loss:diff=" + diffNodeId
                        + ",square=" + squareNodeId
                        + ",reductions=" + reductionsInOrder
        );
    }

    private static boolean isReduction(CompiledNode node) {
        Operation.OpType opType = opType(node);
        return opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN;
    }

    private static boolean hasHomogeneousReductions(
            List<Integer> reductionsInOrder,
            RegionOptimizationContext context
    ) {
        Operation.OpType first = null;
        for (int reductionNodeId : reductionsInOrder) {
            Operation.OpType current = opType(context.compiledNode(reductionNodeId));
            if (current != Operation.OpType.SUM && current != Operation.OpType.MEAN) {
                return false;
            }
            if (first == null) {
                first = current;
            } else if (first != current) {
                return false;
            }
        }
        return first != null;
    }

    private static Operation.OpType opType(CompiledNode node) {
        return node == null || node.operation() == null ? null : node.operation().opType();
    }
}
