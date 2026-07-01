package planning.partition.execution;

import graph.model.CompiledNode;
import planning.partition.Partition;
import planning.partition.PartitionTarget;
import planning.value.GraphValueRef;
import operations.Operation;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

final class ElementwiseFusionPlanner {
    private ElementwiseFusionPlanner() {
    }

    static boolean shouldFuseWholePartition(Partition partition, PartitionExecutionPlanningContext context) {
        if (partition == null || partition.orderedNodeIds().size() < 2) {
            return false;
        }
        if (partition.outputValueRefs().size() != 1) {
            return false;
        }
        if (partition.target() == PartitionTarget.NONE) {
            return false;
        }
        for (int nodeId : partition.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (!isSubchainFusable(node)) {
                return false;
            }
        }
        return true;
    }

    static List<GraphValueRef> unitOutputsForChain(
            Partition partition,
            List<Integer> chain,
            PartitionExecutionPlanningContext context
    ) {
        Set<Integer> chainSet = Set.copyOf(chain);
        LinkedHashSet<GraphValueRef> outputRefs = new LinkedHashSet<>();
        for (int nodeId : chain) {
            boolean escapesUnit = partition.outputValueRefs().contains(GraphValueRef.node(nodeId));
            if (!escapesUnit) {
                for (int candidateId : partition.orderedNodeIds()) {
                    if (chainSet.contains(candidateId)) {
                        continue;
                    }
                    CompiledNode candidate = context.compiledNode(candidateId);
                    if (candidate != null && candidate.inputIds().contains(nodeId)) {
                        escapesUnit = true;
                        break;
                    }
                }
            }
            if (escapesUnit) {
                outputRefs.add(GraphValueRef.node(nodeId));
            }
        }
        if (outputRefs.isEmpty()) {
            outputRefs.add(GraphValueRef.node(chain.getLast()));
        }
        return List.copyOf(outputRefs);
    }

    static boolean isSubchainFusable(CompiledNode node) {
        return node != null
                && node.operation() != null
                && node.operation().opType() != null
                && node.operation().isFusable();
    }

    static boolean consumesUnitOutput(CompiledNode candidate, List<Integer> chain) {
        if (candidate == null || candidate.inputIds().isEmpty()) {
            return false;
        }
        int lastNodeId = chain.getLast();
        return candidate.inputIds().contains(lastNodeId);
    }
}
