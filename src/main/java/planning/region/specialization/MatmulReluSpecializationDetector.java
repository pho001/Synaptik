package planning.region.specialization;

import graph.model.CompiledNode;
import planning.partition.Partition;
import planning.region.RegionPlanningContext;
import planning.value.GraphValueRef;
import operations.Operation;

import java.util.List;

/**
 * Finds exact MATMUL -> RELU specialization candidates in planned regions.
 */
final class MatmulReluSpecializationDetector {
    private MatmulReluSpecializationDetector() {
    }

    static List<RegionSpecializationCandidate> findCandidates(
            Partition partition,
            RegionPlanningContext context
    ) {
        if (partition == null || context == null || partition.orderedNodeIds().size() != 2) {
            return List.of();
        }
        int matmulNodeId = partition.orderedNodeIds().get(0);
        int reluNodeId = partition.orderedNodeIds().get(1);
        CompiledNode matmul = context.compiledNode(matmulNodeId);
        CompiledNode relu = context.compiledNode(reluNodeId);
        if (opType(matmul) != Operation.OpType.MATMUL || opType(relu) != Operation.OpType.RELU) {
            return List.of();
        }
        if (matmul.inputIds().size() != 2 || relu.inputIds().size() != 1 || relu.inputIds().getFirst() != matmulNodeId) {
            return List.of();
        }
        GraphValueRef matmulRef = GraphValueRef.node(matmulNodeId);
        GraphValueRef reluRef = GraphValueRef.node(reluNodeId);
        if (!partition.outputValueRefs().contains(reluRef)) {
            return List.of();
        }
        if (partition.outputValueRefs().contains(matmulRef)
                || partition.requiredMaterializedValueRefs().contains(matmulRef)) {
            return List.of();
        }
        RegionSpecializationCandidate candidate = new RegionSpecializationCandidate(
                RegionSpecializationKind.MATMUL_RELU,
                List.of(matmulNodeId, reluNodeId),
                List.of(GraphValueRef.node(matmul.inputIds().get(0)), GraphValueRef.node(matmul.inputIds().get(1))),
                reluRef,
                reluNodeId,
                "matmul-relu:matmul=" + matmulNodeId + ",relu=" + reluNodeId
        );
        return List.of(candidate);
    }

    private static Operation.OpType opType(CompiledNode node) {
        return node == null || node.operation() == null ? null : node.operation().opType();
    }
}
