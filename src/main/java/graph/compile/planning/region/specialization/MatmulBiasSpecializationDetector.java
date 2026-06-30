package graph.compile.planning.region.specialization;

import graph.model.CompiledNode;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.compile.planning.value.GraphValueRef;
import operations.Operation;
import operations.linalg.linear;

import java.util.List;

/**
 * Finds exact MATMUL + bias epilogue specialization candidates.
 */
final class MatmulBiasSpecializationDetector {
    private MatmulBiasSpecializationDetector() {
    }

    static List<RegionSpecializationCandidate> findCandidates(
            Partition partition,
            RegionOptimizationContext context
    ) {
        if (partition == null || context == null) {
            return List.of();
        }
        return switch (partition.orderedNodeIds().size()) {
            case 1 -> findLinearBiasCandidate(partition, context);
            case 2 -> findMatmulAddBiasCandidate(partition, context);
            default -> List.of();
        };
    }

    private static List<RegionSpecializationCandidate> findLinearBiasCandidate(
            Partition partition,
            RegionOptimizationContext context
    ) {
        int linearNodeId = partition.orderedNodeIds().getFirst();
        CompiledNode linearNode = context.compiledNode(linearNodeId);
        if (opType(linearNode) != Operation.OpType.LINEAR
                || !(linearNode.operation() instanceof linear linearOp)
                || !linearOp.hasBias()
                || linearNode.inputIds().size() != 3) {
            return List.of();
        }
        GraphValueRef linearRef = GraphValueRef.node(linearNodeId);
        if (!partition.outputValueRefs().contains(linearRef)) {
            return List.of();
        }
        return List.of(new RegionSpecializationCandidate(
                RegionSpecializationKind.MATMUL_ADD_BIAS,
                List.of(linearNodeId),
                List.of(
                        GraphValueRef.node(linearNode.inputIds().get(0)),
                        GraphValueRef.node(linearNode.inputIds().get(1)),
                        GraphValueRef.node(linearNode.inputIds().get(2))
                ),
                linearRef,
                linearNodeId,
                "linear-bias:linear=" + linearNodeId
        ));
    }

    private static List<RegionSpecializationCandidate> findMatmulAddBiasCandidate(
            Partition partition,
            RegionOptimizationContext context
    ) {
        int matmulNodeId = partition.orderedNodeIds().get(0);
        int addNodeId = partition.orderedNodeIds().get(1);
        CompiledNode matmul = context.compiledNode(matmulNodeId);
        CompiledNode add = context.compiledNode(addNodeId);
        if (opType(matmul) != Operation.OpType.MATMUL || opType(add) != Operation.OpType.ADD) {
            return List.of();
        }
        if (matmul.inputIds().size() != 2 || add.inputIds().size() != 2) {
            return List.of();
        }
        int biasNodeId = biasNodeId(add, matmulNodeId);
        if (biasNodeId < 0) {
            return List.of();
        }
        GraphValueRef matmulRef = GraphValueRef.node(matmulNodeId);
        GraphValueRef addRef = GraphValueRef.node(addNodeId);
        if (!partition.outputValueRefs().contains(addRef)) {
            return List.of();
        }
        if (partition.outputValueRefs().contains(matmulRef)
                || partition.requiredMaterializedValueRefs().contains(matmulRef)) {
            return List.of();
        }
        return List.of(new RegionSpecializationCandidate(
                RegionSpecializationKind.MATMUL_ADD_BIAS,
                List.of(matmulNodeId, addNodeId),
                List.of(
                        GraphValueRef.node(matmul.inputIds().get(0)),
                        GraphValueRef.node(matmul.inputIds().get(1)),
                        GraphValueRef.node(biasNodeId)
                ),
                addRef,
                addNodeId,
                "matmul-add-bias:matmul=" + matmulNodeId + ",add=" + addNodeId + ",bias=" + biasNodeId
        ));
    }

    private static int biasNodeId(CompiledNode add, int matmulNodeId) {
        int first = add.inputIds().get(0);
        int second = add.inputIds().get(1);
        if (first == matmulNodeId && second != matmulNodeId) {
            return second;
        }
        if (second == matmulNodeId && first != matmulNodeId) {
            return first;
        }
        return -1;
    }

    private static Operation.OpType opType(CompiledNode node) {
        return node == null || node.operation() == null ? null : node.operation().opType();
    }
}
