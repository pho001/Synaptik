package graph.compile.planning.region.specialization;

import graph.model.CompiledNode;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.compile.planning.value.GraphValueRef;
import operations.Operation;

import java.util.List;

/**
 * Finds exact MATMUL -> ADD(bias) -> RELU specialization candidates in optimized regions.
 */
final class MatmulBiasReluSpecializationDetector {
    private MatmulBiasReluSpecializationDetector() {
    }

    static List<RegionSpecializationCandidate> findCandidates(
            Partition partition,
            RegionOptimizationContext context
    ) {
        if (partition == null || context == null) {
            return List.of();
        }
        if (partition.orderedNodeIds().size() == 2) {
            return findLinearReluCandidate(partition, context);
        }
        if (partition.orderedNodeIds().size() != 3) {
            return List.of();
        }
        int matmulNodeId = partition.orderedNodeIds().get(0);
        int addNodeId = partition.orderedNodeIds().get(1);
        int reluNodeId = partition.orderedNodeIds().get(2);
        CompiledNode matmul = context.compiledNode(matmulNodeId);
        CompiledNode add = context.compiledNode(addNodeId);
        CompiledNode relu = context.compiledNode(reluNodeId);
        if (opType(matmul) != Operation.OpType.MATMUL
                || opType(add) != Operation.OpType.ADD
                || opType(relu) != Operation.OpType.RELU) {
            return List.of();
        }
        if (matmul.inputIds().size() != 2
                || add.inputIds().size() != 2
                || relu.inputIds().size() != 1
                || relu.inputIds().getFirst() != addNodeId) {
            return List.of();
        }
        int biasNodeId = biasNodeId(add, matmulNodeId);
        if (biasNodeId < 0) {
            return List.of();
        }
        GraphValueRef matmulRef = GraphValueRef.node(matmulNodeId);
        GraphValueRef addRef = GraphValueRef.node(addNodeId);
        GraphValueRef reluRef = GraphValueRef.node(reluNodeId);
        if (!partition.outputValueRefs().contains(reluRef)) {
            return List.of();
        }
        if (partition.outputValueRefs().contains(matmulRef)
                || partition.outputValueRefs().contains(addRef)
                || partition.requiredMaterializedValueRefs().contains(matmulRef)
                || partition.requiredMaterializedValueRefs().contains(addRef)) {
            return List.of();
        }
        RegionSpecializationCandidate candidate = new RegionSpecializationCandidate(
                RegionSpecializationKind.MATMUL_ADD_BIAS_RELU,
                List.of(matmulNodeId, addNodeId, reluNodeId),
                List.of(
                        GraphValueRef.node(matmul.inputIds().get(0)),
                        GraphValueRef.node(matmul.inputIds().get(1)),
                        GraphValueRef.node(biasNodeId)
                ),
                reluRef,
                reluNodeId,
                "matmul-add-bias-relu:matmul=" + matmulNodeId
                        + ",add=" + addNodeId
                        + ",bias=" + biasNodeId
                        + ",relu=" + reluNodeId
        );
        return List.of(candidate);
    }

    private static List<RegionSpecializationCandidate> findLinearReluCandidate(
            Partition partition,
            RegionOptimizationContext context
    ) {
        int linearNodeId = partition.orderedNodeIds().get(0);
        int reluNodeId = partition.orderedNodeIds().get(1);
        CompiledNode linear = context.compiledNode(linearNodeId);
        CompiledNode relu = context.compiledNode(reluNodeId);
        if (opType(linear) != Operation.OpType.LINEAR || opType(relu) != Operation.OpType.RELU) {
            return List.of();
        }
        if (linear.inputIds().size() != 3
                || relu.inputIds().size() != 1
                || relu.inputIds().getFirst() != linearNodeId) {
            return List.of();
        }
        GraphValueRef linearRef = GraphValueRef.node(linearNodeId);
        GraphValueRef reluRef = GraphValueRef.node(reluNodeId);
        if (!partition.outputValueRefs().contains(reluRef)) {
            return List.of();
        }
        if (partition.outputValueRefs().contains(linearRef)
                || partition.requiredMaterializedValueRefs().contains(linearRef)) {
            return List.of();
        }
        RegionSpecializationCandidate candidate = new RegionSpecializationCandidate(
                RegionSpecializationKind.MATMUL_ADD_BIAS_RELU,
                List.of(linearNodeId, reluNodeId),
                List.of(
                        GraphValueRef.node(linear.inputIds().get(0)),
                        GraphValueRef.node(linear.inputIds().get(1)),
                        GraphValueRef.node(linear.inputIds().get(2))
                ),
                reluRef,
                reluNodeId,
                "linear-bias-relu:linear=" + linearNodeId + ",relu=" + reluNodeId
        );
        return List.of(candidate);
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
