package backend.cpu.fused.plan;

import backend.lowering.LoweredExecutionUnit;
import graph.CompiledNode;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.function.IntFunction;

/**
 * Builds the lowered artifact used to prepare fused CPU execution units.
 */
public final class LoweredFusedOperationBuilder {
    private LoweredFusedOperationBuilder() {
    }

    /**
     * Returns an existing fused artifact or builds one from the lowered unit node order.
     */
    public static FusedOperationPreparation build(
            LoweredExecutionUnit loweredUnit,
            IntFunction<CompiledNode> compiledNodeResolver
    ) {
        Objects.requireNonNull(loweredUnit, "loweredUnit cannot be null");
        if (loweredUnit.artifact() instanceof FusedOperationPreparation preparation) {
            return preparation;
        }
        return build(loweredUnit.orderedNodeIds(), compiledNodeResolver);
    }

    /**
     * Builds a fused artifact from ordered node ids and a compiled-node resolver.
     */
    public static FusedOperationPreparation build(
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver
    ) {
        Objects.requireNonNull(orderedNodeIds, "orderedNodeIds cannot be null");
        Objects.requireNonNull(compiledNodeResolver, "compiledNodeResolver cannot be null");
        List<Integer> safeOrderedNodeIds = List.copyOf(orderedNodeIds);
        if (safeOrderedNodeIds.isEmpty()) {
            throw new IllegalArgumentException("lowered fused unit must contain at least one node");
        }

        List<Tensor> cluster = clusterTensors(safeOrderedNodeIds, compiledNodeResolver);
        List<Tensor> externalInputs = externalInputTensors(safeOrderedNodeIds, compiledNodeResolver);
        FusedOperationFactory.Result fused = FusedOperationFactory.create(
                cluster,
                cluster.getLast(),
                externalInputs
        );
        return new FusedOperationPreparation(fused.operation(), fused.runtimeInputs());
    }

    private static List<Tensor> clusterTensors(
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver
    ) {
        List<Tensor> cluster = new ArrayList<>(orderedNodeIds.size());
        for (int nodeId : orderedNodeIds) {
            cluster.add(requireNode(compiledNodeResolver, nodeId, "lowered fused unit").semanticTensor());
        }
        return List.copyOf(cluster);
    }

    private static List<Tensor> externalInputTensors(
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver
    ) {
        LinkedHashSet<Integer> chainNodeIds = new LinkedHashSet<>(orderedNodeIds);
        LinkedHashSet<Tensor> externalInputs = new LinkedHashSet<>();
        for (int nodeId : orderedNodeIds) {
            CompiledNode node = requireNode(compiledNodeResolver, nodeId, "lowered fused unit");
            for (int inputNodeId : node.inputIds()) {
                if (chainNodeIds.contains(inputNodeId)) {
                    continue;
                }
                externalInputs.add(requireNode(compiledNodeResolver, inputNodeId, "lowered fused unit input").semanticTensor());
            }
        }
        return List.copyOf(externalInputs);
    }

    private static CompiledNode requireNode(
            IntFunction<CompiledNode> compiledNodeResolver,
            int nodeId,
            String context
    ) {
        CompiledNode node = compiledNodeResolver.apply(nodeId);
        if (node == null) {
            throw new IllegalStateException("Missing compiled node for " + context + " nodeId=" + nodeId);
        }
        return node;
    }
}
