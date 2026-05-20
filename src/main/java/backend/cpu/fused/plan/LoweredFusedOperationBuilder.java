package backend.cpu.fused.plan;

import backend.lowering.LoweredExecutionUnit;
import backend.lowering.region.CpuFusedRegionPayload;
import backend.lowering.region.RegionExecutionPlan;
import graph.CompiledNode;

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
            IntFunction<CompiledNode> compiledNodeResolver,
            graph.compile.descriptor.CompiledTensorDescriptorIndex descriptorIndex
    ) {
        Objects.requireNonNull(loweredUnit, "loweredUnit cannot be null");
        if (loweredUnit.artifact() instanceof FusedOperationPreparation preparation) {
            return preparation;
        }
        if (loweredUnit.artifact() instanceof RegionExecutionPlan plan
                && plan.backendPayload() instanceof CpuFusedRegionPayload payload) {
            return payload.requirePreparation(FusedOperationPreparation.class);
        }
        return build(loweredUnit.orderedNodeIds(), compiledNodeResolver, descriptorIndex);
    }

    /**
     * Builds a fused artifact from ordered node ids and a compiled-node resolver.
     */
    public static FusedOperationPreparation build(
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver,
            graph.compile.descriptor.CompiledTensorDescriptorIndex descriptorIndex
    ) {
        Objects.requireNonNull(orderedNodeIds, "orderedNodeIds cannot be null");
        Objects.requireNonNull(compiledNodeResolver, "compiledNodeResolver cannot be null");
        Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        List<Integer> safeOrderedNodeIds = List.copyOf(orderedNodeIds);
        if (safeOrderedNodeIds.isEmpty()) {
            throw new IllegalArgumentException("lowered fused unit must contain at least one node");
        }

        List<Integer> externalInputNodeIds = externalInputNodeIds(safeOrderedNodeIds, compiledNodeResolver);
        FusedOperationFactory.NodeIdResult fused = FusedOperationFactory.create(
                safeOrderedNodeIds,
                externalInputNodeIds,
                compiledNodeResolver,
                descriptorIndex
        );
        return new FusedOperationPreparation(fused.operation(), fused.runtimeInputNodeIds());
    }

    private static List<Integer> externalInputNodeIds(
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver
    ) {
        LinkedHashSet<Integer> chainNodeIds = new LinkedHashSet<>(orderedNodeIds);
        LinkedHashSet<Integer> externalInputs = new LinkedHashSet<>();
        for (int nodeId : orderedNodeIds) {
            CompiledNode node = requireNode(compiledNodeResolver, nodeId, "lowered fused unit");
            for (int inputNodeId : node.inputIds()) {
                if (chainNodeIds.contains(inputNodeId)) {
                    continue;
                }
                requireNode(compiledNodeResolver, inputNodeId, "lowered fused unit input");
                externalInputs.add(inputNodeId);
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
