package backend.cpu.fused.plan;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedIrBuilder;
import backend.cpu.fused.numeric.FusedApproximationContract;
import backend.cpu.fused.numeric.FusedNumericContract;
import backend.cpu.fused.numeric.FusedNumericContractResolver;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.region.CpuFusedRegionPayload;
import backend.lowering.region.RegionExecutionPlan;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.function.IntFunction;

/**
 * Builds fused CPU operation descriptors from lowered graph node order.
 */
public final class FusedOperationBuilder {
    private FusedOperationBuilder() {
    }

    public static FusedOperationPreparation build(
            LoweredExecutionUnit loweredUnit,
            IntFunction<CompiledNode> compiledNodeResolver,
            CompiledTensorDescriptorIndex descriptorIndex
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

    public static FusedOperationPreparation build(
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        Objects.requireNonNull(orderedNodeIds, "orderedNodeIds cannot be null");
        Objects.requireNonNull(compiledNodeResolver, "compiledNodeResolver cannot be null");
        Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        List<Integer> safeOrderedNodeIds = List.copyOf(orderedNodeIds);
        if (safeOrderedNodeIds.isEmpty()) {
            throw new IllegalArgumentException("lowered fused unit must contain at least one node");
        }

        List<Integer> externalInputNodeIds = externalInputNodeIds(safeOrderedNodeIds, compiledNodeResolver);
        FusedExpressionPlan plan = FusedIrBuilder.build(
                safeOrderedNodeIds,
                externalInputNodeIds,
                compiledNodeResolver,
                descriptorIndex
        );
        FusedNumericContract numericContract = FusedNumericContractResolver.resolve(plan);
        boolean lowCostHint = FusedDispatchPlanner.resolveLowCostHint(plan);
        FusedDispatchFamily dispatchFamily = FusedDispatchPlanner.resolveDispatchFamily(plan);

        FusedOperation operation = new FusedOperation(
                "fused(" + safeOrderedNodeIds.size() + ")",
                numericContract,
                FusedApproximationContract.STRICT,
                lowCostHint,
                dispatchFamily,
                FusedSignatureBuilder.buildFromPlan(plan, numericContract, FusedApproximationContract.STRICT),
                plan
        );
        return new FusedOperationPreparation(operation, externalInputNodeIds);
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
