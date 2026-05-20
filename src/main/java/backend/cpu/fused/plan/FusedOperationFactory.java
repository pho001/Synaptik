package backend.cpu.fused.plan;

import backend.cpu.fused.codegen.FusedExpressionPlan;
import backend.cpu.fused.codegen.FusedPlanBuilder;
import backend.cpu.fused.optimize.FusedCostModel;
import backend.cpu.fused.optimize.FusedDispatchFamily;
import backend.cpu.fused.optimize.FusedPrecisionResolver;
import backend.cpu.fused.optimize.FusedSignatureBuilder;

import java.util.List;

/**
 * Factory for converting lowered fused node ids into a fused operation descriptor.
 */
public final class FusedOperationFactory {
    private FusedOperationFactory() {}

    public static NodeIdResult create(
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            java.util.function.IntFunction<graph.CompiledNode> compiledNodeResolver,
            graph.compile.descriptor.CompiledTensorDescriptorIndex descriptorIndex
    ) {
        FusedExpressionPlan plan = FusedPlanBuilder.build(
                orderedNodeIds,
                externalInputNodeIds,
                compiledNodeResolver,
                descriptorIndex
        );
        int precisionMode = FusedPrecisionResolver.resolve(plan);
        boolean lowCostHint = FusedCostModel.resolveLowCostHint(plan);
        FusedDispatchFamily dispatchFamily = FusedCostModel.resolveDispatchFamily(plan);
        int dispatchComplexity = FusedCostModel.estimateDispatchComplexity(plan);
        int dispatchScale = FusedCostModel.resolveDispatchScale(dispatchComplexity);

        return new NodeIdResult(
                new FusedOperation(
                        "fused(" + orderedNodeIds.size() + ")",
                        precisionMode,
                        lowCostHint,
                        dispatchFamily,
                        FusedSignatureBuilder.buildFromPlan(plan, precisionMode),
                        dispatchScale,
                        plan
                ),
                List.copyOf(externalInputNodeIds)
        );
    }

    public record NodeIdResult(
            FusedOperation operation,
            List<Integer> runtimeInputNodeIds
    ) {}
}
