package backend.cpu.fused.plan;

import backend.cpu.fused.codegen.FusedExpressionPlan;
import backend.cpu.fused.codegen.FusedPlanBuilder;
import backend.cpu.fused.optimize.FusedAccessResolver;
import backend.cpu.fused.optimize.FusedCostModel;
import backend.cpu.fused.optimize.FusedDispatchFamily;
import backend.cpu.fused.optimize.FusedPrecisionResolver;
import backend.cpu.fused.optimize.FusedSignatureBuilder;
import tensor.Tensor;

import java.util.List;

/**
 * Factory for converting a fused tensor cluster into a fused operation descriptor.
 */
public final class FusedOperationFactory {
    private FusedOperationFactory() {}

    /**
     * Builds a fused operation and its runtime input bindings from a tensor cluster.
     */
    public static Result create(
            List<Tensor> cluster,
            Tensor root,
            List<Tensor> externalInputsInOrder
    ) {
        List<Tensor> runtimeInputs = resolveRuntimeInputs(externalInputsInOrder);
        FusedExpressionPlan plan = FusedPlanBuilder.build(cluster, externalInputsInOrder, root);

        int precisionMode = FusedPrecisionResolver.resolve(cluster, root, externalInputsInOrder);
        boolean lowCostHint = FusedCostModel.resolveLowCostHint(plan);
        FusedDispatchFamily dispatchFamily = FusedCostModel.resolveDispatchFamily(plan);
        int dispatchComplexity = FusedCostModel.estimateDispatchComplexity(plan);
        int dispatchScale = FusedCostModel.resolveDispatchScale(dispatchComplexity);

        return new Result(
                new FusedOperation(
                        "fused(" + cluster.size() + ")",
                        precisionMode,
                        lowCostHint,
                        dispatchFamily,
                        FusedSignatureBuilder.buildFromPlan(plan, precisionMode),
                        dispatchScale,
                        plan
                ),
                runtimeInputs
        );
    }

    private static List<Tensor> resolveRuntimeInputs(List<Tensor> externalInputsInOrder) {
        java.util.ArrayList<Tensor> resolved = new java.util.ArrayList<>(externalInputsInOrder.size());
        for (Tensor externalInput : externalInputsInOrder) {
            resolved.add(FusedAccessResolver.resolve(externalInput).backingTensor());
        }
        return java.util.List.copyOf(resolved);
    }

    /**
     * Factory result containing the descriptor and runtime input list.
     *
     * @param operation fused operation descriptor
     * @param runtimeInputs backing tensors needed by the executable
     */
    public record Result(
            FusedOperation operation,
            List<Tensor> runtimeInputs
    ) {}
}
