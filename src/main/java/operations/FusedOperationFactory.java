package operations;

import graph.codegen.FusedExpressionPlan;
import graph.codegen.FusedPlanBuilder;
import graph.optimizer.fusion.FusedCostModel;
import graph.optimizer.fusion.FusedPrecisionResolver;
import graph.optimizer.fusion.FusedSignatureBuilder;
import tensor.Tensor;

import java.util.List;

public final class FusedOperationFactory {
    private FusedOperationFactory() {}

    public static FusedOperation create(
            List<Tensor> cluster,
            Tensor root,
            List<Tensor> externalInputsInOrder
    ) {
        FusedExpressionPlan plan = FusedPlanBuilder.build(cluster, externalInputsInOrder, root);

        int precisionMode = FusedPrecisionResolver.resolve(cluster, root, externalInputsInOrder);
        boolean lowCostHint = FusedCostModel.resolveLowCostHint(cluster);
        int dispatchComplexity = FusedCostModel.estimateDispatchComplexity(cluster);
        int dispatchScale = FusedCostModel.resolveDispatchScale(dispatchComplexity);

        return new FusedOperation(
                "fused(" + cluster.size() + ")",
                precisionMode,
                lowCostHint,
                FusedSignatureBuilder.buildFromPlan(plan, precisionMode),
                dispatchScale,
                plan
        );
    }
}
