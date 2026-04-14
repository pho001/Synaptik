package graph.codegen;

/**
 * Routes fused kernel codegen to dtype-specialized generators.
 */
public final class FusedKernelGeneratorRouter {
    private FusedKernelGeneratorRouter() {}

    public static byte[] generate(
            String internalClassName,
            FusedExpressionPlan plan,
            int precisionMode,
            int vectorWidth
    ) {
        return FusedOperationGenerator.generate(internalClassName, plan, precisionMode, vectorWidth);
    }
}
