package graph.codegen;

/**
 * Routes fused kernel codegen to dtype-specialized generators.
 */
public final class FusedKernelGeneratorRouter {
    private FusedKernelGeneratorRouter() {}

    public static byte[] generate(
            String internalClassName,
            FusedExpressionPlan plan,
            int precisionMode
    ) {
        return switch (precisionMode) {
            case FusedDTypeOps.MODE_F64, FusedDTypeOps.MODE_F32 ->
                    FusedOperationGenerator.generate(internalClassName, plan, precisionMode);
            case FusedDTypeOps.MODE_F16 ->
                    HFusedOperationGenerator.generate(internalClassName, plan);
            default -> throw new IllegalArgumentException("Unsupported precision mode: " + precisionMode);
        };
    }
}
