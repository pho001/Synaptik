package backend.cpu.fused.codegen;

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
        FusedAsmSpecializationKind specializationKind = FusedAsmSpecializationMatcher.match(plan, precisionMode);
        return FusedOperationGenerator.generate(internalClassName, plan, precisionMode, vectorWidth, specializationKind);
    }

    public static byte[] generate(
            String internalClassName,
            FusedExpressionPlan plan,
            int precisionMode,
            int vectorWidth,
            FusedAsmSpecializationKind specializationKind
    ) {
        return FusedOperationGenerator.generate(internalClassName, plan, precisionMode, vectorWidth, specializationKind);
    }
}
