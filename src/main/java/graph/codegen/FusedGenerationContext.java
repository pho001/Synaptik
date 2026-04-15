package graph.codegen;

import java.util.Objects;

public record FusedGenerationContext(
        String internalClassName,
        int precisionMode,
        int vectorWidth,
        FusedExpressionPlan plan,
        FusedAsmSpecializationKind specializationKind
) {
    public static FusedGenerationContext create(
            String internalClassName,
            FusedExpressionPlan plan,
            int precisionMode,
            int vectorWidth,
            FusedAsmSpecializationKind specializationKind
    ) {
        Objects.requireNonNull(internalClassName, "internalClassName cannot be null");
        Objects.requireNonNull(plan, "plan cannot be null");
        Objects.requireNonNull(specializationKind, "specializationKind cannot be null");
        return new FusedGenerationContext(
                internalClassName,
                precisionMode,
                Math.max(1, vectorWidth),
                plan,
                specializationKind
        );
    }

    public int inputCount() {
        return plan.inputCount();
    }

    public int nodeCount() {
        return plan.nodeCount();
    }
}
