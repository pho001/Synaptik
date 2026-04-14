package graph.codegen;

import java.util.Objects;

public record FusedGenerationContext(
        String internalClassName,
        int precisionMode,
        int vectorWidth,
        FusedExpressionPlan plan
) {
    public static FusedGenerationContext create(
            String internalClassName,
            FusedExpressionPlan plan,
            int precisionMode,
            int vectorWidth
    ) {
        Objects.requireNonNull(internalClassName, "internalClassName cannot be null");
        Objects.requireNonNull(plan, "plan cannot be null");
        return new FusedGenerationContext(internalClassName, precisionMode, Math.max(1, vectorWidth), plan);
    }

    public int inputCount() {
        return plan.inputCount();
    }

    public int nodeCount() {
        return plan.nodeCount();
    }
}
