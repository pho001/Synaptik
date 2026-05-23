package backend.cpu.fused.asm;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.numeric.FusedApproximationContract;
import backend.cpu.fused.numeric.FusedNumericContract;

import java.util.Objects;

/**
 * Internal code-generation context shared by fused ASM emitters.
 */
public record FusedGenerationContext(
        String internalClassName,
        FusedNumericContract numericContract,
        FusedApproximationContract approximationContract,
        int vectorWidth,
        FusedExpressionPlan plan,
        FusedAsmSpecializationKind specializationKind
) {
    public static FusedGenerationContext create(
            String internalClassName,
            FusedExpressionPlan plan,
            FusedNumericContract numericContract,
            FusedApproximationContract approximationContract,
            int vectorWidth,
            FusedAsmSpecializationKind specializationKind
    ) {
        Objects.requireNonNull(internalClassName, "internalClassName cannot be null");
        Objects.requireNonNull(plan, "plan cannot be null");
        Objects.requireNonNull(numericContract, "numericContract cannot be null");
        Objects.requireNonNull(approximationContract, "approximationContract cannot be null");
        Objects.requireNonNull(specializationKind, "specializationKind cannot be null");
        return new FusedGenerationContext(
                internalClassName,
                numericContract,
                approximationContract,
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
