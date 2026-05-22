package backend.cpu.fused.asm;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.numeric.FusedComputeKind;
import backend.cpu.fused.numeric.FusedNumericContract;
import backend.cpu.fused.numeric.FusedValueLane;
import backend.cpu.fused.runtime.FusedDTypeOps;

import java.util.Objects;

/**
 * Internal code-generation context shared by fused ASM emitters.
 */
public record FusedGenerationContext(
        String internalClassName,
        FusedNumericContract numericContract,
        int vectorWidth,
        FusedExpressionPlan plan,
        FusedAsmSpecializationKind specializationKind
) {
    public static FusedGenerationContext create(
            String internalClassName,
            FusedExpressionPlan plan,
            FusedNumericContract numericContract,
            int vectorWidth,
            FusedAsmSpecializationKind specializationKind
    ) {
        Objects.requireNonNull(internalClassName, "internalClassName cannot be null");
        Objects.requireNonNull(plan, "plan cannot be null");
        Objects.requireNonNull(numericContract, "numericContract cannot be null");
        Objects.requireNonNull(specializationKind, "specializationKind cannot be null");
        return new FusedGenerationContext(
                internalClassName,
                numericContract,
                Math.max(1, vectorWidth),
                plan,
                specializationKind
        );
    }

    public int precisionMode() {
        if (numericContract.computeKind() == FusedComputeKind.F64) {
            return FusedDTypeOps.MODE_F64;
        }
        return numericContract.outputValueLane() == FusedValueLane.BF16
                ? FusedDTypeOps.MODE_BF16
                : FusedDTypeOps.MODE_F32;
    }

    public int inputCount() {
        return plan.inputCount();
    }

    public int nodeCount() {
        return plan.nodeCount();
    }
}
