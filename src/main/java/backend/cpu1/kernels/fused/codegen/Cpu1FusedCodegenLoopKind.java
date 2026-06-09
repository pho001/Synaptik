package backend.cpu1.kernels.fused.codegen;

import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.prepare.dispatch.Cpu1FusedDispatchDecision;

/**
 * Loop shape expected from the generated ASM kernel.
 */
public enum Cpu1FusedCodegenLoopKind {
    CONTIGUOUS_VECTOR,
    CONTIGUOUS_SCALAR,
    STRIDED_SCALAR;

    public static Cpu1FusedCodegenLoopKind select(
            Cpu1FusedExpressionPlan plan,
            Cpu1LayoutKind layoutKind,
            Cpu1FusedDispatchDecision dispatchDecision
    ) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        if (layoutKind == null) {
            throw new IllegalArgumentException("layoutKind cannot be null");
        }
        if (dispatchDecision == null) {
            throw new IllegalArgumentException("dispatchDecision cannot be null");
        }
        if (layoutKind == Cpu1LayoutKind.CONTIGUOUS && plan.usesOnlyLinearInputs()) {
            return dispatchDecision.requestedVectorizationKind() == Cpu1VectorizationKind.VECTOR
                    ? CONTIGUOUS_VECTOR
                    : CONTIGUOUS_SCALAR;
        }
        return STRIDED_SCALAR;
    }
}
