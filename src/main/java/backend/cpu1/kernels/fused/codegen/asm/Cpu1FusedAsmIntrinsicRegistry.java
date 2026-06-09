package backend.cpu1.kernels.fused.codegen.asm;

import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenRejectionReason;
import operations.Operation;

/**
 * Prepare-time support matrix for cpu1 fused ASM emission.
 */
public final class Cpu1FusedAsmIntrinsicRegistry {
    private Cpu1FusedAsmIntrinsicRegistry() {
    }

    public static Cpu1FusedCodegenRejectionReason rejectionReason(Cpu1FusedNodePlan node) {
        if (node == null) {
            throw new IllegalArgumentException("node cannot be null");
        }
        Operation.OpType opType = node.opType();
        if (requiresScalar(opType)) {
            return node.scalarParameter().present()
                    ? Cpu1FusedCodegenRejectionReason.NONE
                    : Cpu1FusedCodegenRejectionReason.UNSUPPORTED_SCALAR_BINDING;
        }
        if (node.scalarParameter().present() && opType != Operation.OpType.CONST_SCALAR) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_SCALAR_BINDING;
        }
        return switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX, NEG, INV, ABS, RELU, NOOP, WHERE, CONST_SCALAR ->
                    Cpu1FusedCodegenRejectionReason.NONE;
            case EXP, FAST_EXP, LOG, TANH, FAST_TANH, ERF, POW, POW_TENSOR, SQRT, SIGMOID ->
                    Cpu1FusedCodegenRejectionReason.UNSUPPORTED_INTRINSIC;
            default -> Cpu1FusedCodegenRejectionReason.UNSUPPORTED_OPERATION;
        };
    }

    public static boolean requiresScalar(Operation.OpType opType) {
        return opType == Operation.OpType.MUL_SCALAR
                || opType == Operation.OpType.CLAMP_MIN
                || opType == Operation.OpType.CLAMP_MAX;
    }
}
