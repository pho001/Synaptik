package graph.codegen;

import org.objectweb.asm.MethodVisitor;
import utils.SlotManager;

public final class FusedVectorExpressionEmitter {
    private FusedVectorExpressionEmitter() {}

    public static void emitNodeEvaluationBytecode(
            MethodVisitor mv,
            FusedNodePlan current,
            int[] nodeVectorSlots,
            SlotManager sm,
            int precisionMode
    ) {
        for (int ref : current.inputRefs()) {
            FusedAsmSupport.loadVectorRef(mv, ref, nodeVectorSlots, sm);
        }
        operations.Operation.OpType opType = current.opType();
        switch (opType) {
            case ADD -> FusedAsmSupport.emitVectorBinaryOpCall(mv, "add", precisionMode);
            case SUB -> FusedAsmSupport.emitVectorBinaryOpCall(mv, "sub", precisionMode);
            case MUL -> FusedAsmSupport.emitVectorBinaryOpCall(mv, "mul", precisionMode);
            case DIV -> FusedAsmSupport.emitVectorBinaryOpCall(mv, "div", precisionMode);
            case MIN -> FusedAsmSupport.emitVectorBinaryOpCall(mv, "min", precisionMode);
            case MAX -> FusedAsmSupport.emitVectorBinaryOpCall(mv, "max", precisionMode);
            case NEG -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "neg", precisionMode);
            case INV -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "inv", precisionMode);
            case LOG -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "log", precisionMode);
            case EXP -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "exp", precisionMode, sm);
            case FAST_EXP -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "fastExp", precisionMode);
            case TANH -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "tanh", precisionMode, sm);
            case FAST_TANH -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "fastTanh", precisionMode);
            case SQRT -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "sqrt", precisionMode);
            case RELU -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "relu", precisionMode);
            case SIGMOID -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "sigmoid", precisionMode);
            case NOOP -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "noop", precisionMode);
            case MUL_SCALAR -> {
                double scalar = ((Number) current.parameter()).doubleValue();
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) scalar);
                } else {
                    mv.visitLdcInsn(scalar);
                }
                FusedAsmSupport.emitVectorMulScalarCall(mv, precisionMode);
            }
            case POW -> {
                double exponent = ((Number) current.parameter()).doubleValue();
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) exponent);
                } else {
                    mv.visitLdcInsn(exponent);
                }
                FusedAsmSupport.emitVectorPowCall(mv, precisionMode);
            }
            default -> throw new UnsupportedOperationException("Operation " + opType + " is not supported for fused vector execution.");
        }
    }
}
