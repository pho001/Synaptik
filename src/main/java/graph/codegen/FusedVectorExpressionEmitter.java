package graph.codegen;

import org.objectweb.asm.MethodVisitor;
import utils.SlotManager;

import static org.objectweb.asm.Opcodes.INVOKESTATIC;

public final class FusedVectorExpressionEmitter {
    private FusedVectorExpressionEmitter() {}

    public static void emitNodeEvaluationBytecode(
            MethodVisitor mv,
            FusedExpressionPlan plan,
            FusedNodePlan current,
            int[] nodeVectorSlots,
            SlotManager sm,
            int precisionMode
    ) {
        for (int ref : current.inputRefs()) {
            FusedAsmSupport.loadVectorRef(mv, ref, plan, nodeVectorSlots, sm, precisionMode);
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
            case GT -> emitVectorBoolOpCall(mv, "gt", precisionMode);
            case GE -> emitVectorBoolOpCall(mv, "ge", precisionMode);
            case LT -> emitVectorBoolOpCall(mv, "lt", precisionMode);
            case LE -> emitVectorBoolOpCall(mv, "le", precisionMode);
            case EQ -> emitVectorBoolOpCall(mv, "eq", precisionMode);
            case NE -> emitVectorBoolOpCall(mv, "ne", precisionMode);
            case LOGICAL_AND -> emitVectorBoolBinaryCall(mv, "logicalAnd", precisionMode);
            case LOGICAL_OR -> emitVectorBoolBinaryCall(mv, "logicalOr", precisionMode);
            case LOGICAL_NOT -> emitVectorBoolUnaryCall(mv, "logicalNot", precisionMode);
            case WHERE -> emitVectorWhereCall(mv, precisionMode);
            case MUL_SCALAR -> {
                double scalar = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) scalar);
                } else {
                    mv.visitLdcInsn(scalar);
                }
                FusedAsmSupport.emitVectorMulScalarCall(mv, precisionMode);
            }
            case POW -> {
                double exponent = ((ScalarDoubleAttribute) current.attributes()).value();
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

    private static void emitVectorBoolOpCall(MethodVisitor mv, String op, int precisionMode) {
        mv.visitLdcInsn(precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", op, "(Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
    }

    private static void emitVectorBoolBinaryCall(MethodVisitor mv, String op, int precisionMode) {
        mv.visitLdcInsn(precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", op, "(Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
    }

    private static void emitVectorBoolUnaryCall(MethodVisitor mv, String op, int precisionMode) {
        mv.visitLdcInsn(precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", op, "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
    }

    private static void emitVectorWhereCall(MethodVisitor mv, int precisionMode) {
        mv.visitLdcInsn(precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", "where", "(Ljava/lang/Object;Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
    }
}
