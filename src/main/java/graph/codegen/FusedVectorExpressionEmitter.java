package graph.codegen;

import org.objectweb.asm.MethodVisitor;
import utils.SlotManager;

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
            case ABS -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "abs", precisionMode);
            case RELU -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "relu", precisionMode);
            case CLAMP_MIN -> {
                double minValue = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) minValue);
                } else {
                    mv.visitLdcInsn(minValue);
                }
                FusedAsmSupport.emitVectorClampCall(mv, "clampMin", precisionMode);
            }
            case CLAMP_MAX -> {
                double maxValue = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) maxValue);
                } else {
                    mv.visitLdcInsn(maxValue);
                }
                FusedAsmSupport.emitVectorClampCall(mv, "clampMax", precisionMode);
            }
            case SIGMOID -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "sigmoid", precisionMode);
            case NOOP -> FusedAsmSupport.emitVectorUnaryOpCall(mv, "noop", precisionMode);
            case GT -> FusedAsmSupport.emitVectorCompareOpCall(mv, "gt", precisionMode);
            case GE -> FusedAsmSupport.emitVectorCompareOpCall(mv, "ge", precisionMode);
            case LT -> FusedAsmSupport.emitVectorCompareOpCall(mv, "lt", precisionMode);
            case LE -> FusedAsmSupport.emitVectorCompareOpCall(mv, "le", precisionMode);
            case EQ -> FusedAsmSupport.emitVectorCompareOpCall(mv, "eq", precisionMode);
            case NE -> FusedAsmSupport.emitVectorCompareOpCall(mv, "ne", precisionMode);
            case LOGICAL_AND -> FusedAsmSupport.emitVectorLogicalBinaryOpCall(mv, "logicalAnd", precisionMode);
            case LOGICAL_OR -> FusedAsmSupport.emitVectorLogicalBinaryOpCall(mv, "logicalOr", precisionMode);
            case LOGICAL_NOT -> FusedAsmSupport.emitVectorLogicalUnaryOpCall(mv, "logicalNot", precisionMode);
            case WHERE -> FusedAsmSupport.emitVectorWhereCall(mv, precisionMode);
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
}
