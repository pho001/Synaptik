package backend.cpu.fused.asm.emit;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.ir.ScalarDoubleAttribute;

import org.objectweb.asm.MethodVisitor;
import utils.SlotManager;

/**
 * Internal ASM emitter for vector expression evaluation in generated fused kernels.
 */
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
            FusedVectorBytecode.loadVectorRef(mv, ref, plan, nodeVectorSlots, sm, precisionMode);
        }
        operations.Operation.OpType opType = current.opType();
        switch (opType) {
            case ADD -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "add", precisionMode);
            case SUB -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "sub", precisionMode);
            case MUL -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "mul", precisionMode);
            case DIV -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "div", precisionMode);
            case MIN -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "min", precisionMode);
            case MAX -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "max", precisionMode);
            case NEG -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "neg", precisionMode);
            case INV -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "inv", precisionMode);
            case LOG -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "log", precisionMode);
            case EXP -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "exp", precisionMode, sm);
            case FAST_EXP -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "fastExp", precisionMode);
            case TANH -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "tanh", precisionMode, sm);
            case FAST_TANH -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "fastTanh", precisionMode);
            case SQRT -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "sqrt", precisionMode);
            case ABS -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "abs", precisionMode);
            case RELU -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "relu", precisionMode);
            case CLAMP_MIN -> {
                double minValue = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == backend.cpu.fused.runtime.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) minValue);
                } else {
                    mv.visitLdcInsn(minValue);
                }
                FusedVectorBytecode.emitVectorClampCall(mv, "clampMin", precisionMode);
            }
            case CLAMP_MAX -> {
                double maxValue = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == backend.cpu.fused.runtime.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) maxValue);
                } else {
                    mv.visitLdcInsn(maxValue);
                }
                FusedVectorBytecode.emitVectorClampCall(mv, "clampMax", precisionMode);
            }
            case SIGMOID -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "sigmoid", precisionMode);
            case NOOP -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "noop", precisionMode);
            case CONST_SCALAR -> {
                double constant = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == backend.cpu.fused.runtime.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) constant);
                } else {
                    mv.visitLdcInsn(constant);
                }
                FusedVectorBytecode.emitVectorConstantCall(mv, precisionMode);
            }
            case GT -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "gt", precisionMode);
            case GE -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "ge", precisionMode);
            case LT -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "lt", precisionMode);
            case LE -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "le", precisionMode);
            case EQ -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "eq", precisionMode);
            case NE -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "ne", precisionMode);
            case LOGICAL_AND -> FusedVectorBytecode.emitVectorLogicalBinaryOpCall(mv, "logicalAnd", precisionMode);
            case LOGICAL_OR -> FusedVectorBytecode.emitVectorLogicalBinaryOpCall(mv, "logicalOr", precisionMode);
            case LOGICAL_NOT -> FusedVectorBytecode.emitVectorLogicalUnaryOpCall(mv, "logicalNot", precisionMode);
            case WHERE -> FusedVectorBytecode.emitVectorWhereCall(mv, precisionMode);
            case MUL_SCALAR -> {
                double scalar = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == backend.cpu.fused.runtime.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) scalar);
                } else {
                    mv.visitLdcInsn(scalar);
                }
                FusedVectorBytecode.emitVectorMulScalarCall(mv, precisionMode);
            }
            case POW -> {
                double exponent = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == backend.cpu.fused.runtime.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) exponent);
                } else {
                    mv.visitLdcInsn(exponent);
                }
                FusedVectorBytecode.emitVectorPowCall(mv, precisionMode);
            }
            default -> throw new UnsupportedOperationException("Operation " + opType + " is not supported for fused vector execution.");
        }
    }
}
