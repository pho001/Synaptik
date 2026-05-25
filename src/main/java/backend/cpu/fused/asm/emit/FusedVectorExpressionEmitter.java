package backend.cpu.fused.asm.emit;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.ir.ScalarDoubleAttribute;
import backend.cpu.fused.numeric.FusedApproximationContract;
import backend.cpu.fused.numeric.FusedNumericContract;

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
            FusedNumericContract numericContract,
            FusedApproximationContract approximationContract,
            int vectorWidth
    ) {
        for (int ref : current.inputRefs()) {
            FusedVectorBytecode.loadVectorRef(mv, ref, plan, nodeVectorSlots, sm, numericContract);
        }
        operations.Operation.OpType opType = current.opType();
        switch (opType) {
            case ADD -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "add", numericContract);
            case SUB -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "sub", numericContract);
            case MUL -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "mul", numericContract);
            case DIV -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "div", numericContract);
            case MIN -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "min", numericContract);
            case MAX -> FusedVectorBytecode.emitVectorBinaryOpCall(mv, "max", numericContract);
            case NEG -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "neg", numericContract);
            case INV -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "inv", numericContract);
            case LOG -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "log", numericContract);
            case EXP -> FusedVectorBytecode.emitVectorUnaryOpCall(
                    mv,
                    approximationContract.useFastExp() ? "fastExp" : "exp",
                    numericContract
            );
            case FAST_EXP -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "fastExp", numericContract);
            case TANH -> FusedVectorBytecode.emitVectorUnaryOpCall(
                    mv,
                    approximationContract.useFastTanh() ? "fastTanh" : "tanh",
                    numericContract
            );
            case FAST_TANH -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "fastTanh", numericContract);
            case SQRT -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "sqrt", numericContract);
            case ABS -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "abs", numericContract);
            case RELU -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "relu", numericContract);
            case CLAMP_MIN -> {
                double minValue = ((ScalarDoubleAttribute) current.attributes()).value();
                if (numericContract.usesFloatCompute()) {
                    mv.visitLdcInsn((float) minValue);
                } else {
                    mv.visitLdcInsn(minValue);
                }
                FusedVectorBytecode.emitVectorClampCall(mv, "clampMin", numericContract);
            }
            case CLAMP_MAX -> {
                double maxValue = ((ScalarDoubleAttribute) current.attributes()).value();
                if (numericContract.usesFloatCompute()) {
                    mv.visitLdcInsn((float) maxValue);
                } else {
                    mv.visitLdcInsn(maxValue);
                }
                FusedVectorBytecode.emitVectorClampCall(mv, "clampMax", numericContract);
            }
            case SIGMOID -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "sigmoid", numericContract);
            case NOOP -> FusedVectorBytecode.emitVectorUnaryOpCall(mv, "noop", numericContract);
            case CONST_SCALAR -> {
                double constant = ((ScalarDoubleAttribute) current.attributes()).value();
                FusedVectorBytecode.emitVectorConstantCall(mv, constant, numericContract, vectorWidth);
            }
            case GT -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "gt", numericContract);
            case GE -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "ge", numericContract);
            case LT -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "lt", numericContract);
            case LE -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "le", numericContract);
            case EQ -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "eq", numericContract);
            case NE -> FusedVectorBytecode.emitVectorCompareOpCall(mv, "ne", numericContract);
            case LOGICAL_AND -> FusedVectorBytecode.emitVectorLogicalBinaryOpCall(mv, "logicalAnd", numericContract);
            case LOGICAL_OR -> FusedVectorBytecode.emitVectorLogicalBinaryOpCall(mv, "logicalOr", numericContract);
            case LOGICAL_NOT -> FusedVectorBytecode.emitVectorLogicalUnaryOpCall(mv, "logicalNot", numericContract);
            case WHERE -> FusedVectorBytecode.emitVectorWhereCall(mv, numericContract);
            case MUL_SCALAR -> {
                double scalar = ((ScalarDoubleAttribute) current.attributes()).value();
                if (numericContract.usesFloatCompute()) {
                    mv.visitLdcInsn((float) scalar);
                } else {
                    mv.visitLdcInsn(scalar);
                }
                FusedVectorBytecode.emitVectorMulScalarCall(mv, numericContract);
            }
            case POW -> {
                double exponent = ((ScalarDoubleAttribute) current.attributes()).value();
                FusedVectorBytecode.emitVectorPowSpecializedCall(mv, exponent, numericContract, vectorWidth);
            }
            default -> throw new UnsupportedOperationException("Operation " + opType + " is not supported for fused vector execution.");
        }
    }
}
