package backend.cpu.fused.asm.emit;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.numeric.FusedNumericContract;
import backend.cpu.fused.runtime.FusedVectorSpecies;
import org.objectweb.asm.MethodVisitor;
import tensor.DataType;
import utils.SlotKey;
import utils.SlotManager;

import static org.objectweb.asm.Opcodes.*;

final class FusedVectorBytecode {
    private FusedVectorBytecode() {}

    static void emitVectorWidthConstant(MethodVisitor mv, int vectorWidth) {
        mv.visitLdcInsn(Math.max(1, vectorWidth));
    }

    static void emitF32VectorSpeciesConstant(MethodVisitor mv, int vectorWidth) {
        mv.visitFieldInsn(
                GETSTATIC,
                "jdk/incubator/vector/FloatVector",
                FusedVectorSpecies.f32FieldName(vectorWidth),
                "Ljdk/incubator/vector/VectorSpecies;"
        );
    }

    static void emitVectorSpeciesConstant(MethodVisitor mv, FusedNumericContract numericContract, int vectorWidth) {
        String owner;
        String fieldName;
        if (numericContract.usesFloatCompute()) {
            owner = "jdk/incubator/vector/FloatVector";
            fieldName = FusedVectorSpecies.f32FieldName(vectorWidth);
        } else {
            owner = "jdk/incubator/vector/DoubleVector";
            fieldName = FusedVectorSpecies.f64FieldName(vectorWidth);
        }
        mv.visitFieldInsn(GETSTATIC, owner, fieldName, "Ljdk/incubator/vector/VectorSpecies;");
    }

    static void emitVectorBinaryOpCall(MethodVisitor mv, String op, FusedNumericContract numericContract) {
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + vd + vd + ")" + vd, false);
    }

    static void emitVectorCompareOpCall(MethodVisitor mv, String op, FusedNumericContract numericContract) {
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + vd + vd + ")" + FusedRuntimeCalls.maskTypeDesc(), false);
    }

    static void emitVectorLogicalBinaryOpCall(MethodVisitor mv, String op, FusedNumericContract numericContract) {
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String md = FusedRuntimeCalls.maskTypeDesc();
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + md + md + ")" + md, false);
    }

    static void emitVectorLogicalUnaryOpCall(MethodVisitor mv, String op, FusedNumericContract numericContract) {
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String md = FusedRuntimeCalls.maskTypeDesc();
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + md + ")" + md, false);
    }

    static void emitVectorUnaryOpCall(MethodVisitor mv, String op, FusedNumericContract numericContract) {
        if ("inv".equals(op)) {
            op = "reciprocal";
        }
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + vd + ")" + vd, false);
    }

    static void emitVectorConstantCall(MethodVisitor mv, double value, FusedNumericContract numericContract, int vectorWidth) {
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        if (numericContract.usesFloatCompute()) {
            emitVectorWidthConstant(mv, vectorWidth);
            mv.visitLdcInsn((float) value);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "constantF32", "(IF)" + vd, false);
        } else {
            emitVectorWidthConstant(mv, vectorWidth);
            mv.visitLdcInsn(value);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "constantF64", "(ID)" + vd, false);
        }
    }

    static void emitVectorMulScalarCall(MethodVisitor mv, FusedNumericContract numericContract) {
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "mulScalarF32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "mulScalarF64", "(" + vd + "D)" + vd, false);
        }
    }

    static void emitVectorPowSpecializedCall(
            MethodVisitor mv,
            double exponentValue,
            FusedNumericContract numericContract,
            int vectorWidth
    ) {
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        if (numericContract.usesFloatCompute()) {
            float exponent = (float) exponentValue;
            if (Float.compare(exponent, -2.0f) == 0) {
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "reciprocalSquare" + suffix, "(" + vd + ")" + vd, false);
                return;
            }
            if (Float.compare(exponent, -1.0f) == 0) {
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "reciprocal" + suffix, "(" + vd + ")" + vd, false);
                return;
            }
            if (Float.compare(exponent, 0.0f) == 0) {
                mv.visitInsn(POP);
                emitVectorConstantCall(mv, 1.0d, numericContract, vectorWidth);
                return;
            }
            if (Float.compare(exponent, 1.0f) == 0) {
                return;
            }
            if (Float.compare(exponent, 2.0f) == 0) {
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "square" + suffix, "(" + vd + ")" + vd, false);
                return;
            }
            if (Float.compare(exponent, 0.5f) == 0) {
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "sqrt" + suffix, "(" + vd + ")" + vd, false);
                return;
            }
            mv.visitLdcInsn(exponent);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "powF32", "(" + vd + "F)" + vd, false);
            return;
        }
        double exponent = exponentValue;
        if (Double.compare(exponent, -2.0d) == 0) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "reciprocalSquare" + suffix, "(" + vd + ")" + vd, false);
            return;
        }
        if (Double.compare(exponent, -1.0d) == 0) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "reciprocal" + suffix, "(" + vd + ")" + vd, false);
            return;
        }
        if (Double.compare(exponent, 0.0d) == 0) {
            mv.visitInsn(POP);
            emitVectorConstantCall(mv, 1.0d, numericContract, vectorWidth);
            return;
        }
        if (Double.compare(exponent, 1.0d) == 0) {
            return;
        }
        if (Double.compare(exponent, 2.0d) == 0) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "square" + suffix, "(" + vd + ")" + vd, false);
            return;
        }
        if (Double.compare(exponent, 0.5d) == 0) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "sqrt" + suffix, "(" + vd + ")" + vd, false);
            return;
        }
        mv.visitLdcInsn(exponent);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "powF64", "(" + vd + "D)" + vd, false);
    }

    static void emitVectorPowTensorCall(MethodVisitor mv, FusedNumericContract numericContract) {
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "pow" + suffix, "(" + vd + vd + ")" + vd, false);
    }

    static void emitVectorSigmoidComposite(
            MethodVisitor mv,
            FusedNumericContract numericContract,
            int vectorWidth,
            int tmpVectorSlot
    ) {
        emitVectorUnaryOpCall(mv, "neg", numericContract);
        emitVectorUnaryOpCall(mv, "exp", numericContract);
        emitVectorConstantCall(mv, 1.0d, numericContract, vectorWidth);
        emitVectorBinaryOpCall(mv, "add", numericContract);
        mv.visitVarInsn(ASTORE, tmpVectorSlot);

        emitVectorConstantCall(mv, 1.0d, numericContract, vectorWidth);
        mv.visitVarInsn(ALOAD, tmpVectorSlot);
        emitVectorBinaryOpCall(mv, "div", numericContract);
    }

    static void emitVectorClampCall(MethodVisitor mv, String op, FusedNumericContract numericContract) {
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "F32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "F64", "(" + vd + "D)" + vd, false);
        }
    }

    static void emitVectorWhereCall(MethodVisitor mv, FusedNumericContract numericContract) {
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String md = FusedRuntimeCalls.maskTypeDesc();
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "where" + suffix, "(" + md + vd + vd + ")" + vd, false);
    }

    static void loadVectorRef(
            MethodVisitor mv,
            int ref,
            FusedExpressionPlan plan,
            int[] nodeVectorSlots,
            SlotManager sm,
            FusedNumericContract numericContract
    ) {
        if (ref < sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS).size()) {
            int cachedSlot = sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS).get(ref);
            mv.visitVarInsn(ALOAD, cachedSlot);
            emitVectorRefCast(mv, plan.inputs().get(ref).dataType(), numericContract);
            return;
        }

        int nodeIndex = ref - sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS).size();
        if (nodeIndex < 0 || nodeIndex >= nodeVectorSlots.length) {
            throw new IllegalArgumentException("Invalid fused vector ref " + ref);
        }
        mv.visitVarInsn(ALOAD, nodeVectorSlots[nodeIndex]);
        emitVectorRefCast(mv, plan.nodes().get(nodeIndex).outputType(), numericContract);
    }

    static void emitVectorRefCast(MethodVisitor mv, DataType dataType, FusedNumericContract numericContract) {
        if (dataType == DataType.BOOL) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/VectorMask");
        } else if (numericContract.usesFloatCompute()) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/FloatVector");
        } else {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/DoubleVector");
        }
    }
}
