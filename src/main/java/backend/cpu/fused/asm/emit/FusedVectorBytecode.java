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
        if (numericContract.writesBf16()) {
            String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "BF16", "(" + vd + vd + ")" + vd, false);
            return;
        }
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + vd + vd + ")" + vd, false);
    }

    static void emitVectorCompareOpCall(MethodVisitor mv, String op, FusedNumericContract numericContract) {
        if (numericContract.writesBf16()) {
            String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "BF16", "(" + vd + vd + ")" + FusedRuntimeCalls.maskTypeDesc(), false);
            return;
        }
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + vd + vd + ")" + FusedRuntimeCalls.maskTypeDesc(), false);
    }

    static void emitVectorLogicalBinaryOpCall(MethodVisitor mv, String op, FusedNumericContract numericContract) {
        if (numericContract.writesBf16()) {
            String md = FusedRuntimeCalls.maskTypeDesc();
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "BF16", "(" + md + md + ")" + md, false);
            return;
        }
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String md = FusedRuntimeCalls.maskTypeDesc();
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + md + md + ")" + md, false);
    }

    static void emitVectorLogicalUnaryOpCall(MethodVisitor mv, String op, FusedNumericContract numericContract) {
        if (numericContract.writesBf16()) {
            String md = FusedRuntimeCalls.maskTypeDesc();
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "BF16", "(" + md + ")" + md, false);
            return;
        }
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String md = FusedRuntimeCalls.maskTypeDesc();
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + md + ")" + md, false);
    }

    static void emitVectorUnaryOpCall(MethodVisitor mv, String op, FusedNumericContract numericContract) {
        if (numericContract.writesBf16()) {
            String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "BF16", "(" + vd + ")" + vd, false);
            return;
        }
        String suffix = numericContract.usesFloatCompute() ? "F32" : "F64";
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + vd + ")" + vd, false);
    }

    static void emitVectorConstantCall(MethodVisitor mv, FusedNumericContract numericContract) {
        if (numericContract.writesBf16()) {
            String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "constantBF16", "(D)" + vd, false);
            return;
        }
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "constantF32", "(F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "constantF64", "(D)" + vd, false);
        }
    }

    static void emitVectorMulScalarCall(MethodVisitor mv, FusedNumericContract numericContract) {
        if (numericContract.writesBf16()) {
            String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "mulScalarBF16", "(" + vd + "D)" + vd, false);
            return;
        }
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "mulScalarF32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "mulScalarF64", "(" + vd + "D)" + vd, false);
        }
    }

    static void emitVectorPowCall(MethodVisitor mv, FusedNumericContract numericContract) {
        if (numericContract.writesBf16()) {
            String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "powBF16", "(" + vd + "D)" + vd, false);
            return;
        }
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "powF32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "powF64", "(" + vd + "D)" + vd, false);
        }
    }

    static void emitVectorClampCall(MethodVisitor mv, String op, FusedNumericContract numericContract) {
        if (numericContract.writesBf16()) {
            String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "BF16", "(" + vd + "D)" + vd, false);
            return;
        }
        String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "F32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "F64", "(" + vd + "D)" + vd, false);
        }
    }

    static void emitVectorWhereCall(MethodVisitor mv, FusedNumericContract numericContract) {
        if (numericContract.writesBf16()) {
            String md = FusedRuntimeCalls.maskTypeDesc();
            String vd = FusedRuntimeCalls.vectorTypeDesc(numericContract);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "whereBF16", "(" + md + vd + vd + ")" + vd, false);
            return;
        }
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
