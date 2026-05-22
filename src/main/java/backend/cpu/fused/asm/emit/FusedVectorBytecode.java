package backend.cpu.fused.asm.emit;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.runtime.FusedDTypeOps;
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

    static void emitVectorSpeciesConstant(MethodVisitor mv, int precisionMode, int vectorWidth) {
        String owner;
        String fieldName;
        if (precisionMode == FusedDTypeOps.MODE_F32 || precisionMode == FusedDTypeOps.MODE_BF16) {
            owner = "jdk/incubator/vector/FloatVector";
            fieldName = switch (normalizedVectorWidth(vectorWidth)) {
                case 2 -> "SPECIES_64";
                case 4 -> "SPECIES_128";
                case 8 -> "SPECIES_256";
                default -> "SPECIES_PREFERRED";
            };
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            owner = "jdk/incubator/vector/DoubleVector";
            fieldName = switch (normalizedVectorWidth(vectorWidth)) {
                case 1 -> "SPECIES_64";
                case 2 -> "SPECIES_128";
                case 4 -> "SPECIES_256";
                case 8 -> "SPECIES_512";
                default -> "SPECIES_PREFERRED";
            };
        } else {
            throw new UnsupportedOperationException("Vector species constants are supported only for F32/F64/BF16 fused modes.");
        }
        mv.visitFieldInsn(GETSTATIC, owner, fieldName, "Ljdk/incubator/vector/VectorSpecies;");
    }

    private static int normalizedVectorWidth(int vectorWidth) {
        if (vectorWidth <= 1) {
            return 1;
        }
        if (vectorWidth <= 2) {
            return 2;
        }
        if (vectorWidth <= 4) {
            return 4;
        }
        return 8;
    }

    static void emitVectorBinaryOpCall(MethodVisitor mv, String op, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op, "(Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
            return;
        }
        String suffix = precisionMode == FusedDTypeOps.MODE_F32 ? "F32" : "F64";
        String vd = FusedRuntimeCalls.vectorTypeDesc(precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + vd + vd + ")" + vd, false);
    }

    static void emitVectorCompareOpCall(MethodVisitor mv, String op, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op, "(Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
            return;
        }
        String suffix = precisionMode == FusedDTypeOps.MODE_F32 ? "F32" : "F64";
        String vd = FusedRuntimeCalls.vectorTypeDesc(precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + vd + vd + ")" + FusedRuntimeCalls.maskTypeDesc(), false);
    }

    static void emitVectorLogicalBinaryOpCall(MethodVisitor mv, String op, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op, "(Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
            return;
        }
        String suffix = precisionMode == FusedDTypeOps.MODE_F32 ? "F32" : "F64";
        String md = FusedRuntimeCalls.maskTypeDesc();
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + md + md + ")" + md, false);
    }

    static void emitVectorLogicalUnaryOpCall(MethodVisitor mv, String op, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op, "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            return;
        }
        String suffix = precisionMode == FusedDTypeOps.MODE_F32 ? "F32" : "F64";
        String md = FusedRuntimeCalls.maskTypeDesc();
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + md + ")" + md, false);
    }

    static void emitVectorUnaryOpCall(MethodVisitor mv, String op, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op, "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            return;
        }
        String suffix = precisionMode == FusedDTypeOps.MODE_F32 ? "F32" : "F64";
        String vd = FusedRuntimeCalls.vectorTypeDesc(precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + suffix, "(" + vd + ")" + vd, false);
    }

    static void emitVectorConstantCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "constant", "(DI)Ljava/lang/Object;", false);
            return;
        }
        String vd = FusedRuntimeCalls.vectorTypeDesc(precisionMode);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "constantF32", "(F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "constantF64", "(D)" + vd, false);
        }
    }

    static void emitVectorMulScalarCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "mulScalar", "(Ljava/lang/Object;DI)Ljava/lang/Object;", false);
            return;
        }
        String vd = FusedRuntimeCalls.vectorTypeDesc(precisionMode);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "mulScalarF32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "mulScalarF64", "(" + vd + "D)" + vd, false);
        }
    }

    static void emitVectorPowCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "pow", "(Ljava/lang/Object;DI)Ljava/lang/Object;", false);
            return;
        }
        String vd = FusedRuntimeCalls.vectorTypeDesc(precisionMode);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "powF32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "powF64", "(" + vd + "D)" + vd, false);
        }
    }

    static void emitVectorClampCall(MethodVisitor mv, String op, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op, "(Ljava/lang/Object;DI)Ljava/lang/Object;", false);
            return;
        }
        String vd = FusedRuntimeCalls.vectorTypeDesc(precisionMode);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "F32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", op + "F64", "(" + vd + "D)" + vd, false);
        }
    }

    static void emitVectorWhereCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "where", "(Ljava/lang/Object;Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
            return;
        }
        String suffix = precisionMode == FusedDTypeOps.MODE_F32 ? "F32" : "F64";
        String md = FusedRuntimeCalls.maskTypeDesc();
        String vd = FusedRuntimeCalls.vectorTypeDesc(precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedVectorOps", "where" + suffix, "(" + md + vd + vd + ")" + vd, false);
    }

    static void loadVectorRef(
            MethodVisitor mv,
            int ref,
            FusedExpressionPlan plan,
            int[] nodeVectorSlots,
            SlotManager sm,
            int precisionMode
    ) {
        if (ref < sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS).size()) {
            int cachedSlot = sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS).get(ref);
            mv.visitVarInsn(ALOAD, cachedSlot);
            emitVectorRefCast(mv, plan.inputs().get(ref).dataType(), precisionMode);
            return;
        }

        int nodeIndex = ref - sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS).size();
        if (nodeIndex < 0 || nodeIndex >= nodeVectorSlots.length) {
            throw new IllegalArgumentException("Invalid fused vector ref " + ref);
        }
        mv.visitVarInsn(ALOAD, nodeVectorSlots[nodeIndex]);
        emitVectorRefCast(mv, plan.nodes().get(nodeIndex).outputType(), precisionMode);
    }

    private static void emitVectorRefCast(MethodVisitor mv, DataType dataType, int precisionMode) {
        if (precisionMode != FusedDTypeOps.MODE_F32 && precisionMode != FusedDTypeOps.MODE_F64) {
            return;
        }
        if (dataType == DataType.BOOL) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/VectorMask");
        } else if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/FloatVector");
        } else {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/DoubleVector");
        }
    }
}
