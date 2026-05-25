package backend.cpu.fused.asm.emit;

import backend.cpu.fused.numeric.FusedNumericContract;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import tensor.DataType;
import utils.SlotKey;
import utils.SlotManager;

import static org.objectweb.asm.Opcodes.*;

final class FusedRuntimeCalls {
    private FusedRuntimeCalls() {}

    static void emitGetRawArrayFromTensorCall(MethodVisitor mv, DataType dataType) {
        switch (dataType) {
            case FLOAT32 -> mv.visitMethodInsn(INVOKESTATIC, "tensor/TensorInternalAccess", "float32Data", "(Ltensor/Tensor;)[F", false);
            case FLOAT64 -> mv.visitMethodInsn(INVOKESTATIC, "tensor/TensorInternalAccess", "float64Data", "(Ltensor/Tensor;)[D", false);
            case BFLOAT16 -> mv.visitMethodInsn(INVOKESTATIC, "tensor/TensorInternalAccess", "bfloat16Data", "(Ltensor/Tensor;)[S", false);
            case BOOL -> mv.visitMethodInsn(INVOKESTATIC, "tensor/TensorInternalAccess", "boolData", "(Ltensor/Tensor;)[B", false);
            case INT32 -> mv.visitMethodInsn(INVOKESTATIC, "tensor/TensorInternalAccess", "int32Data", "(Ltensor/Tensor;)[I", false);
            case INT64 -> mv.visitMethodInsn(INVOKESTATIC, "tensor/TensorInternalAccess", "int64Data", "(Ltensor/Tensor;)[J", false);
        }
    }

    static void emitGetNativeInputSegmentCall(MethodVisitor mv) {
        mv.visitMethodInsn(
                INVOKEVIRTUAL,
                "backend/cpu/kernels/CpuKernelContext",
                "fusedNativeInputSegment",
                "(I)Ljava/lang/foreign/MemorySegment;",
                false
        );
    }

    static void emitGetNativeOutputSegmentCall(MethodVisitor mv) {
        mv.visitMethodInsn(
                INVOKEVIRTUAL,
                "backend/cpu/kernels/CpuKernelContext",
                "fusedNativeOutputSegment",
                "()Ljava/lang/foreign/MemorySegment;",
                false
        );
    }

    static void emitLoadScalarFromSegmentCall(MethodVisitor mv, DataType dataType, FusedNumericContract numericContract) {
        switch (dataType) {
            case FLOAT32 -> {
                emitElementIndexToByteOffset(mv, Float.BYTES);
                emitSegmentGet(mv, "JAVA_FLOAT", "Ljava/lang/foreign/ValueLayout$OfFloat;", "F");
                if (numericContract.usesDoubleCompute()) {
                    mv.visitInsn(F2D);
                }
            }
            case FLOAT64 -> {
                emitElementIndexToByteOffset(mv, Double.BYTES);
                emitSegmentGet(mv, "JAVA_DOUBLE", "Ljava/lang/foreign/ValueLayout$OfDouble;", "D");
                if (numericContract.usesFloatCompute()) {
                    mv.visitInsn(D2F);
                }
            }
            case BFLOAT16 -> {
                emitElementIndexToByteOffset(mv, Short.BYTES);
                emitSegmentGet(mv, "JAVA_SHORT", "Ljava/lang/foreign/ValueLayout$OfShort;", "S");
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/kernels/CpuDTypeOps", "fromBFloat16Bits", "(S)F", false);
                if (numericContract.usesDoubleCompute()) {
                    mv.visitInsn(F2D);
                }
            }
            case BOOL -> emitLoadBoolFromSegmentCall(mv);
            case INT32, INT64 -> throw new UnsupportedOperationException("INT32/INT64 segment scalar loads are not supported for fused execution.");
        }
    }

    static void emitLoadBoolFromSegmentCall(MethodVisitor mv) {
        emitElementIndexToByteOffset(mv, Byte.BYTES);
        emitSegmentGet(mv, "JAVA_BYTE", "Ljava/lang/foreign/ValueLayout$OfByte;", "B");
        Label zero = new Label();
        Label done = new Label();
        mv.visitJumpInsn(IFEQ, zero);
        mv.visitInsn(ICONST_1);
        mv.visitJumpInsn(GOTO, done);
        mv.visitLabel(zero);
        mv.visitInsn(ICONST_0);
        mv.visitLabel(done);
    }

    static void emitStoreScalarToSegmentCall(
            MethodVisitor mv,
            DataType dataType,
            FusedNumericContract numericContract,
            SlotManager sm
    ) {
        int tmp = sm.get(SlotKey.TMP_REGISTER);
        switch (dataType) {
            case FLOAT32 -> {
                if (numericContract.usesDoubleCompute()) {
                    mv.visitInsn(D2F);
                }
                mv.visitVarInsn(FSTORE, tmp);
                emitElementIndexToByteOffset(mv, Float.BYTES);
                emitSegmentSetPrefix(mv, "JAVA_FLOAT", "Ljava/lang/foreign/ValueLayout$OfFloat;");
                mv.visitVarInsn(FLOAD, tmp);
                emitSegmentSet(mv, "Ljava/lang/foreign/ValueLayout$OfFloat;", "F");
            }
            case FLOAT64 -> {
                if (numericContract.usesFloatCompute()) {
                    mv.visitInsn(F2D);
                }
                mv.visitVarInsn(DSTORE, tmp);
                emitElementIndexToByteOffset(mv, Double.BYTES);
                emitSegmentSetPrefix(mv, "JAVA_DOUBLE", "Ljava/lang/foreign/ValueLayout$OfDouble;");
                mv.visitVarInsn(DLOAD, tmp);
                emitSegmentSet(mv, "Ljava/lang/foreign/ValueLayout$OfDouble;", "D");
            }
            case BFLOAT16 -> {
                if (numericContract.usesDoubleCompute()) {
                    mv.visitInsn(D2F);
                }
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/kernels/CpuDTypeOps", "toBFloat16Bits", "(F)S", false);
                mv.visitVarInsn(ISTORE, tmp);
                emitElementIndexToByteOffset(mv, Short.BYTES);
                emitSegmentSetPrefix(mv, "JAVA_SHORT", "Ljava/lang/foreign/ValueLayout$OfShort;");
                mv.visitVarInsn(ILOAD, tmp);
                emitSegmentSet(mv, "Ljava/lang/foreign/ValueLayout$OfShort;", "S");
            }
            case BOOL -> {
                mv.visitVarInsn(ISTORE, tmp);
                emitElementIndexToByteOffset(mv, Byte.BYTES);
                emitSegmentSetPrefix(mv, "JAVA_BYTE", "Ljava/lang/foreign/ValueLayout$OfByte;");
                mv.visitVarInsn(ILOAD, tmp);
                emitSegmentSet(mv, "Ljava/lang/foreign/ValueLayout$OfByte;", "B");
            }
            case INT32, INT64 -> throw new UnsupportedOperationException("INT32/INT64 segment scalar stores are not supported for fused execution.");
        }
    }

    static void emitDirectLinearVectorLoad(MethodVisitor mv, DataType dataType, FusedNumericContract numericContract, int vectorWidth) {
        FusedVectorBytecode.emitVectorSpeciesConstant(mv, numericContract, vectorWidth);
        mv.visitInsn(DUP_X2);
        mv.visitInsn(POP);
        if (dataType == DataType.FLOAT32 && numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(INVOKESTATIC, "jdk/incubator/vector/FloatVector", "fromArray", "(Ljdk/incubator/vector/VectorSpecies;[FI)Ljdk/incubator/vector/FloatVector;", false);
        } else if (dataType == DataType.FLOAT64 && numericContract.usesDoubleCompute()) {
            mv.visitMethodInsn(INVOKESTATIC, "jdk/incubator/vector/DoubleVector", "fromArray", "(Ljdk/incubator/vector/VectorSpecies;[DI)Ljdk/incubator/vector/DoubleVector;", false);
        } else {
            throw new UnsupportedOperationException("Direct vector loads require storage dtype to match fused compute kind.");
        }
    }

    static void emitBroadcastArrayVectorLoad(
            MethodVisitor mv,
            DataType dataType,
            FusedNumericContract numericContract,
            int vectorWidth
    ) {
        FusedVectorBytecode.emitVectorSpeciesConstant(mv, numericContract, vectorWidth);
        mv.visitInsn(DUP_X2);
        mv.visitInsn(POP);
        if (dataType == DataType.FLOAT32 && numericContract.usesFloatCompute()) {
            mv.visitInsn(FALOAD);
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/FloatVector",
                    "broadcast",
                    "(Ljdk/incubator/vector/VectorSpecies;F)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else if (dataType == DataType.FLOAT64 && numericContract.usesDoubleCompute()) {
            mv.visitInsn(DALOAD);
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/DoubleVector",
                    "broadcast",
                    "(Ljdk/incubator/vector/VectorSpecies;D)Ljdk/incubator/vector/DoubleVector;",
                    false
            );
        } else {
            throw new UnsupportedOperationException("Broadcast vector loads require storage dtype to match fused compute kind.");
        }
    }

    static void emitDirectLinearSegmentVectorLoad(MethodVisitor mv, FusedNumericContract numericContract, int vectorWidth) {
        FusedVectorBytecode.emitVectorSpeciesConstant(mv, numericContract, vectorWidth);
        mv.visitInsn(DUP_X2);
        mv.visitInsn(POP);
        emitElementIndexToByteOffset(mv, numericContract);
        mv.visitMethodInsn(INVOKESTATIC, "java/nio/ByteOrder", "nativeOrder", "()Ljava/nio/ByteOrder;", false);
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/FloatVector",
                    "fromMemorySegment",
                    "(Ljdk/incubator/vector/VectorSpecies;Ljava/lang/foreign/MemorySegment;JLjava/nio/ByteOrder;)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/DoubleVector",
                    "fromMemorySegment",
                    "(Ljdk/incubator/vector/VectorSpecies;Ljava/lang/foreign/MemorySegment;JLjava/nio/ByteOrder;)Ljdk/incubator/vector/DoubleVector;",
                    false
            );
        }
    }

    static void emitBroadcastSegmentVectorLoad(MethodVisitor mv, DataType dataType, FusedNumericContract numericContract, int vectorWidth) {
        FusedVectorBytecode.emitVectorSpeciesConstant(mv, numericContract, vectorWidth);
        mv.visitInsn(DUP_X2);
        mv.visitInsn(POP);
        emitLoadScalarFromSegmentCall(mv, dataType, numericContract);
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/FloatVector",
                    "broadcast",
                    "(Ljdk/incubator/vector/VectorSpecies;F)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/DoubleVector",
                    "broadcast",
                    "(Ljdk/incubator/vector/VectorSpecies;D)Ljdk/incubator/vector/DoubleVector;",
                    false
            );
        }
    }

    static void emitLoadBoolVectorFromArrayCall(MethodVisitor mv, FusedNumericContract numericContract) {
        throw new UnsupportedOperationException("BOOL vector loads must fall back before vector bytecode generation.");
    }

    static void emitStoreVectorToArrayCall(MethodVisitor mv, DataType dataType, FusedNumericContract numericContract) {
        if (dataType == DataType.FLOAT32 && numericContract.usesFloatCompute()) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/FloatVector");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/FloatVector", "intoArray", "([FI)V", false);
        } else if (dataType == DataType.FLOAT64 && numericContract.usesDoubleCompute()) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/DoubleVector");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/DoubleVector", "intoArray", "([DI)V", false);
        } else {
            throw new UnsupportedOperationException(dataType + " vector array store is not supported for fused numeric values.");
        }
    }

    static void emitDirectStoreVectorToArrayCall(MethodVisitor mv, FusedNumericContract numericContract) {
        if (numericContract.usesFloatCompute()) {
            emitDirectStoreF32VectorToArrayCall(mv);
        } else {
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/DoubleVector", "intoArray", "([DI)V", false);
        }
    }

    static void emitDirectStoreF32VectorToArrayCall(MethodVisitor mv) {
        mv.visitInsn(DUP_X2);
        mv.visitInsn(POP);
        mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/FloatVector", "intoArray", "([FI)V", false);
    }

    static void emitDirectStoreVectorToSegmentCall(MethodVisitor mv, FusedNumericContract numericContract) {
        emitElementIndexToByteOffset(mv, numericContract);
        mv.visitMethodInsn(INVOKESTATIC, "java/nio/ByteOrder", "nativeOrder", "()Ljava/nio/ByteOrder;", false);
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(
                    INVOKEVIRTUAL,
                    "jdk/incubator/vector/FloatVector",
                    "intoMemorySegment",
                    "(Ljava/lang/foreign/MemorySegment;JLjava/nio/ByteOrder;)V",
                    false
            );
        } else {
            mv.visitMethodInsn(
                    INVOKEVIRTUAL,
                    "jdk/incubator/vector/DoubleVector",
                    "intoMemorySegment",
                    "(Ljava/lang/foreign/MemorySegment;JLjava/nio/ByteOrder;)V",
                    false
            );
        }
    }

    static void emitStoreBoolVectorToArrayCall(MethodVisitor mv, FusedNumericContract numericContract) {
        throw new UnsupportedOperationException("BOOL vector stores must fall back before vector bytecode generation.");
    }

    static String vectorTypeDesc(FusedNumericContract numericContract) {
        return numericContract.usesFloatCompute()
                ? "Ljdk/incubator/vector/FloatVector;"
                : "Ljdk/incubator/vector/DoubleVector;";
    }

    static String maskTypeDesc() {
        return "Ljdk/incubator/vector/VectorMask;";
    }

    private static void emitElementIndexToByteOffset(MethodVisitor mv, FusedNumericContract numericContract) {
        emitElementIndexToByteOffset(mv, numericContract.usesDoubleCompute() ? Double.BYTES : Float.BYTES);
    }

    private static void emitElementIndexToByteOffset(MethodVisitor mv, int bytesPerElement) {
        mv.visitInsn(I2L);
        mv.visitLdcInsn((long) bytesPerElement);
        mv.visitInsn(LMUL);
    }

    private static void emitSegmentGet(MethodVisitor mv, String layoutField, String layoutDesc, String valueDesc) {
        mv.visitFieldInsn(GETSTATIC, "java/lang/foreign/ValueLayout", layoutField, layoutDesc);
        mv.visitInsn(DUP_X2);
        mv.visitInsn(POP);
        mv.visitMethodInsn(
                INVOKEINTERFACE,
                "java/lang/foreign/MemorySegment",
                "get",
                "(" + layoutDesc + "J)" + valueDesc,
                true
        );
    }

    private static void emitSegmentSetPrefix(MethodVisitor mv, String layoutField, String layoutDesc) {
        mv.visitFieldInsn(GETSTATIC, "java/lang/foreign/ValueLayout", layoutField, layoutDesc);
        mv.visitInsn(DUP_X2);
        mv.visitInsn(POP);
    }

    private static void emitSegmentSet(MethodVisitor mv, String layoutDesc, String valueDesc) {
        mv.visitMethodInsn(
                INVOKEINTERFACE,
                "java/lang/foreign/MemorySegment",
                "set",
                "(" + layoutDesc + "J" + valueDesc + ")V",
                true
        );
    }
}
