package backend.cpu.fused.asm.emit;

import backend.cpu.fused.numeric.FusedNumericContract;
import org.objectweb.asm.MethodVisitor;
import tensor.DataType;

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
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadScalarF32Segment", "(Ljava/lang/foreign/MemorySegment;I)F", false);
                if (numericContract.usesDoubleCompute()) {
                    mv.visitInsn(F2D);
                }
            }
            case FLOAT64 -> {
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadScalarF64Segment", "(Ljava/lang/foreign/MemorySegment;I)D", false);
                if (numericContract.usesFloatCompute()) {
                    mv.visitInsn(D2F);
                }
            }
            case BFLOAT16 -> {
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadScalarBF16Segment", "(Ljava/lang/foreign/MemorySegment;I)D", false);
                if (numericContract.usesFloatCompute()) {
                    mv.visitInsn(D2F);
                }
            }
            case BOOL -> emitLoadBoolFromSegmentCall(mv);
            case INT32, INT64 -> throw new UnsupportedOperationException("INT32/INT64 segment scalar loads are not supported for fused execution.");
        }
    }

    static void emitLoadBoolFromSegmentCall(MethodVisitor mv) {
        mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadScalarBoolSegment", "(Ljava/lang/foreign/MemorySegment;I)I", false);
    }

    static void emitStoreScalarToSegmentCall(MethodVisitor mv, DataType dataType, FusedNumericContract numericContract) {
        switch (dataType) {
            case FLOAT32 -> {
                if (numericContract.usesDoubleCompute()) {
                    mv.visitInsn(D2F);
                }
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeScalarF32Segment", "(Ljava/lang/foreign/MemorySegment;IF)V", false);
            }
            case FLOAT64 -> {
                if (numericContract.usesFloatCompute()) {
                    mv.visitInsn(F2D);
                }
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeScalarF64Segment", "(Ljava/lang/foreign/MemorySegment;ID)V", false);
            }
            case BFLOAT16 -> {
                if (numericContract.usesFloatCompute()) {
                    mv.visitInsn(F2D);
                }
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeScalarBF16Segment", "(Ljava/lang/foreign/MemorySegment;ID)V", false);
            }
            case BOOL -> mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeScalarBoolSegment", "(Ljava/lang/foreign/MemorySegment;II)V", false);
            case INT32, INT64 -> throw new UnsupportedOperationException("INT32/INT64 segment scalar stores are not supported for fused execution.");
        }
    }

    static void emitLoadVectorFromArrayCall(MethodVisitor mv, DataType dataType, FusedNumericContract numericContract) {
        switch (dataType) {
            case FLOAT32 -> mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadVectorF32Array", "([FII)Ljdk/incubator/vector/FloatVector;", false);
            case FLOAT64 -> mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadVectorF64Array", "([DII)Ljdk/incubator/vector/DoubleVector;", false);
            case BFLOAT16 -> mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadVectorBF16Array", "([SII)Ljava/lang/Object;", false);
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException(
                    dataType + " vector array load is not supported for fused numeric values."
            );
        }
    }

    static void emitDirectLinearVectorLoad(MethodVisitor mv, DataType dataType, FusedNumericContract numericContract, int vectorWidth) {
        if (dataType == DataType.BFLOAT16) {
            FusedVectorBytecode.emitVectorWidthConstant(mv, vectorWidth);
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadVectorBF16Array", "([SII)Ljava/lang/Object;", false);
            return;
        }
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
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadMaskF32Array", "([BII)Ljava/lang/Object;", false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadMaskF64Array", "([BII)Ljava/lang/Object;", false);
        }
    }

    static void emitLoadVectorFromCursorCall(MethodVisitor mv, DataType dataType, FusedNumericContract numericContract) {
        if (dataType == DataType.FLOAT32 && numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadVectorF32",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[FI)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else if (dataType == DataType.FLOAT64 && numericContract.usesDoubleCompute()) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadVectorF64",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[DI)Ljdk/incubator/vector/DoubleVector;",
                    false
            );
        } else if (dataType == DataType.BFLOAT16) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadVectorBF16",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[SI)Ljava/lang/Object;",
                    false
            );
        } else {
            throw new UnsupportedOperationException("Cursor vector loads require storage dtype to match fused compute kind.");
        }
    }

    static void emitLoadVectorFromContinuationCursorCall(MethodVisitor mv, FusedNumericContract numericContract) {
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadVectorF32",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[FI)Ljdk/incubator/vector/FloatVector;",
                    false
            );
            return;
        }
        throw new UnsupportedOperationException("Continuation cursor vector loads are supported only for F32 fused compute.");
    }

    static void emitLoadBoolVectorFromCursorCall(MethodVisitor mv, FusedNumericContract numericContract) {
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadMaskF32",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[BI)Ljava/lang/Object;",
                    false
            );
        } else {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadMaskF64",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[BI)Ljava/lang/Object;",
                    false
            );
        }
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
        } else if (dataType == DataType.BFLOAT16) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeVectorBF16Array", "([SILjava/lang/Object;)V", false);
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
        if (numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeMaskF32Array", "([BILjava/lang/Object;I)V", false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeMaskF64Array", "([BILjava/lang/Object;I)V", false);
        }
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
        mv.visitInsn(I2L);
        mv.visitLdcInsn(numericContract.usesDoubleCompute() ? (long) Double.BYTES : (long) Float.BYTES);
        mv.visitInsn(LMUL);
    }
}
