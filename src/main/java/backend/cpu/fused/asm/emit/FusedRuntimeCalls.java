package backend.cpu.fused.asm.emit;

import backend.cpu.fused.runtime.FusedDTypeOps;
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

    static void emitLoadScalarFromSegmentCall(MethodVisitor mv, DataType dataType, int precisionMode) {
        switch (dataType) {
            case FLOAT32 -> {
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadScalarF32Segment", "(Ljava/lang/foreign/MemorySegment;I)F", false);
                if (precisionMode != FusedDTypeOps.MODE_F32) {
                    mv.visitInsn(F2D);
                }
            }
            case FLOAT64 -> {
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadScalarF64Segment", "(Ljava/lang/foreign/MemorySegment;I)D", false);
                if (precisionMode == FusedDTypeOps.MODE_F32) {
                    mv.visitInsn(D2F);
                }
            }
            case BFLOAT16 -> {
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadScalarBF16Segment", "(Ljava/lang/foreign/MemorySegment;I)D", false);
                if (precisionMode == FusedDTypeOps.MODE_F32) {
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

    static void emitStoreScalarToSegmentCall(MethodVisitor mv, DataType dataType, int precisionMode) {
        switch (dataType) {
            case FLOAT32 -> {
                if (precisionMode != FusedDTypeOps.MODE_F32) {
                    mv.visitInsn(D2F);
                }
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeScalarF32Segment", "(Ljava/lang/foreign/MemorySegment;IF)V", false);
            }
            case FLOAT64 -> {
                if (precisionMode == FusedDTypeOps.MODE_F32) {
                    mv.visitInsn(F2D);
                }
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeScalarF64Segment", "(Ljava/lang/foreign/MemorySegment;ID)V", false);
            }
            case BFLOAT16 -> {
                if (precisionMode == FusedDTypeOps.MODE_F32) {
                    mv.visitInsn(F2D);
                }
                mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeScalarBF16Segment", "(Ljava/lang/foreign/MemorySegment;ID)V", false);
            }
            case BOOL -> mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeScalarBoolSegment", "(Ljava/lang/foreign/MemorySegment;II)V", false);
            case INT32, INT64 -> throw new UnsupportedOperationException("INT32/INT64 segment scalar stores are not supported for fused execution.");
        }
    }

    static void emitLoadVectorFromArrayCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadVectorF32Array", "([FII)Ljdk/incubator/vector/FloatVector;", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadVectorF64Array", "([DII)Ljdk/incubator/vector/DoubleVector;", false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadVectorBF16Array", "([SII)Ljava/lang/Object;", false);
        }
    }

    static void emitDirectLinearVectorLoad(MethodVisitor mv, int precisionMode, int vectorWidth) {
        FusedVectorBytecode.emitVectorSpeciesConstant(mv, precisionMode, vectorWidth);
        mv.visitInsn(DUP_X2);
        mv.visitInsn(POP);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/FloatVector",
                    "fromArray",
                    "(Ljdk/incubator/vector/VectorSpecies;[FI)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/DoubleVector",
                    "fromArray",
                    "(Ljdk/incubator/vector/VectorSpecies;[DI)Ljdk/incubator/vector/DoubleVector;",
                    false
            );
        } else if (precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/FloatVector",
                    "fromArray",
                    "(Ljdk/incubator/vector/VectorSpecies;[FI)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else {
            throw new UnsupportedOperationException("Direct linear vector loads are supported only for F32/F64 fused modes.");
        }
    }

    static void emitDirectLinearSegmentVectorLoad(MethodVisitor mv, int precisionMode, int vectorWidth) {
        FusedVectorBytecode.emitVectorSpeciesConstant(mv, precisionMode, vectorWidth);
        mv.visitInsn(DUP_X2);
        mv.visitInsn(POP);
        emitElementIndexToByteOffset(mv, precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "java/nio/ByteOrder", "nativeOrder", "()Ljava/nio/ByteOrder;", false);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/FloatVector",
                    "fromMemorySegment",
                    "(Ljdk/incubator/vector/VectorSpecies;Ljava/lang/foreign/MemorySegment;JLjava/nio/ByteOrder;)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/DoubleVector",
                    "fromMemorySegment",
                    "(Ljdk/incubator/vector/VectorSpecies;Ljava/lang/foreign/MemorySegment;JLjava/nio/ByteOrder;)Ljdk/incubator/vector/DoubleVector;",
                    false
            );
        } else {
            throw new UnsupportedOperationException("Direct segment vector loads are supported only for F32/F64 fused modes.");
        }
    }

    static void emitBroadcastSegmentVectorLoad(MethodVisitor mv, DataType dataType, int precisionMode, int vectorWidth) {
        FusedVectorBytecode.emitVectorSpeciesConstant(mv, precisionMode, vectorWidth);
        mv.visitInsn(DUP_X2);
        mv.visitInsn(POP);
        emitLoadScalarFromSegmentCall(mv, dataType, precisionMode);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/FloatVector",
                    "broadcast",
                    "(Ljdk/incubator/vector/VectorSpecies;F)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/DoubleVector",
                    "broadcast",
                    "(Ljdk/incubator/vector/VectorSpecies;D)Ljdk/incubator/vector/DoubleVector;",
                    false
            );
        } else {
            throw new UnsupportedOperationException("Segment broadcast vector loads are supported only for F32/F64 fused modes.");
        }
    }

    static void emitLoadBoolVectorFromArrayCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32 || precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadMaskF32Array", "([BII)Ljava/lang/Object;", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadMaskF64Array", "([BII)Ljava/lang/Object;", false);
        } else {
            throw new UnsupportedOperationException("BOOL vector loads are supported only for F32/F64 fused modes.");
        }
    }

    static void emitLoadVectorFromCursorCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadVectorF32",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[FI)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadVectorF64",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[DI)Ljdk/incubator/vector/DoubleVector;",
                    false
            );
        } else {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadVectorBF16",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[SI)Ljava/lang/Object;",
                    false
            );
        }
    }

    static void emitLoadVectorFromContinuationCursorCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32 || precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadVectorF32",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[FI)Ljdk/incubator/vector/FloatVector;",
                    false
            );
            return;
        }
        throw new UnsupportedOperationException("Continuation cursor vector loads are supported only for F32/BF16 fused modes.");
    }

    static void emitLoadBoolVectorFromCursorCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32 || precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadMaskF32",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[BI)Ljava/lang/Object;",
                    false
            );
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastVectorOps",
                    "loadMaskF64",
                    "(Lbackend/cpu/fused/runtime/FusedBroadcastCursor;[BI)Ljava/lang/Object;",
                    false
            );
        } else {
            throw new UnsupportedOperationException("BOOL vector cursor loads are supported only for F32/F64 fused modes.");
        }
    }

    static void emitStoreVectorToArrayCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/FloatVector");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/FloatVector", "intoArray", "([FI)V", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/DoubleVector");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/DoubleVector", "intoArray", "([DI)V", false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeVectorBF16Array", "([SILjava/lang/Object;)V", false);
        }
    }

    static void emitDirectStoreVectorToArrayCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/FloatVector", "intoArray", "([FI)V", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/DoubleVector", "intoArray", "([DI)V", false);
        } else {
            throw new UnsupportedOperationException("Direct vector stores are supported only for F32/F64 fused modes.");
        }
    }

    static void emitDirectStoreVectorToSegmentCall(MethodVisitor mv, int precisionMode) {
        emitElementIndexToByteOffset(mv, precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "java/nio/ByteOrder", "nativeOrder", "()Ljava/nio/ByteOrder;", false);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(
                    INVOKEVIRTUAL,
                    "jdk/incubator/vector/FloatVector",
                    "intoMemorySegment",
                    "(Ljava/lang/foreign/MemorySegment;JLjava/nio/ByteOrder;)V",
                    false
            );
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(
                    INVOKEVIRTUAL,
                    "jdk/incubator/vector/DoubleVector",
                    "intoMemorySegment",
                    "(Ljava/lang/foreign/MemorySegment;JLjava/nio/ByteOrder;)V",
                    false
            );
        } else {
            throw new UnsupportedOperationException("Direct segment vector stores are supported only for F32/F64 fused modes.");
        }
    }

    static void emitStoreBoolVectorToArrayCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32 || precisionMode == FusedDTypeOps.MODE_BF16) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeMaskF32Array", "([BILjava/lang/Object;I)V", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeMaskF64Array", "([BILjava/lang/Object;I)V", false);
        } else {
            throw new UnsupportedOperationException("BOOL vector stores are supported only for F32/F64 fused modes.");
        }
    }

    static String vectorTypeDesc(int precisionMode) {
        return precisionMode == FusedDTypeOps.MODE_F32 || precisionMode == FusedDTypeOps.MODE_BF16
                ? "Ljdk/incubator/vector/FloatVector;"
                : "Ljdk/incubator/vector/DoubleVector;";
    }

    static String maskTypeDesc() {
        return "Ljdk/incubator/vector/VectorMask;";
    }

    private static void emitElementIndexToByteOffset(MethodVisitor mv, int precisionMode) {
        mv.visitInsn(I2L);
        mv.visitLdcInsn(precisionMode == FusedDTypeOps.MODE_F64 ? (long) Double.BYTES : (long) Float.BYTES);
        mv.visitInsn(LMUL);
    }
}
