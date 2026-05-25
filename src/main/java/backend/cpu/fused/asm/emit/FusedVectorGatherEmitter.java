package backend.cpu.fused.asm.emit;

import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.numeric.FusedNumericContract;
import org.objectweb.asm.MethodVisitor;
import tensor.DataType;
import utils.SlotKey;
import utils.SlotManager;

import static org.objectweb.asm.Opcodes.*;

/**
 * Emits generated vector gather loads for non-linear F32/F64 fused inputs.
 */
final class FusedVectorGatherEmitter {
    private FusedVectorGatherEmitter() {}

    static boolean requiresGather(FusedExternalInputPlan input) {
        return !input.isLinearAccess() && !isZeroStrideBroadcast(input);
    }

    static void emitIndexMap(
            MethodVisitor mv,
            FusedExternalInputPlan input,
            SlotManager sm,
            int indexMapSlot,
            int vectorWidth
    ) {
        int[] denseStrides = input.logicalOutputDenseStrides();
        int[] effectiveStrides = input.effectiveStrides();
        for (int lane = 0; lane < vectorWidth; lane++) {
            emitLaneStorageIndex(mv, input.storageOffset(), denseStrides, effectiveStrides, sm, lane);
            mv.visitVarInsn(ALOAD, indexMapSlot);
            mv.visitLdcInsn(lane);
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.FUSED_VECTOR_STORAGE_INDEX));
            mv.visitInsn(IASTORE);
        }
    }

    static void emitArrayGatherLoad(
            MethodVisitor mv,
            int inputSlot,
            int indexMapSlot,
            DataType dataType,
            FusedNumericContract numericContract,
            int vectorWidth
    ) {
        FusedVectorBytecode.emitVectorSpeciesConstant(mv, numericContract, vectorWidth);
        mv.visitVarInsn(ALOAD, inputSlot);
        mv.visitInsn(ICONST_0);
        mv.visitVarInsn(ALOAD, indexMapSlot);
        mv.visitInsn(ICONST_0);
        if (dataType == DataType.FLOAT32 && numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/FloatVector",
                    "fromArray",
                    "(Ljdk/incubator/vector/VectorSpecies;[FI[II)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else if (dataType == DataType.FLOAT64 && numericContract.usesDoubleCompute()) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/DoubleVector",
                    "fromArray",
                    "(Ljdk/incubator/vector/VectorSpecies;[DI[II)Ljdk/incubator/vector/DoubleVector;",
                    false
            );
        } else {
            throw new UnsupportedOperationException("Generated vector gather requires storage dtype to match fused compute kind.");
        }
    }

    static void emitSegmentGatherLoad(
            MethodVisitor mv,
            int segmentSlot,
            int indexMapSlot,
            int laneScratchSlot,
            DataType dataType,
            FusedNumericContract numericContract,
            int vectorWidth
    ) {
        for (int lane = 0; lane < vectorWidth; lane++) {
            mv.visitVarInsn(ALOAD, laneScratchSlot);
            mv.visitLdcInsn(lane);
            mv.visitVarInsn(ALOAD, segmentSlot);
            mv.visitVarInsn(ALOAD, indexMapSlot);
            mv.visitLdcInsn(lane);
            mv.visitInsn(IALOAD);
            FusedRuntimeCalls.emitLoadScalarFromSegmentCall(mv, dataType, numericContract);
            if (dataType == DataType.FLOAT32 && numericContract.usesFloatCompute()) {
                mv.visitInsn(FASTORE);
            } else if (dataType == DataType.FLOAT64 && numericContract.usesDoubleCompute()) {
                mv.visitInsn(DASTORE);
            } else {
                throw new UnsupportedOperationException("Generated segment vector gather requires storage dtype to match fused compute kind.");
            }
        }

        FusedVectorBytecode.emitVectorSpeciesConstant(mv, numericContract, vectorWidth);
        mv.visitVarInsn(ALOAD, laneScratchSlot);
        mv.visitInsn(ICONST_0);
        if (dataType == DataType.FLOAT32 && numericContract.usesFloatCompute()) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/FloatVector",
                    "fromArray",
                    "(Ljdk/incubator/vector/VectorSpecies;[FI)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else if (dataType == DataType.FLOAT64 && numericContract.usesDoubleCompute()) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "jdk/incubator/vector/DoubleVector",
                    "fromArray",
                    "(Ljdk/incubator/vector/VectorSpecies;[DI)Ljdk/incubator/vector/DoubleVector;",
                    false
            );
        } else {
            throw new UnsupportedOperationException("Generated segment vector gather requires storage dtype to match fused compute kind.");
        }
    }

    private static void emitLaneStorageIndex(
            MethodVisitor mv,
            int storageOffset,
            int[] denseStrides,
            int[] effectiveStrides,
            SlotManager sm,
            int lane
    ) {
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        if (lane != 0) {
            mv.visitLdcInsn(lane);
            mv.visitInsn(IADD);
        }
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.FUSED_VECTOR_REMAIN));

        mv.visitLdcInsn(storageOffset);
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.FUSED_VECTOR_STORAGE_INDEX));

        for (int dim = 0; dim < denseStrides.length; dim++) {
            int denseStride = denseStrides[dim];
            int effectiveStride = effectiveStrides[dim];
            if (denseStride != 0 && effectiveStride != 0) {
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.FUSED_VECTOR_STORAGE_INDEX));
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.FUSED_VECTOR_REMAIN));
                if (denseStride != 1) {
                    mv.visitLdcInsn(denseStride);
                    mv.visitInsn(IDIV);
                }
                if (effectiveStride != 1) {
                    mv.visitLdcInsn(effectiveStride);
                    mv.visitInsn(IMUL);
                }
                mv.visitInsn(IADD);
                mv.visitVarInsn(ISTORE, sm.get(SlotKey.FUSED_VECTOR_STORAGE_INDEX));
            }
            if (dim + 1 < denseStrides.length && denseStride != 0) {
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.FUSED_VECTOR_REMAIN));
                if (denseStride == 1) {
                    mv.visitInsn(ICONST_0);
                } else {
                    mv.visitLdcInsn(denseStride);
                    mv.visitInsn(IREM);
                }
                mv.visitVarInsn(ISTORE, sm.get(SlotKey.FUSED_VECTOR_REMAIN));
            }
        }
    }

    private static boolean isZeroStrideBroadcast(FusedExternalInputPlan input) {
        for (int stride : input.effectiveStrides()) {
            if (stride != 0) {
                return false;
            }
        }
        return true;
    }
}
