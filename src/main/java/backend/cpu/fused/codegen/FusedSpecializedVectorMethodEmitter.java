package backend.cpu.fused.codegen;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;

import static backend.cpu.fused.codegen.FusedMethodDescriptors.RANGE_METHOD_DESC;
import static org.objectweb.asm.Opcodes.ACC_PUBLIC;
import static org.objectweb.asm.Opcodes.ALOAD;
import static org.objectweb.asm.Opcodes.ASTORE;
import static org.objectweb.asm.Opcodes.BALOAD;
import static org.objectweb.asm.Opcodes.CHECKCAST;
import static org.objectweb.asm.Opcodes.GOTO;
import static org.objectweb.asm.Opcodes.IADD;
import static org.objectweb.asm.Opcodes.IFEQ;
import static org.objectweb.asm.Opcodes.IF_ICMPGE;
import static org.objectweb.asm.Opcodes.ILOAD;
import static org.objectweb.asm.Opcodes.INVOKEINTERFACE;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;
import static org.objectweb.asm.Opcodes.INVOKEVIRTUAL;
import static org.objectweb.asm.Opcodes.IREM;
import static org.objectweb.asm.Opcodes.ISUB;
import static org.objectweb.asm.Opcodes.ISTORE;
import static org.objectweb.asm.Opcodes.LCONST_0;
import static org.objectweb.asm.Opcodes.LLOAD;
import static org.objectweb.asm.Opcodes.LOR;
import static org.objectweb.asm.Opcodes.LSTORE;
import static org.objectweb.asm.Opcodes.RETURN;

/**
 * Internal ASM emitter for specialized vector fused-kernel fast paths.
 */
public final class FusedSpecializedVectorMethodEmitter {
    private static final int MASK_ARRAY_SLOT = 7;
    private static final int FILL_ARRAY_SLOT = 8;
    private static final int VALUE_ARRAY_SLOT = 9;
    private static final int OUT_ARRAY_SLOT = 10;
    private static final int OUT_BASE_SLOT = 11;
    private static final int SPECIES_SLOT = 12;
    private static final int FILL_VEC_SLOT = 13;
    private static final int SCALE_VEC_SLOT = 14;
    private static final int RANGE_UPPER_SLOT = 15;
    private static final int LOOP_COUNTER_SLOT = 16;
    private static final int MASK_VEC_SLOT = 17;
    private static final int VALUE_VEC_SLOT = 18;
    private static final int RESULT_VEC_SLOT = 19;
    private static final int MASK_BITS_SLOT = 20;

    private FusedSpecializedVectorMethodEmitter() {}

    public static void emit(ClassWriter cw, FusedGenerationContext context) {
        switch (context.specializationKind()) {
            case F32_MASKED_SCALE_WHERE -> emitMaskedScaleWhereF32(cw, context, true);
            case F32_MASKED_SCALE_WHERE_INVERTED -> emitMaskedScaleWhereF32(cw, context, false);
            case NONE -> throw new IllegalArgumentException("Specialized emitter requires a specialization kind.");
        }
    }

    private static void emitMaskedScaleWhereF32(
            ClassWriter cw,
            FusedGenerationContext context,
            boolean fillWhenMaskTrue
    ) {
        MethodVisitor mv = cw.visitMethod(
                ACC_PUBLIC,
                "applyRangeVector",
                RANGE_METHOD_DESC,
                null,
                null
        );
        mv.visitCode();

        if (context.vectorWidth() <= 1) {
            emitScalarDelegate(mv, context);
            return;
        }

        FusedExternalInputPlan maskInput = context.plan().inputs().get(0);
        FusedExternalInputPlan fillInput = context.plan().inputs().get(1);
        FusedExternalInputPlan valueInput = context.plan().inputs().get(2);
        float scale = FusedAsmSpecializationMatcher.requireF32MaskedScaleWhereScalar(context.plan());

        emitInputArrayBinding(mv, 0, "getBoolData", "()[B", MASK_ARRAY_SLOT);
        emitInputArrayBinding(mv, 1, "getFloat32Data", "()[F", FILL_ARRAY_SLOT);
        emitInputArrayBinding(mv, 2, "getFloat32Data", "()[F", VALUE_ARRAY_SLOT);

        mv.visitVarInsn(ALOAD, 2);
        mv.visitMethodInsn(INVOKEVIRTUAL, "tensor/Tensor", "getFloat32Data", "()[F", false);
        mv.visitVarInsn(ASTORE, OUT_ARRAY_SLOT);
        mv.visitVarInsn(ALOAD, 2);
        mv.visitMethodInsn(INVOKEVIRTUAL, "tensor/Tensor", "getStorageOffsetUnsafe", "()I", false);
        mv.visitVarInsn(ISTORE, OUT_BASE_SLOT);

        FusedAsmSupport.emitVectorSpeciesConstant(mv, FusedDTypeOps.MODE_F32, context.vectorWidth());
        mv.visitVarInsn(ASTORE, SPECIES_SLOT);

        mv.visitVarInsn(ALOAD, SPECIES_SLOT);
        mv.visitVarInsn(ALOAD, FILL_ARRAY_SLOT);
        emitLinearIndex(mv, fillInput.storageOffset(), 0);
        mv.visitInsn(org.objectweb.asm.Opcodes.FALOAD);
        mv.visitMethodInsn(
                INVOKESTATIC,
                "jdk/incubator/vector/FloatVector",
                "broadcast",
                "(Ljdk/incubator/vector/VectorSpecies;F)Ljdk/incubator/vector/FloatVector;",
                false
        );
        mv.visitVarInsn(ASTORE, FILL_VEC_SLOT);

        mv.visitVarInsn(ALOAD, SPECIES_SLOT);
        mv.visitLdcInsn(scale);
        mv.visitMethodInsn(
                INVOKESTATIC,
                "jdk/incubator/vector/FloatVector",
                "broadcast",
                "(Ljdk/incubator/vector/VectorSpecies;F)Ljdk/incubator/vector/FloatVector;",
                false
        );
        mv.visitVarInsn(ASTORE, SCALE_VEC_SLOT);

        mv.visitVarInsn(ILOAD, 5);
        mv.visitVarInsn(ILOAD, 5);
        mv.visitVarInsn(ILOAD, 4);
        mv.visitInsn(ISUB);
        FusedAsmSupport.emitVectorWidthConstant(mv, context.vectorWidth());
        mv.visitInsn(IREM);
        mv.visitInsn(ISUB);
        mv.visitVarInsn(ISTORE, RANGE_UPPER_SLOT);

        mv.visitVarInsn(ILOAD, 4);
        mv.visitVarInsn(ISTORE, LOOP_COUNTER_SLOT);

        Label loopStart = new Label();
        Label loopEnd = new Label();
        mv.visitLabel(loopStart);
        mv.visitVarInsn(ILOAD, LOOP_COUNTER_SLOT);
        mv.visitVarInsn(ILOAD, RANGE_UPPER_SLOT);
        mv.visitJumpInsn(IF_ICMPGE, loopEnd);

        emitDirectMaskLoad(mv, maskInput.storageOffset(), context.vectorWidth());

        mv.visitVarInsn(ALOAD, SPECIES_SLOT);
        mv.visitVarInsn(ALOAD, VALUE_ARRAY_SLOT);
        emitLoopIndex(mv, LOOP_COUNTER_SLOT, valueInput.storageOffset(), 0);
        mv.visitMethodInsn(
                INVOKESTATIC,
                "jdk/incubator/vector/FloatVector",
                "fromArray",
                "(Ljdk/incubator/vector/VectorSpecies;[FI)Ljdk/incubator/vector/FloatVector;",
                false
        );
        mv.visitVarInsn(ASTORE, VALUE_VEC_SLOT);

        mv.visitVarInsn(ALOAD, VALUE_VEC_SLOT);
        mv.visitVarInsn(ALOAD, SCALE_VEC_SLOT);
        mv.visitMethodInsn(
                INVOKEVIRTUAL,
                "jdk/incubator/vector/FloatVector",
                "mul",
                "(Ljdk/incubator/vector/Vector;)Ljdk/incubator/vector/FloatVector;",
                false
        );
        mv.visitVarInsn(ASTORE, RESULT_VEC_SLOT);

        if (fillWhenMaskTrue) {
            mv.visitVarInsn(ALOAD, RESULT_VEC_SLOT);
            mv.visitVarInsn(ALOAD, FILL_VEC_SLOT);
            mv.visitVarInsn(ALOAD, MASK_VEC_SLOT);
        } else {
            mv.visitVarInsn(ALOAD, FILL_VEC_SLOT);
            mv.visitVarInsn(ALOAD, RESULT_VEC_SLOT);
            mv.visitVarInsn(ALOAD, MASK_VEC_SLOT);
        }
        mv.visitMethodInsn(
                INVOKEVIRTUAL,
                "jdk/incubator/vector/FloatVector",
                "blend",
                "(Ljdk/incubator/vector/Vector;Ljdk/incubator/vector/VectorMask;)Ljdk/incubator/vector/FloatVector;",
                false
        );
        mv.visitVarInsn(ASTORE, RESULT_VEC_SLOT);

        mv.visitVarInsn(ALOAD, OUT_ARRAY_SLOT);
        mv.visitVarInsn(ILOAD, LOOP_COUNTER_SLOT);
        mv.visitVarInsn(ILOAD, OUT_BASE_SLOT);
        mv.visitInsn(IADD);
        mv.visitVarInsn(ALOAD, RESULT_VEC_SLOT);
        FusedAsmSupport.emitDirectStoreVectorToArrayCall(mv, FusedDTypeOps.MODE_F32);

        mv.visitVarInsn(ILOAD, LOOP_COUNTER_SLOT);
        FusedAsmSupport.emitVectorWidthConstant(mv, context.vectorWidth());
        mv.visitInsn(IADD);
        mv.visitVarInsn(ISTORE, LOOP_COUNTER_SLOT);
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);

        Label noTail = new Label();
        mv.visitVarInsn(ILOAD, RANGE_UPPER_SLOT);
        mv.visitVarInsn(ILOAD, 5);
        mv.visitJumpInsn(IF_ICMPGE, noTail);
        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ALOAD, 1);
        mv.visitVarInsn(ALOAD, 2);
        mv.visitVarInsn(ALOAD, 3);
        mv.visitVarInsn(ILOAD, RANGE_UPPER_SLOT);
        mv.visitVarInsn(ILOAD, 5);
        mv.visitVarInsn(ALOAD, 6);
        mv.visitMethodInsn(
                INVOKEVIRTUAL,
                context.internalClassName(),
                "applyRangeScalar",
                RANGE_METHOD_DESC,
                false
        );
        mv.visitLabel(noTail);

        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void emitDirectMaskLoad(MethodVisitor mv, int storageOffset, int vectorWidth) {
        mv.visitInsn(LCONST_0);
        mv.visitVarInsn(LSTORE, MASK_BITS_SLOT);
        for (int lane = 0; lane < vectorWidth; lane++) {
            Label laneUnset = new Label();
            mv.visitVarInsn(ALOAD, MASK_ARRAY_SLOT);
            emitLoopIndex(mv, LOOP_COUNTER_SLOT, storageOffset, lane);
            mv.visitInsn(BALOAD);
            mv.visitJumpInsn(IFEQ, laneUnset);
            mv.visitVarInsn(LLOAD, MASK_BITS_SLOT);
            mv.visitLdcInsn(1L << lane);
            mv.visitInsn(LOR);
            mv.visitVarInsn(LSTORE, MASK_BITS_SLOT);
            mv.visitLabel(laneUnset);
        }
        mv.visitVarInsn(ALOAD, SPECIES_SLOT);
        mv.visitVarInsn(LLOAD, MASK_BITS_SLOT);
        mv.visitMethodInsn(
                INVOKESTATIC,
                "jdk/incubator/vector/VectorMask",
                "fromLong",
                "(Ljdk/incubator/vector/VectorSpecies;J)Ljdk/incubator/vector/VectorMask;",
                false
        );
        mv.visitVarInsn(ASTORE, MASK_VEC_SLOT);
    }

    private static void emitLoopIndex(MethodVisitor mv, int indexSlot, int storageOffset, int laneOffset) {
        mv.visitVarInsn(ILOAD, indexSlot);
        int totalOffset = storageOffset + laneOffset;
        if (totalOffset != 0) {
            mv.visitLdcInsn(totalOffset);
            mv.visitInsn(IADD);
        }
    }

    private static void emitLinearIndex(MethodVisitor mv, int storageOffset, int laneOffset) {
        int totalOffset = storageOffset + laneOffset;
        if (totalOffset == 0) {
            mv.visitInsn(org.objectweb.asm.Opcodes.ICONST_0);
        } else {
            mv.visitLdcInsn(totalOffset);
        }
    }

    private static void emitInputArrayBinding(
            MethodVisitor mv,
            int inputIndex,
            String accessorName,
            String accessorDesc,
            int targetSlot
    ) {
        mv.visitVarInsn(ALOAD, 1);
        mv.visitLdcInsn(inputIndex);
        mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
        mv.visitTypeInsn(CHECKCAST, "tensor/Tensor");
        mv.visitMethodInsn(INVOKEVIRTUAL, "tensor/Tensor", accessorName, accessorDesc, false);
        mv.visitVarInsn(ASTORE, targetSlot);
    }

    private static void emitScalarDelegate(MethodVisitor mv, FusedGenerationContext context) {
        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ALOAD, 1);
        mv.visitVarInsn(ALOAD, 2);
        mv.visitVarInsn(ALOAD, 3);
        mv.visitVarInsn(ILOAD, 4);
        mv.visitVarInsn(ILOAD, 5);
        mv.visitVarInsn(ALOAD, 6);
        mv.visitMethodInsn(
                INVOKEVIRTUAL,
                context.internalClassName(),
                "applyRangeScalar",
                RANGE_METHOD_DESC,
                false
        );
        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }
}
