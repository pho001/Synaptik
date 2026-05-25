package backend.cpu.fused.asm.emit;

import backend.cpu.fused.asm.FusedGenerationContext;

import backend.cpu.fused.numeric.FusedStorageKind;

import backend.cpu.fused.ir.FusedExternalInputPlan;

import org.objectweb.asm.MethodVisitor;
import tensor.DataType;
import tensor.Tensor;
import utils.SlotKey;
import utils.SlotManager;

import java.util.List;

import static org.objectweb.asm.Opcodes.*;

/**
 * Internal ASM emitter for loading fused runtime inputs into generated method slots.
 */
public final class FusedInputBindingEmitter {
    private FusedInputBindingEmitter() {}

    public static void emitScalarBindings(
            MethodVisitor mv,
            FusedGenerationContext context,
            SlotManager sm
    ) {
        List<Integer> inputSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS);
        List<Integer> continuationSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_CONTINUATION_ARRAYS);
        for (int i = 0; i < context.plan().inputCount(); i++) {
            if (context.numericContract().inputStorageKind() == FusedStorageKind.CPU_MEMORY_SEGMENT) {
                mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_CONTEXT));
                mv.visitLdcInsn(i);
                FusedRuntimeCalls.emitGetNativeInputSegmentCall(mv);
            } else {
                mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
                mv.visitLdcInsn(i);
                mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
                mv.visitTypeInsn(CHECKCAST, "tensor/Tensor");
                FusedRuntimeCalls.emitGetRawArrayFromTensorCall(mv, context.plan().inputs().get(i).dataType());
            }
            mv.visitVarInsn(ASTORE, inputSlots.get(i));

            if (context.numericContract().inputStorageKind() == FusedStorageKind.CPU_JAVA_ARRAY
                    && context.plan().inputs().get(i).dataType() == DataType.BFLOAT16) {
                mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_CONTEXT));
                mv.visitLdcInsn(i);
                mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
                mv.visitLdcInsn(i);
                mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
                mv.visitTypeInsn(CHECKCAST, "tensor/Tensor");
                mv.visitMethodInsn(INVOKEVIRTUAL, "tensor/Tensor", "getFlatDataSize", "()I", false);
                mv.visitMethodInsn(INVOKEVIRTUAL, "backend/cpu/kernels/CpuKernelContext", "inputFloatContinuation", "(II)[F", false);
            } else {
                mv.visitInsn(ACONST_NULL);
            }
            mv.visitVarInsn(ASTORE, continuationSlots.get(i));
        }
    }

    public static void emitScalarCursorBindings(
            MethodVisitor mv,
            List<FusedExternalInputPlan> inputAccess,
            SlotManager sm
    ) {
        List<Integer> cursorSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS);
        for (int i = 0; i < inputAccess.size(); i++) {
            FusedExternalInputPlan meta = inputAccess.get(i);
            if (!meta.usesCursor()) {
                continue;
            }
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
            FusedScalarBytecode.emitIntArrayConstant(mv, meta.logicalOutputShape());
            FusedScalarBytecode.emitIntArrayConstant(mv, meta.logicalOutputDenseStrides());
            FusedScalarBytecode.emitIntArrayConstant(mv, meta.effectiveStrides());
            mv.visitLdcInsn(meta.storageOffset());
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastCursor",
                    "atStart",
                    "(I[I[I[II)Lbackend/cpu/fused/runtime/FusedBroadcastCursor;",
                    false
            );
            mv.visitVarInsn(ASTORE, cursorSlots.get(i));
        }
    }

    public static void emitVectorBindings(
            MethodVisitor mv,
            FusedGenerationContext context,
            SlotManager sm
    ) {
        List<Integer> inputSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS);
        List<Integer> continuationSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_CONTINUATION_ARRAYS);
        for (int i = 0; i < context.plan().inputCount(); i++) {
            if (context.numericContract().inputStorageKind() == FusedStorageKind.CPU_MEMORY_SEGMENT) {
                mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_CONTEXT));
                mv.visitLdcInsn(i);
                FusedRuntimeCalls.emitGetNativeInputSegmentCall(mv);
            } else {
                mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
                mv.visitLdcInsn(i);
                mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
                mv.visitTypeInsn(CHECKCAST, "tensor/Tensor");
                FusedRuntimeCalls.emitGetRawArrayFromTensorCall(mv, context.plan().inputs().get(i).dataType());
            }
            mv.visitVarInsn(ASTORE, inputSlots.get(i));

            if (context.numericContract().inputStorageKind() == FusedStorageKind.CPU_JAVA_ARRAY
                    && context.plan().inputs().get(i).dataType() == DataType.BFLOAT16) {
                mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_CONTEXT));
                mv.visitLdcInsn(i);
                mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
                mv.visitLdcInsn(i);
                mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
                mv.visitTypeInsn(CHECKCAST, "tensor/Tensor");
                mv.visitMethodInsn(INVOKEVIRTUAL, "tensor/Tensor", "getFlatDataSize", "()I", false);
                mv.visitMethodInsn(INVOKEVIRTUAL, "backend/cpu/kernels/CpuKernelContext", "inputFloatContinuation", "(II)[F", false);
            } else {
                mv.visitInsn(ACONST_NULL);
            }
            mv.visitVarInsn(ASTORE, continuationSlots.get(i));
        }
    }

    public static void emitVectorCursorBindings(
            MethodVisitor mv,
            List<FusedExternalInputPlan> inputAccess,
            SlotManager sm,
            boolean memorySegmentStorage
    ) {
        List<Integer> cursorSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS);
        for (int i = 0; i < inputAccess.size(); i++) {
            FusedExternalInputPlan meta = inputAccess.get(i);
            if (!meta.usesCursor()) {
                continue;
            }
            if (isZeroStrideBroadcast(meta)) {
                continue;
            }
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
            FusedScalarBytecode.emitIntArrayConstant(mv, meta.logicalOutputShape());
            FusedScalarBytecode.emitIntArrayConstant(mv, meta.logicalOutputDenseStrides());
            FusedScalarBytecode.emitIntArrayConstant(mv, meta.effectiveStrides());
            mv.visitLdcInsn(meta.storageOffset());
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "backend/cpu/fused/runtime/FusedBroadcastCursor",
                    "atStart",
                    "(I[I[I[II)Lbackend/cpu/fused/runtime/FusedBroadcastCursor;",
                    false
            );
            mv.visitVarInsn(ASTORE, cursorSlots.get(i));
        }
    }

    public static void emitVectorCachedInputLoads(
            MethodVisitor mv,
            FusedGenerationContext context,
            int inputCount,
            List<FusedExternalInputPlan> inputAccess,
            SlotManager sm,
            int vectorWidth
    ) {
        List<Integer> inputSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS);
        List<Integer> cachedInputVectorSlots = sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS);
        boolean memorySegmentStorage = context.numericContract().inputStorageKind() == FusedStorageKind.CPU_MEMORY_SEGMENT;

        for (int i = 0; i < inputCount; i++) {
            FusedExternalInputPlan meta = inputAccess.get(i);
            if (meta.dataType() == DataType.BOOL) {
                FusedRuntimeCalls.emitLoadBoolVectorFromArrayCall(mv, context.numericContract());
            } else {
                if (meta.isLinearAccess()) {
                    mv.visitVarInsn(ALOAD, inputSlots.get(i));
                    mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
                    if (meta.storageOffset() != 0) {
                        mv.visitLdcInsn(meta.storageOffset());
                        mv.visitInsn(IADD);
                    }
                    if (memorySegmentStorage) {
                        FusedRuntimeCalls.emitDirectLinearSegmentVectorLoad(mv, context.numericContract(), vectorWidth);
                    } else {
                        FusedRuntimeCalls.emitDirectLinearVectorLoad(mv, meta.dataType(), context.numericContract(), vectorWidth);
                    }
                } else {
                    if (memorySegmentStorage) {
                        if (!isZeroStrideBroadcast(meta)) {
                            throw new UnsupportedOperationException(
                                    "Non-linear MemorySegment vector loads are not supported for fused execution."
                            );
                        }
                        mv.visitVarInsn(ALOAD, inputSlots.get(i));
                        mv.visitLdcInsn(meta.storageOffset());
                        FusedRuntimeCalls.emitBroadcastSegmentVectorLoad(
                                mv,
                                meta.dataType(),
                                context.numericContract(),
                                vectorWidth
                        );
                    } else {
                        mv.visitVarInsn(ALOAD, inputSlots.get(i));
                        mv.visitLdcInsn(meta.storageOffset());
                        FusedRuntimeCalls.emitBroadcastArrayVectorLoad(
                                mv,
                                meta.dataType(),
                                context.numericContract(),
                                vectorWidth
                        );
                    }
                }
            }
            mv.visitVarInsn(ASTORE, cachedInputVectorSlots.get(i));
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
