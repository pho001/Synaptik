package graph.codegen;

import org.objectweb.asm.MethodVisitor;
import tensor.Tensor;
import utils.SlotKey;
import utils.SlotManager;

import java.util.List;

import static org.objectweb.asm.Opcodes.*;

public final class FusedInputBindingEmitter {
    private FusedInputBindingEmitter() {}

    public static void emitScalarBindings(
            MethodVisitor mv,
            FusedGenerationContext context,
            SlotManager sm
    ) {
        List<Integer> inputSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS);
        for (int i = 0; i < context.plan().inputCount(); i++) {
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
            mv.visitLdcInsn(i);
            mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
            mv.visitTypeInsn(CHECKCAST, "tensor/Tensor");
            FusedAsmSupport.emitGetRawArrayFromTensorCall(mv, context.precisionMode());
            mv.visitVarInsn(ASTORE, inputSlots.get(i));
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
            if (meta.directIndex()) {
                continue;
            }
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
            FusedAsmSupport.emitIntArrayConstant(mv, meta.outShape());
            FusedAsmSupport.emitIntArrayConstant(mv, meta.outStrides());
            FusedAsmSupport.emitIntArrayConstant(mv, meta.effStrides());
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "graph/codegen/FusedBroadcastCursor",
                    "atStart",
                    "(I[I[I[I)Lgraph/codegen/FusedBroadcastCursor;",
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
        for (int i = 0; i < context.plan().inputCount(); i++) {
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
            mv.visitLdcInsn(i);
            mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
            mv.visitTypeInsn(CHECKCAST, "tensor/Tensor");
            FusedAsmSupport.emitGetRawArrayFromTensorCall(mv, context.precisionMode());
            mv.visitVarInsn(ASTORE, inputSlots.get(i));
        }
    }

    public static void emitVectorCursorBindings(
            MethodVisitor mv,
            List<FusedExternalInputPlan> inputAccess,
            SlotManager sm
    ) {
        List<Integer> cursorSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS);
        for (int i = 0; i < inputAccess.size(); i++) {
            FusedExternalInputPlan meta = inputAccess.get(i);
            if (meta.directIndex()) {
                continue;
            }
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
            FusedAsmSupport.emitIntArrayConstant(mv, meta.outShape());
            FusedAsmSupport.emitIntArrayConstant(mv, meta.outStrides());
            FusedAsmSupport.emitIntArrayConstant(mv, meta.effStrides());
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "graph/codegen/FusedBroadcastCursor",
                    "atStart",
                    "(I[I[I[I)Lgraph/codegen/FusedBroadcastCursor;",
                    false
            );
            mv.visitVarInsn(ASTORE, cursorSlots.get(i));
        }
    }

    public static void emitVectorCachedInputLoads(
            MethodVisitor mv,
            int inputCount,
            List<FusedExternalInputPlan> inputAccess,
            SlotManager sm,
            int precisionMode
    ) {
        List<Integer> inputSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS);
        List<Integer> cachedInputVectorSlots = sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS);
        List<Integer> cursorSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS);

        for (int i = 0; i < inputCount; i++) {
            FusedExternalInputPlan meta = inputAccess.get(i);
            if (meta.directIndex()) {
                mv.visitVarInsn(ALOAD, inputSlots.get(i));
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
                FusedAsmSupport.emitLoadVectorFromArrayCall(mv, precisionMode);
            } else {
                mv.visitVarInsn(ALOAD, cursorSlots.get(i));
                mv.visitVarInsn(ALOAD, inputSlots.get(i));
                FusedAsmSupport.emitLoadVectorFromCursorCall(mv, precisionMode);
            }
            mv.visitVarInsn(ASTORE, cachedInputVectorSlots.get(i));
        }
    }
}
