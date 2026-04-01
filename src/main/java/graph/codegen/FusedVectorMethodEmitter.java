package graph.codegen;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import utils.SlotKey;
import utils.SlotManager;

import static graph.codegen.FusedMethodDescriptors.RANGE_METHOD_DESC;
import static org.objectweb.asm.Opcodes.*;

public final class FusedVectorMethodEmitter {
    private FusedVectorMethodEmitter() {}

    public static void emit(ClassWriter cw, FusedGenerationContext context) {
        MethodVisitor mv = cw.visitMethod(
                ACC_PUBLIC,
                "applyRangeVector",
                RANGE_METHOD_DESC,
                null,
                null
        );
        mv.visitCode();

        FusedExpressionPlan plan = context.plan();
        SlotManager sm = FusedAsmSupport.buildVectorSlotLayout(plan.inputCount(), plan.nodeCount());
        int[] nodeVectorSlots = sm.getGroup(SlotKey.FUSED_NODE_VECTOR_VALUES).stream().mapToInt(Integer::intValue).toArray();

        FusedInputBindingEmitter.emitVectorBindings(mv, context, sm);
        FusedOutputBindingEmitter.emitVectorBinding(mv, context, sm);
        FusedInputBindingEmitter.emitVectorCursorBindings(mv, plan.inputs(), sm);

        FusedAsmSupport.emitVectorWidthCall(mv, context.precisionMode());
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.SECOND_LOOP_COUNTER));

        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
        mv.visitInsn(ISUB);
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.SECOND_LOOP_COUNTER));
        mv.visitInsn(IREM);
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.RANGE_UPPER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_UPPER));
        mv.visitInsn(ISUB);
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.RANGE_UPPER));

        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.LOOP_COUNTER));

        Label loopStart = new Label();
        Label loopEnd = new Label();
        mv.visitLabel(loopStart);
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_UPPER));
        mv.visitJumpInsn(IF_ICMPGE, loopEnd);

        FusedInputBindingEmitter.emitVectorCachedInputLoads(mv, plan.inputCount(), plan.inputs(), sm, context.precisionMode());

        for (FusedNodePlan node : plan.nodes()) {
            FusedVectorExpressionEmitter.emitNodeEvaluationBytecode(
                    mv, node, nodeVectorSlots, sm, context.precisionMode()
            );
            mv.visitVarInsn(ASTORE, nodeVectorSlots[node.index()]);
        }

        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(ALOAD, nodeVectorSlots[plan.outputRef() - plan.inputCount()]);
        FusedAsmSupport.emitStoreVectorToArrayCall(mv, context.precisionMode());

        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.SECOND_LOOP_COUNTER));
        mv.visitInsn(IADD);
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);

        Label noTail = new Label();
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_UPPER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitJumpInsn(IF_ICMPGE, noTail);
        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_UPPER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
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
}
