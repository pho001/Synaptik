package graph.codegen;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import utils.SlotKey;
import utils.SlotManager;

import java.util.List;

import static graph.codegen.FusedMethodDescriptors.RANGE_METHOD_DESC;
import static org.objectweb.asm.Opcodes.*;

public final class FusedScalarMethodEmitter {
    private FusedScalarMethodEmitter() {}

    public static void emit(ClassWriter cw, FusedGenerationContext context) {
        MethodVisitor mv = cw.visitMethod(
                ACC_PUBLIC,
                "applyRangeScalar",
                RANGE_METHOD_DESC,
                null,
                null
        );
        mv.visitCode();

        FusedExpressionPlan plan = context.plan();
        SlotManager sm = FusedAsmSupport.buildRangeSlotLayout(plan.inputCount(), plan.nodeCount());
        int[] nodeValueSlots = sm.getGroup(SlotKey.FUSED_NODE_VALUES).stream().mapToInt(Integer::intValue).toArray();
        int[] nodeBoolSlots = sm.getGroup(SlotKey.FUSED_NODE_BOOL_VALUES).stream().mapToInt(Integer::intValue).toArray();

        FusedInputBindingEmitter.emitScalarBindings(mv, context, sm);
        FusedOutputBindingEmitter.emitScalarBinding(mv, context, sm);
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.LOOP_COUNTER));
        FusedInputBindingEmitter.emitScalarCursorBindings(mv, plan.inputs(), sm);

        Label loopStart = new Label();
        Label loopEnd = new Label();
        mv.visitLabel(loopStart);
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitJumpInsn(IF_ICMPGE, loopEnd);

        for (FusedNodePlan node : plan.nodes()) {
            FusedScalarExpressionEmitter.emitNodeEvaluationBytecode(
                    mv, plan, node, nodeValueSlots, nodeBoolSlots, sm, context.precisionMode(), plan.inputs()
            );
            if (node.outputType() == tensor.DataType.BOOL) {
                FusedAsmSupport.emitBoolScalarStoreInsn(mv, nodeBoolSlots[node.index()]);
            } else {
                FusedAsmSupport.emitScalarStoreInsn(mv, nodeValueSlots[node.index()], context.precisionMode());
            }
        }
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        FusedNodePlan outputNode = plan.outputNode();
        if (outputNode.outputType() == tensor.DataType.BOOL) {
            FusedAsmSupport.emitBoolScalarLoadInsn(
                    mv,
                    nodeBoolSlots[outputNode.index()]
            );
            mv.visitInsn(BASTORE);
        } else {
            FusedAsmSupport.emitScalarLoadInsn(
                    mv,
                    nodeValueSlots[outputNode.index()],
                    context.precisionMode()
            );
            FusedAsmSupport.emitScalarArrayStoreInsn(mv, context.precisionMode());
        }

        List<Integer> cursorSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS);
        for (int i = 0; i < plan.inputCount(); i++) {
            FusedExternalInputPlan meta = plan.inputs().get(i);
            if (!meta.usesCursor()) {
                continue;
            }
            mv.visitVarInsn(ALOAD, cursorSlots.get(i));
            mv.visitMethodInsn(INVOKEVIRTUAL, "graph/codegen/FusedBroadcastCursor", "step", "()V", false);
        }

        mv.visitIincInsn(sm.get(SlotKey.LOOP_COUNTER), 1);
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);
        mv.visitInsn(RETURN);

        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }
}
