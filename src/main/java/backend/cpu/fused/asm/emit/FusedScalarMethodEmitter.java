package backend.cpu.fused.asm.emit;

import backend.cpu.fused.asm.FusedGenerationContext;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.numeric.FusedStorageKind;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import utils.SlotKey;
import utils.SlotManager;

import java.util.List;

import static backend.cpu.fused.asm.emit.FusedMethodDescriptors.RANGE_METHOD_DESC;
import static org.objectweb.asm.Opcodes.*;

/**
 * Internal ASM emitter for the scalar execution method of generated fused kernels.
 */
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
        SlotManager sm = FusedSlotLayouts.buildRangeSlotLayout(plan.inputCount(), plan.nodeCount());
        int[] nodeValueSlots = sm.getGroup(SlotKey.FUSED_NODE_VALUES).stream().mapToInt(Integer::intValue).toArray();
        int[] nodeBoolSlots = sm.getGroup(SlotKey.FUSED_NODE_BOOL_VALUES).stream().mapToInt(Integer::intValue).toArray();
        boolean inputMemorySegmentStorage = context.numericContract().inputStorageKind() == FusedStorageKind.CPU_MEMORY_SEGMENT;
        boolean outputMemorySegmentStorage = context.numericContract().outputStorageKind() == FusedStorageKind.CPU_MEMORY_SEGMENT;

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
                    mv,
                    plan,
                    node,
                    nodeValueSlots,
                    nodeBoolSlots,
                    sm,
                    context.precisionMode(),
                    context.approximationContract(),
                    plan.inputs(),
                    inputMemorySegmentStorage
            );
            if (node.outputType() == tensor.DataType.BOOL) {
                FusedScalarBytecode.emitBoolScalarStoreInsn(mv, nodeBoolSlots[node.index()]);
            } else {
                FusedScalarBytecode.emitScalarStoreInsn(mv, nodeValueSlots[node.index()], context.precisionMode());
            }
        }
        FusedNodePlan outputNode = plan.outputNode();
        if (outputMemorySegmentStorage) {
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
            if (outputNode.outputType() == tensor.DataType.BOOL) {
                FusedScalarBytecode.emitBoolScalarLoadInsn(
                        mv,
                        nodeBoolSlots[outputNode.index()]
                );
            } else {
                FusedScalarBytecode.emitScalarLoadInsn(
                        mv,
                        nodeValueSlots[outputNode.index()],
                        context.precisionMode()
                );
            }
            FusedRuntimeCalls.emitStoreScalarToSegmentCall(mv, outputNode.outputType(), context.precisionMode());
        } else {
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
            if (outputNode.outputType() == tensor.DataType.BOOL) {
                FusedScalarBytecode.emitBoolScalarLoadInsn(
                        mv,
                        nodeBoolSlots[outputNode.index()]
                );
                mv.visitInsn(BASTORE);
            } else {
                FusedScalarBytecode.emitScalarLoadInsn(
                        mv,
                        nodeValueSlots[outputNode.index()],
                        context.precisionMode()
                );
                FusedScalarBytecode.emitScalarArrayStoreInsn(mv, context.precisionMode());
            }
        }

        List<Integer> cursorSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS);
        for (int i = 0; i < plan.inputCount(); i++) {
            FusedExternalInputPlan meta = plan.inputs().get(i);
            if (!meta.usesCursor()) {
                continue;
            }
            mv.visitVarInsn(ALOAD, cursorSlots.get(i));
            mv.visitMethodInsn(INVOKEVIRTUAL, "backend/cpu/fused/runtime/FusedBroadcastCursor", "step", "()V", false);
        }

        mv.visitIincInsn(sm.get(SlotKey.LOOP_COUNTER), 1);
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);
        mv.visitInsn(RETURN);

        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }
}
