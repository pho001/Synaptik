package backend.cpu.fused.asm.emit;

import backend.cpu.fused.asm.FusedGenerationContext;

import backend.cpu.fused.runtime.FusedDTypeOps;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedNodePlan;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import utils.SlotKey;
import utils.SlotManager;

import static backend.cpu.fused.asm.emit.FusedMethodDescriptors.RANGE_METHOD_DESC;
import static org.objectweb.asm.Opcodes.*;

/**
 * Internal ASM emitter for the vector execution method of generated fused kernels.
 */
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
        if (!supportsVector(context, plan)) {
            emitScalarDelegate(mv, context);
            return;
        }
        SlotManager sm = FusedSlotLayouts.buildVectorSlotLayout(plan.inputCount(), plan.nodeCount());
        int[] nodeVectorSlots = sm.getGroup(SlotKey.FUSED_NODE_VECTOR_VALUES).stream().mapToInt(Integer::intValue).toArray();

        FusedInputBindingEmitter.emitVectorBindings(mv, context, sm);
        FusedOutputBindingEmitter.emitVectorBinding(mv, context, sm);
        FusedInputBindingEmitter.emitVectorCursorBindings(mv, plan.inputs(), sm);

        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
        mv.visitInsn(ISUB);
        FusedVectorBytecode.emitVectorWidthConstant(mv, context.vectorWidth());
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

        FusedInputBindingEmitter.emitVectorCachedInputLoads(
                mv,
                plan.inputCount(),
                plan.inputs(),
                sm,
                context.precisionMode(),
                context.vectorWidth()
        );

        for (FusedNodePlan node : plan.nodes()) {
            FusedVectorExpressionEmitter.emitNodeEvaluationBytecode(
                    mv, plan, node, nodeVectorSlots, sm, context.precisionMode(), context.approximationContract()
            );
            mv.visitVarInsn(ASTORE, nodeVectorSlots[node.index()]);
        }

        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(ALOAD, nodeVectorSlots[plan.outputRef() - plan.inputCount()]);
        if (plan.outputNode().outputType() == tensor.DataType.BOOL) {
            FusedVectorBytecode.emitVectorWidthConstant(mv, context.vectorWidth());
            FusedRuntimeCalls.emitStoreBoolVectorToArrayCall(mv, context.precisionMode());
        } else if (context.precisionMode() == FusedDTypeOps.MODE_F32 || context.precisionMode() == FusedDTypeOps.MODE_F64) {
            FusedRuntimeCalls.emitDirectStoreVectorToArrayCall(mv, context.precisionMode());
        } else {
            FusedRuntimeCalls.emitStoreVectorToArrayCall(mv, context.precisionMode());
        }

        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        FusedVectorBytecode.emitVectorWidthConstant(mv, context.vectorWidth());
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
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_CONTEXT));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_UPPER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
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

    private static boolean supportsVector(FusedGenerationContext context, FusedExpressionPlan plan) {
        for (FusedNodePlan node : plan.nodes()) {
            if (node.opType() == operations.Operation.OpType.POW_TENSOR) {
                return false;
            }
        }
        return true;
    }

    private static void emitScalarDelegate(MethodVisitor mv, FusedGenerationContext context) {
        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ALOAD, 1);
        mv.visitVarInsn(ALOAD, 2);
        mv.visitVarInsn(ALOAD, 3);
        mv.visitVarInsn(ILOAD, 4);
        mv.visitVarInsn(ILOAD, 5);
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
