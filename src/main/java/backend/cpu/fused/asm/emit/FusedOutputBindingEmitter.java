package backend.cpu.fused.asm.emit;

import backend.cpu.fused.asm.FusedGenerationContext;

import org.objectweb.asm.MethodVisitor;
import utils.SlotKey;
import utils.SlotManager;

import static org.objectweb.asm.Opcodes.*;

/**
 * Internal ASM emitter for binding fused output tensor storage.
 */
public final class FusedOutputBindingEmitter {
    private FusedOutputBindingEmitter() {}

    public static void emitScalarBinding(MethodVisitor mv, FusedGenerationContext context, SlotManager sm) {
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR));
        FusedRuntimeCalls.emitGetRawArrayFromTensorCall(mv, context.plan().outputNode().outputType());
        mv.visitVarInsn(ASTORE, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
    }

    public static void emitVectorBinding(MethodVisitor mv, FusedGenerationContext context, SlotManager sm) {
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR));
        FusedRuntimeCalls.emitGetRawArrayFromTensorCall(mv, context.plan().outputNode().outputType());
        mv.visitVarInsn(ASTORE, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
    }
}
