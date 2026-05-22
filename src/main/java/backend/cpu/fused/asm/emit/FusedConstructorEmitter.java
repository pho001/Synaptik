package backend.cpu.fused.asm.emit;

import backend.cpu.fused.asm.FusedGenerationContext;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;

import static org.objectweb.asm.Opcodes.*;

/**
 * Internal ASM emitter for generated fused executable constructors.
 */
public final class FusedConstructorEmitter {
    private FusedConstructorEmitter() {}

    public static void emit(ClassWriter cw, FusedGenerationContext context) {
        MethodVisitor mv = cw.visitMethod(ACC_PUBLIC, "<init>", "()V", null, null);
        mv.visitCode();
        mv.visitVarInsn(ALOAD, 0);
        mv.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }
}
