package backend.cpu1.kernels.fused.codegen.asm;

import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenPlan;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;

import static org.objectweb.asm.Opcodes.ACC_FINAL;
import static org.objectweb.asm.Opcodes.ACC_PRIVATE;
import static org.objectweb.asm.Opcodes.ACC_PUBLIC;
import static org.objectweb.asm.Opcodes.ALOAD;
import static org.objectweb.asm.Opcodes.DALOAD;
import static org.objectweb.asm.Opcodes.FALOAD;
import static org.objectweb.asm.Opcodes.INVOKESPECIAL;
import static org.objectweb.asm.Opcodes.PUTFIELD;
import static org.objectweb.asm.Opcodes.RETURN;
import static org.objectweb.asm.Opcodes.V17;

/**
 * Emits a concrete generated class for one cpu1 fused structural signature.
 */
public final class Cpu1FusedAsmClassEmitter {
    private Cpu1FusedAsmClassEmitter() {
    }

    public static EmittedClass emit(String binaryClassName, Cpu1FusedCodegenPlan plan) {
        if (binaryClassName == null || binaryClassName.isBlank()) {
            throw new IllegalArgumentException("binaryClassName cannot be blank");
        }
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        String internalName = binaryClassName.replace('.', '/');
        ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
        cw.visit(
                V17,
                ACC_PUBLIC | ACC_FINAL,
                internalName,
                null,
                "java/lang/Object",
                new String[]{Cpu1FusedAsmMethodEmitter.RUNNER_INTERNAL_NAME}
        );
        emitScalarFields(cw, plan);
        emitConstructor(cw, internalName, scalarCount(plan));
        Cpu1FusedAsmLoopEmitter.emit(cw, internalName, plan);
        cw.visitEnd();
        return new EmittedClass(binaryClassName, cw.toByteArray());
    }

    private static void emitScalarFields(ClassWriter cw, Cpu1FusedCodegenPlan plan) {
        int scalarCount = scalarCount(plan);
        for (int i = 0; i < scalarCount; i++) {
            cw.visitField(ACC_PRIVATE | ACC_FINAL, "f32Scalar" + i, "F", null, null).visitEnd();
            cw.visitField(ACC_PRIVATE | ACC_FINAL, "f64Scalar" + i, "D", null, null).visitEnd();
        }
    }

    private static void emitConstructor(ClassWriter cw, String internalName, int scalarCount) {
        MethodVisitor mv = cw.visitMethod(
                ACC_PUBLIC,
                "<init>",
                Cpu1FusedAsmMethodEmitter.CTOR_SCALARS_DESC,
                null,
                null
        );
        mv.visitCode();
        mv.visitVarInsn(ALOAD, 0);
        mv.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        for (int i = 0; i < scalarCount; i++) {
            mv.visitVarInsn(ALOAD, 0);
            mv.visitVarInsn(ALOAD, 1);
            Cpu1FusedAsmMethodEmitter.pushInt(mv, i);
            mv.visitInsn(FALOAD);
            mv.visitFieldInsn(PUTFIELD, internalName, "f32Scalar" + i, "F");

            mv.visitVarInsn(ALOAD, 0);
            mv.visitVarInsn(ALOAD, 2);
            Cpu1FusedAsmMethodEmitter.pushInt(mv, i);
            mv.visitInsn(DALOAD);
            mv.visitFieldInsn(PUTFIELD, internalName, "f64Scalar" + i, "D");
        }
        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static int scalarCount(Cpu1FusedCodegenPlan plan) {
        int count = 0;
        for (Cpu1FusedNodePlan node : plan.expressionPlan().nodes()) {
            if (node.scalarParameter().present()) {
                count++;
            }
        }
        return count;
    }

    public record EmittedClass(String binaryName, byte[] bytecode) {
        public EmittedClass {
            if (binaryName == null || binaryName.isBlank()) {
                throw new IllegalArgumentException("binaryName cannot be blank");
            }
            if (bytecode == null || bytecode.length == 0) {
                throw new IllegalArgumentException("bytecode cannot be empty");
            }
        }
    }
}
