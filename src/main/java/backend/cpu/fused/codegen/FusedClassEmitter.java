package backend.cpu.fused.codegen;

import org.objectweb.asm.ClassWriter;

import static org.objectweb.asm.Opcodes.*;

/**
 * Internal ASM emitter for the generated fused executable class shell.
 */
public final class FusedClassEmitter {
    private FusedClassEmitter() {}

    public static ClassWriter createClass(FusedGenerationContext context) {
        ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_FRAMES);
        cw.visit(
                V1_8,
                ACC_PUBLIC | ACC_FINAL,
                context.internalClassName(),
                null,
                "java/lang/Object",
                new String[]{"backend/cpu/fused/exec/PreparedFusedExecutable"}
        );
        return cw;
    }
}
