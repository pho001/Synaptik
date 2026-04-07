package graph.codegen;

import org.objectweb.asm.ClassWriter;

import static org.objectweb.asm.Opcodes.*;

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
                new String[]{"graph/fused/PreparedFusedExecutable"}
        );
        return cw;
    }
}
