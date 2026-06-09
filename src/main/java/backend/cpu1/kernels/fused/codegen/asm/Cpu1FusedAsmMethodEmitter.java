package backend.cpu1.kernels.fused.codegen.asm;

import backend.cpu1.exec.Cpu1FusedKernelArgs;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.fused.Cpu1FusedElementwiseRangeRunner;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Type;

import static org.objectweb.asm.Opcodes.BIPUSH;
import static org.objectweb.asm.Opcodes.ICONST_0;
import static org.objectweb.asm.Opcodes.ICONST_1;
import static org.objectweb.asm.Opcodes.ICONST_2;
import static org.objectweb.asm.Opcodes.ICONST_3;
import static org.objectweb.asm.Opcodes.ICONST_4;
import static org.objectweb.asm.Opcodes.ICONST_5;
import static org.objectweb.asm.Opcodes.ICONST_M1;
import static org.objectweb.asm.Opcodes.SIPUSH;

/**
 * Shared JVM names and small bytecode helpers for cpu1 fused ASM emitters.
 */
public final class Cpu1FusedAsmMethodEmitter {
    public static final String RUNNER_INTERNAL_NAME = Type.getInternalName(Cpu1FusedElementwiseRangeRunner.class);
    public static final String ARGS_INTERNAL_NAME = Type.getInternalName(Cpu1FusedKernelArgs.class);
    public static final String VIEW_INTERNAL_NAME = Type.getInternalName(Cpu1TensorView.class);
    public static final String COMPUTE_RANGE_DESC = Type.getMethodDescriptor(
            Type.VOID_TYPE,
            Type.getType(Cpu1FusedKernelArgs.class),
            Type.INT_TYPE,
            Type.INT_TYPE
    );
    public static final String CTOR_SCALARS_DESC = "([F[D)V";

    private Cpu1FusedAsmMethodEmitter() {
    }

    public static void pushInt(MethodVisitor mv, int value) {
        switch (value) {
            case -1 -> mv.visitInsn(ICONST_M1);
            case 0 -> mv.visitInsn(ICONST_0);
            case 1 -> mv.visitInsn(ICONST_1);
            case 2 -> mv.visitInsn(ICONST_2);
            case 3 -> mv.visitInsn(ICONST_3);
            case 4 -> mv.visitInsn(ICONST_4);
            case 5 -> mv.visitInsn(ICONST_5);
            default -> {
                if (value >= Byte.MIN_VALUE && value <= Byte.MAX_VALUE) {
                    mv.visitIntInsn(BIPUSH, value);
                } else if (value >= Short.MIN_VALUE && value <= Short.MAX_VALUE) {
                    mv.visitIntInsn(SIPUSH, value);
                } else {
                    mv.visitLdcInsn(value);
                }
            }
        }
    }
}
