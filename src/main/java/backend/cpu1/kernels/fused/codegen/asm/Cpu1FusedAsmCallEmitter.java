package backend.cpu1.kernels.fused.codegen.asm;

import backend.cpu1.kernels.fused.codegen.support.Cpu1FusedMathSupport;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Type;
import tensor.DataType;

import static org.objectweb.asm.Opcodes.INVOKESTATIC;

/**
 * Emits static primitive helper calls used by generated fused kernels.
 */
public final class Cpu1FusedAsmCallEmitter {
    private static final String MATH_SUPPORT = Type.getInternalName(Cpu1FusedMathSupport.class);

    private Cpu1FusedAsmCallEmitter() {
    }

    public static void emitRelu(MethodVisitor mv, DataType computeType) {
        emit(mv, unaryMethod(computeType, "relu"), unaryDescriptor(computeType));
    }

    public static void emitAbs(MethodVisitor mv, DataType computeType) {
        emit(mv, unaryMethod(computeType, "abs"), unaryDescriptor(computeType));
    }

    public static void emitMin(MethodVisitor mv, DataType computeType) {
        emit(mv, binaryMethod(computeType, "min"), binaryDescriptor(computeType));
    }

    public static void emitMax(MethodVisitor mv, DataType computeType) {
        emit(mv, binaryMethod(computeType, "max"), binaryDescriptor(computeType));
    }

    public static String reluTarget(DataType computeType) {
        return unaryTarget(computeType, "relu");
    }

    public static String absTarget(DataType computeType) {
        return unaryTarget(computeType, "abs");
    }

    public static String minTarget(DataType computeType) {
        return binaryTarget(computeType, "min");
    }

    public static String maxTarget(DataType computeType) {
        return binaryTarget(computeType, "max");
    }

    private static void emit(MethodVisitor mv, String method, String descriptor) {
        mv.visitMethodInsn(INVOKESTATIC, MATH_SUPPORT, method, descriptor, false);
    }

    private static String unaryTarget(DataType computeType, String namePrefix) {
        return target(unaryMethod(computeType, namePrefix), unaryDescriptor(computeType));
    }

    private static String binaryTarget(DataType computeType, String namePrefix) {
        return target(binaryMethod(computeType, namePrefix), binaryDescriptor(computeType));
    }

    private static String unaryMethod(DataType computeType, String namePrefix) {
        if (computeType == DataType.FLOAT32) {
            return namePrefix + "F32";
        }
        if (computeType == DataType.FLOAT64) {
            return namePrefix + "F64";
        }
        throw new IllegalArgumentException("Unsupported helper compute type " + computeType);
    }

    private static String binaryMethod(DataType computeType, String namePrefix) {
        return unaryMethod(computeType, namePrefix);
    }

    private static String unaryDescriptor(DataType computeType) {
        if (computeType == DataType.FLOAT32) {
            return "(F)F";
        }
        if (computeType == DataType.FLOAT64) {
            return "(D)D";
        }
        throw new IllegalArgumentException("Unsupported helper compute type " + computeType);
    }

    private static String binaryDescriptor(DataType computeType) {
        if (computeType == DataType.FLOAT32) {
            return "(FF)F";
        }
        if (computeType == DataType.FLOAT64) {
            return "(DD)D";
        }
        throw new IllegalArgumentException("Unsupported helper compute type " + computeType);
    }

    private static String target(String method, String descriptor) {
        return MATH_SUPPORT + "." + method + descriptor;
    }
}
