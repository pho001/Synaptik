package backend.cpu1.kernels.fused.codegen.asm;

import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import operations.Operation;
import org.objectweb.asm.MethodVisitor;
import tensor.DataType;

import static org.objectweb.asm.Opcodes.ALOAD;
import static org.objectweb.asm.Opcodes.ASTORE;
import static org.objectweb.asm.Opcodes.DCONST_0;
import static org.objectweb.asm.Opcodes.DCONST_1;
import static org.objectweb.asm.Opcodes.FCONST_0;
import static org.objectweb.asm.Opcodes.FCONST_1;
import static org.objectweb.asm.Opcodes.GETFIELD;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;
import static org.objectweb.asm.Opcodes.INVOKEVIRTUAL;

/**
 * Emits straight-line Vector API expression bytecode for one contiguous fused vector iteration.
 */
public final class Cpu1FusedAsmVectorExpressionEmitter {
    private static final String VECTOR_DESC = "Ljdk/incubator/vector/Vector;";
    private static final String VECTOR_SPECIES_DESC = "Ljdk/incubator/vector/VectorSpecies;";
    private static final String FLOAT_VECTOR = "jdk/incubator/vector/FloatVector";
    private static final String DOUBLE_VECTOR = "jdk/incubator/vector/DoubleVector";

    private Cpu1FusedAsmVectorExpressionEmitter() {
    }

    public static void emitNode(
            MethodVisitor mv,
            Cpu1FusedAsmLoopEmitter.VectorContext context,
            Cpu1FusedNodePlan node
    ) {
        Operation.OpType opType = node.opType();
        switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX -> {
                emitLoadRef(mv, context, node.inputRefs().get(0));
                emitLoadRef(mv, context, node.inputRefs().get(1));
                mv.visitMethodInsn(INVOKEVIRTUAL, vectorOwner(context.computeType()), binaryMethod(opType),
                        binaryVectorDescriptor(context.computeType()), false);
            }
            case NEG, ABS, NOOP -> {
                emitLoadRef(mv, context, node.inputRefs().get(0));
                if (opType != Operation.OpType.NOOP) {
                    mv.visitMethodInsn(INVOKEVIRTUAL, vectorOwner(context.computeType()), unaryMethod(opType),
                            unaryDescriptor(context.computeType()), false);
                }
            }
            case RELU -> {
                emitLoadRef(mv, context, node.inputRefs().get(0));
                emitZero(mv, context);
                mv.visitMethodInsn(INVOKEVIRTUAL, vectorOwner(context.computeType()), "max",
                        scalarOperandDescriptor(context.computeType()), false);
            }
            case INV -> {
                emitOneVector(mv, context);
                emitLoadRef(mv, context, node.inputRefs().get(0));
                mv.visitMethodInsn(INVOKEVIRTUAL, vectorOwner(context.computeType()), "div",
                        binaryVectorDescriptor(context.computeType()), false);
            }
            case CLAMP_MIN -> {
                emitLoadRef(mv, context, node.inputRefs().get(0));
                emitLoadScalarField(mv, context, node);
                mv.visitMethodInsn(INVOKEVIRTUAL, vectorOwner(context.computeType()), "max",
                        scalarOperandDescriptor(context.computeType()), false);
            }
            case CLAMP_MAX -> {
                emitLoadRef(mv, context, node.inputRefs().get(0));
                emitLoadScalarField(mv, context, node);
                mv.visitMethodInsn(INVOKEVIRTUAL, vectorOwner(context.computeType()), "min",
                        scalarOperandDescriptor(context.computeType()), false);
            }
            case MUL_SCALAR -> {
                emitLoadRef(mv, context, node.inputRefs().get(0));
                emitLoadScalarField(mv, context, node);
                mv.visitMethodInsn(INVOKEVIRTUAL, vectorOwner(context.computeType()), "mul",
                        scalarOperandDescriptor(context.computeType()), false);
            }
            case CONST_SCALAR -> {
                mv.visitVarInsn(ALOAD, context.speciesLocal());
                emitLoadScalarField(mv, context, node);
                mv.visitMethodInsn(INVOKESTATIC, vectorOwner(context.computeType()), "broadcast",
                        broadcastDescriptor(context.computeType()), false);
            }
            default -> throw new UnsupportedOperationException("Unsupported cpu1 fused vector ASM op " + opType);
        }
        mv.visitVarInsn(ASTORE, context.nodeVectorLocal(node.index()));
    }

    public static void emitLoadRef(
            MethodVisitor mv,
            Cpu1FusedAsmLoopEmitter.VectorContext context,
            int ref
    ) {
        if (ref < context.inputCount()) {
            Cpu1FusedAsmLoopEmitter.TensorBinding binding = context.inputBinding(ref);
            mv.visitVarInsn(ALOAD, context.speciesLocal());
            mv.visitVarInsn(ALOAD, binding.storageLocal());
            mv.visitVarInsn(org.objectweb.asm.Opcodes.ILOAD, binding.offsetLocal());
            if (binding.dataType() == DataType.FLOAT32) {
                mv.visitMethodInsn(INVOKESTATIC, FLOAT_VECTOR, "fromArray",
                        "(" + VECTOR_SPECIES_DESC + "[FI)" + vectorDescriptor(DataType.FLOAT32), false);
            } else if (binding.dataType() == DataType.FLOAT64) {
                mv.visitMethodInsn(INVOKESTATIC, DOUBLE_VECTOR, "fromArray",
                        "(" + VECTOR_SPECIES_DESC + "[DI)" + vectorDescriptor(DataType.FLOAT64), false);
            } else {
                throw new UnsupportedOperationException("Unsupported fused vector input dtype " + binding.dataType());
            }
            return;
        }
        int nodeIndex = ref - context.inputCount();
        mv.visitVarInsn(ALOAD, context.nodeVectorLocal(nodeIndex));
    }

    private static void emitOneVector(MethodVisitor mv, Cpu1FusedAsmLoopEmitter.VectorContext context) {
        mv.visitVarInsn(ALOAD, context.speciesLocal());
        if (context.usesFloatCompute()) {
            mv.visitInsn(FCONST_1);
        } else {
            mv.visitInsn(DCONST_1);
        }
        mv.visitMethodInsn(INVOKESTATIC, vectorOwner(context.computeType()), "broadcast",
                broadcastDescriptor(context.computeType()), false);
    }

    private static void emitZero(MethodVisitor mv, Cpu1FusedAsmLoopEmitter.VectorContext context) {
        if (context.usesFloatCompute()) {
            mv.visitInsn(FCONST_0);
        } else {
            mv.visitInsn(DCONST_0);
        }
    }

    private static void emitLoadScalarField(
            MethodVisitor mv,
            Cpu1FusedAsmLoopEmitter.VectorContext context,
            Cpu1FusedNodePlan node
    ) {
        int ordinal = context.scalarOrdinal(node.index());
        if (ordinal < 0) {
            throw new IllegalStateException("Node does not have a scalar binding: " + node.index());
        }
        mv.visitVarInsn(ALOAD, 0);
        if (context.usesFloatCompute()) {
            mv.visitFieldInsn(GETFIELD, context.internalClassName(), "f32Scalar" + ordinal, "F");
        } else {
            mv.visitFieldInsn(GETFIELD, context.internalClassName(), "f64Scalar" + ordinal, "D");
        }
    }

    private static String unaryMethod(Operation.OpType opType) {
        return switch (opType) {
            case NEG -> "neg";
            case ABS -> "abs";
            default -> throw new IllegalArgumentException("Unsupported unary vector op " + opType);
        };
    }

    private static String binaryMethod(Operation.OpType opType) {
        return switch (opType) {
            case ADD -> "add";
            case SUB -> "sub";
            case MUL -> "mul";
            case DIV -> "div";
            case MIN -> "min";
            case MAX -> "max";
            default -> throw new IllegalArgumentException("Unsupported binary vector op " + opType);
        };
    }

    private static String unaryDescriptor(DataType computeType) {
        return "()" + vectorDescriptor(computeType);
    }

    private static String binaryVectorDescriptor(DataType computeType) {
        return "(" + VECTOR_DESC + ")" + vectorDescriptor(computeType);
    }

    private static String scalarOperandDescriptor(DataType computeType) {
        if (computeType == DataType.FLOAT32) {
            return "(F)" + vectorDescriptor(computeType);
        }
        if (computeType == DataType.FLOAT64) {
            return "(D)" + vectorDescriptor(computeType);
        }
        throw new IllegalArgumentException("Unsupported vector compute type " + computeType);
    }

    private static String broadcastDescriptor(DataType computeType) {
        if (computeType == DataType.FLOAT32) {
            return "(" + VECTOR_SPECIES_DESC + "F)" + vectorDescriptor(computeType);
        }
        if (computeType == DataType.FLOAT64) {
            return "(" + VECTOR_SPECIES_DESC + "D)" + vectorDescriptor(computeType);
        }
        throw new IllegalArgumentException("Unsupported vector compute type " + computeType);
    }

    private static String vectorDescriptor(DataType computeType) {
        return "L" + vectorOwner(computeType) + ";";
    }

    private static String vectorOwner(DataType computeType) {
        if (computeType == DataType.FLOAT32) {
            return FLOAT_VECTOR;
        }
        if (computeType == DataType.FLOAT64) {
            return DOUBLE_VECTOR;
        }
        throw new IllegalArgumentException("Unsupported vector compute type " + computeType);
    }
}
