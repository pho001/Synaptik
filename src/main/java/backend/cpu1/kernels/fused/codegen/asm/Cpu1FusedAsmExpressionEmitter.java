package backend.cpu1.kernels.fused.codegen.asm;

import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import operations.Operation;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;

import static org.objectweb.asm.Opcodes.ALOAD;
import static org.objectweb.asm.Opcodes.DADD;
import static org.objectweb.asm.Opcodes.DCONST_1;
import static org.objectweb.asm.Opcodes.DDIV;
import static org.objectweb.asm.Opcodes.DLOAD;
import static org.objectweb.asm.Opcodes.DMUL;
import static org.objectweb.asm.Opcodes.DNEG;
import static org.objectweb.asm.Opcodes.DSTORE;
import static org.objectweb.asm.Opcodes.DSUB;
import static org.objectweb.asm.Opcodes.FADD;
import static org.objectweb.asm.Opcodes.FCONST_1;
import static org.objectweb.asm.Opcodes.FDIV;
import static org.objectweb.asm.Opcodes.FLOAD;
import static org.objectweb.asm.Opcodes.FMUL;
import static org.objectweb.asm.Opcodes.FNEG;
import static org.objectweb.asm.Opcodes.FSTORE;
import static org.objectweb.asm.Opcodes.FSUB;
import static org.objectweb.asm.Opcodes.GETFIELD;
import static org.objectweb.asm.Opcodes.GOTO;
import static org.objectweb.asm.Opcodes.IFEQ;

/**
 * Emits straight-line scalar expression bytecode for one fused loop iteration.
 */
public final class Cpu1FusedAsmExpressionEmitter {
    private Cpu1FusedAsmExpressionEmitter() {
    }

    public static void emitNode(
            MethodVisitor mv,
            Cpu1FusedAsmLoopEmitter.LoopContext context,
            Cpu1FusedNodePlan node
    ) {
        Operation.OpType opType = node.opType();
        if (opType == Operation.OpType.WHERE) {
            emitWhere(mv, context, node);
            storeNodeValue(mv, context, node);
            return;
        }
        for (int ref : node.inputRefs()) {
            emitLoadRef(mv, context, ref);
        }
        switch (opType) {
            case ADD -> mv.visitInsn(context.usesFloatCompute() ? FADD : DADD);
            case SUB -> mv.visitInsn(context.usesFloatCompute() ? FSUB : DSUB);
            case MUL -> mv.visitInsn(context.usesFloatCompute() ? FMUL : DMUL);
            case DIV -> mv.visitInsn(context.usesFloatCompute() ? FDIV : DDIV);
            case MIN -> Cpu1FusedAsmCallEmitter.emitMin(mv, context.computeType());
            case MAX -> Cpu1FusedAsmCallEmitter.emitMax(mv, context.computeType());
            case NEG -> mv.visitInsn(context.usesFloatCompute() ? FNEG : DNEG);
            case INV -> emitInv(mv, context);
            case ABS -> Cpu1FusedAsmCallEmitter.emitAbs(mv, context.computeType());
            case RELU -> Cpu1FusedAsmCallEmitter.emitRelu(mv, context.computeType());
            case EXP, FAST_EXP, LOG, TANH, FAST_TANH, ERF, SQRT, SIGMOID, FLOOR, CEIL, SIGN ->
                    Cpu1FusedAsmCallEmitter.emitUnaryIntrinsic(mv, effectiveUnaryIntrinsic(opType, context),
                            context.computeType());
            case POW -> {
                emitLoadScalarField(mv, context, node);
                Cpu1FusedAsmCallEmitter.emitPow(mv, context.computeType());
            }
            case POW_TENSOR -> Cpu1FusedAsmCallEmitter.emitPow(mv, context.computeType());
            case CLAMP_MIN -> {
                emitLoadScalarField(mv, context, node);
                Cpu1FusedAsmCallEmitter.emitMax(mv, context.computeType());
            }
            case CLAMP_MAX -> {
                emitLoadScalarField(mv, context, node);
                Cpu1FusedAsmCallEmitter.emitMin(mv, context.computeType());
            }
            case MUL_SCALAR -> {
                emitLoadScalarField(mv, context, node);
                mv.visitInsn(context.usesFloatCompute() ? FMUL : DMUL);
            }
            case CONST_SCALAR -> emitLoadScalarField(mv, context, node);
            case NOOP -> {
            }
            default -> throw new UnsupportedOperationException("Unsupported cpu1 fused ASM op " + opType);
        }
        storeNodeValue(mv, context, node);
    }

    private static void emitWhere(
            MethodVisitor mv,
            Cpu1FusedAsmLoopEmitter.LoopContext context,
            Cpu1FusedNodePlan node
    ) {
        if (node.inputRefs().size() != 3) {
            throw new IllegalArgumentException("WHERE requires 3 input refs");
        }
        Label falseLabel = new Label();
        Label done = new Label();
        emitLoadRef(mv, context, node.inputRefs().get(0));
        mv.visitJumpInsn(IFEQ, falseLabel);
        emitLoadRef(mv, context, node.inputRefs().get(1));
        mv.visitJumpInsn(GOTO, done);
        mv.visitLabel(falseLabel);
        emitLoadRef(mv, context, node.inputRefs().get(2));
        mv.visitLabel(done);
    }

    private static Operation.OpType effectiveUnaryIntrinsic(
            Operation.OpType opType,
            Cpu1FusedAsmLoopEmitter.LoopContext context
    ) {
        return switch (opType) {
            case EXP -> context.useFastExpApprox() ? Operation.OpType.FAST_EXP : Operation.OpType.EXP;
            case TANH -> context.useFastTanhApprox() ? Operation.OpType.FAST_TANH : Operation.OpType.TANH;
            default -> opType;
        };
    }

    private static void emitInv(MethodVisitor mv, Cpu1FusedAsmLoopEmitter.LoopContext context) {
        if (context.usesFloatCompute()) {
            mv.visitVarInsn(FSTORE, context.tempScalarLocal());
            mv.visitInsn(FCONST_1);
            mv.visitVarInsn(FLOAD, context.tempScalarLocal());
            mv.visitInsn(FDIV);
        } else {
            mv.visitVarInsn(DSTORE, context.tempScalarLocal());
            mv.visitInsn(DCONST_1);
            mv.visitVarInsn(DLOAD, context.tempScalarLocal());
            mv.visitInsn(DDIV);
        }
    }

    private static void emitLoadRef(
            MethodVisitor mv,
            Cpu1FusedAsmLoopEmitter.LoopContext context,
            int ref
    ) {
        if (ref < context.inputCount()) {
            Cpu1FusedAsmLoopEmitter.TensorBinding binding = context.inputBinding(ref);
            Cpu1FusedAsmLoopEmitter.emitLoadStorageValue(mv, binding);
            return;
        }
        int nodeIndex = ref - context.inputCount();
        if (context.usesFloatCompute()) {
            mv.visitVarInsn(FLOAD, context.nodeValueLocal(nodeIndex));
        } else {
            mv.visitVarInsn(DLOAD, context.nodeValueLocal(nodeIndex));
        }
    }

    private static void storeNodeValue(
            MethodVisitor mv,
            Cpu1FusedAsmLoopEmitter.LoopContext context,
            Cpu1FusedNodePlan node
    ) {
        if (context.usesFloatCompute()) {
            mv.visitVarInsn(FSTORE, context.nodeValueLocal(node.index()));
        } else {
            mv.visitVarInsn(DSTORE, context.nodeValueLocal(node.index()));
        }
    }

    private static void emitLoadScalarField(
            MethodVisitor mv,
            Cpu1FusedAsmLoopEmitter.LoopContext context,
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
}
