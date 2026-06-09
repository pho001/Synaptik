package backend.cpu1.kernels.fused.codegen.asm;

import backend.cpu1.fused.ir.Cpu1FusedInputPlan;
import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenLoopKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenPlan;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import tensor.DataType;

import java.util.Arrays;

import static org.objectweb.asm.Opcodes.ALOAD;
import static org.objectweb.asm.Opcodes.ASTORE;
import static org.objectweb.asm.Opcodes.DASTORE;
import static org.objectweb.asm.Opcodes.DLOAD;
import static org.objectweb.asm.Opcodes.FASTORE;
import static org.objectweb.asm.Opcodes.FLOAD;
import static org.objectweb.asm.Opcodes.GOTO;
import static org.objectweb.asm.Opcodes.IADD;
import static org.objectweb.asm.Opcodes.IDIV;
import static org.objectweb.asm.Opcodes.IF_ICMPGE;
import static org.objectweb.asm.Opcodes.ILOAD;
import static org.objectweb.asm.Opcodes.IMUL;
import static org.objectweb.asm.Opcodes.INVOKEVIRTUAL;
import static org.objectweb.asm.Opcodes.IREM;
import static org.objectweb.asm.Opcodes.ISTORE;
import static org.objectweb.asm.Opcodes.RETURN;

/**
 * Emits the generated fused range loop.
 */
public final class Cpu1FusedAsmLoopEmitter {
    private Cpu1FusedAsmLoopEmitter() {
    }

    public static void emit(ClassWriter cw, String internalClassName, Cpu1FusedCodegenPlan plan) {
        MethodVisitor mv = cw.visitMethod(
                org.objectweb.asm.Opcodes.ACC_PUBLIC,
                "computeRange",
                Cpu1FusedAsmMethodEmitter.COMPUTE_RANGE_DESC,
                null,
                null
        );
        mv.visitCode();

        LocalAllocator locals = new LocalAllocator(4);
        int rank = outputRank(plan);
        TensorBinding[] inputs = bindInputs(mv, locals, plan, rank);
        TensorBinding output = bindOutput(mv, locals, plan, rank);
        int loopIndexLocal = locals.allocateInt();
        int remainderLocal = locals.allocateInt();
        int[] coordinateLocals = new int[rank];
        for (int dim = 0; dim < rank; dim++) {
            coordinateLocals[dim] = locals.allocateInt();
        }
        int[] nodeValueLocals = new int[plan.expressionPlan().nodeCount()];
        int valueSize = plan.computeType() == DataType.FLOAT64 ? 2 : 1;
        for (int i = 0; i < nodeValueLocals.length; i++) {
            nodeValueLocals[i] = locals.allocate(valueSize);
        }
        int tempScalarLocal = locals.allocate(valueSize);
        int[] scalarOrdinals = scalarOrdinals(plan);
        LoopContext context = new LoopContext(
                internalClassName,
                plan.computeType(),
                inputs,
                output,
                plan.expressionPlan().outputNode(),
                nodeValueLocals,
                tempScalarLocal,
                scalarOrdinals
        );

        mv.visitVarInsn(ILOAD, 2);
        mv.visitVarInsn(ISTORE, loopIndexLocal);

        Label loopStart = new Label();
        Label loopEnd = new Label();
        mv.visitLabel(loopStart);
        mv.visitVarInsn(ILOAD, loopIndexLocal);
        mv.visitVarInsn(ILOAD, 3);
        mv.visitJumpInsn(IF_ICMPGE, loopEnd);

        if (plan.loopKind() == Cpu1FusedCodegenLoopKind.STRIDED_SCALAR) {
            emitCoordinates(mv, loopIndexLocal, remainderLocal, coordinateLocals, outputShape(plan));
        }
        for (TensorBinding input : inputs) {
            emitOffset(mv, input, loopIndexLocal, coordinateLocals, plan.loopKind());
        }
        emitOffset(mv, output, loopIndexLocal, coordinateLocals, plan.loopKind());

        for (Cpu1FusedNodePlan node : plan.expressionPlan().nodes()) {
            Cpu1FusedAsmExpressionEmitter.emitNode(mv, context, node);
        }

        emitStoreOutput(mv, context);
        mv.visitIincInsn(loopIndexLocal, 1);
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);
        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static TensorBinding[] bindInputs(
            MethodVisitor mv,
            LocalAllocator locals,
            Cpu1FusedCodegenPlan plan,
            int rank
    ) {
        TensorBinding[] bindings = new TensorBinding[plan.expressionPlan().inputCount()];
        for (int i = 0; i < bindings.length; i++) {
            Cpu1FusedInputPlan input = plan.expressionPlan().inputs().get(i);
            int viewLocal = locals.allocateObject();
            int arrayLocal = locals.allocateObject();
            int baseLocal = locals.allocateInt();
            int offsetLocal = locals.allocateInt();
            int[] strideLocals = allocateStrideLocals(locals, rank);

            mv.visitVarInsn(ALOAD, 1);
            Cpu1FusedAsmMethodEmitter.pushInt(mv, i);
            mv.visitMethodInsn(INVOKEVIRTUAL, Cpu1FusedAsmMethodEmitter.ARGS_INTERNAL_NAME,
                    "input", "(I)L" + Cpu1FusedAsmMethodEmitter.VIEW_INTERNAL_NAME + ";", false);
            mv.visitVarInsn(ASTORE, viewLocal);
            emitArrayBinding(mv, input.dataType(), viewLocal, arrayLocal);
            emitBaseAndStrides(mv, viewLocal, baseLocal, strideLocals);

            bindings[i] = new TensorBinding(input.dataType(), arrayLocal, baseLocal, strideLocals, offsetLocal);
        }
        return bindings;
    }

    private static TensorBinding bindOutput(
            MethodVisitor mv,
            LocalAllocator locals,
            Cpu1FusedCodegenPlan plan,
            int rank
    ) {
        int viewLocal = locals.allocateObject();
        int arrayLocal = locals.allocateObject();
        int baseLocal = locals.allocateInt();
        int offsetLocal = locals.allocateInt();
        int[] strideLocals = allocateStrideLocals(locals, rank);

        mv.visitVarInsn(ALOAD, 1);
        mv.visitMethodInsn(INVOKEVIRTUAL, Cpu1FusedAsmMethodEmitter.ARGS_INTERNAL_NAME,
                "output", "()L" + Cpu1FusedAsmMethodEmitter.VIEW_INTERNAL_NAME + ";", false);
        mv.visitVarInsn(ASTORE, viewLocal);
        emitArrayBinding(mv, plan.computeType(), viewLocal, arrayLocal);
        emitBaseAndStrides(mv, viewLocal, baseLocal, strideLocals);

        return new TensorBinding(plan.computeType(), arrayLocal, baseLocal, strideLocals, offsetLocal);
    }

    private static int[] allocateStrideLocals(LocalAllocator locals, int rank) {
        int[] strideLocals = new int[rank];
        for (int dim = 0; dim < rank; dim++) {
            strideLocals[dim] = locals.allocateInt();
        }
        return strideLocals;
    }

    private static void emitArrayBinding(MethodVisitor mv, DataType dataType, int viewLocal, int arrayLocal) {
        mv.visitVarInsn(ALOAD, viewLocal);
        String methodName = switch (dataType) {
            case FLOAT32 -> "float32Array";
            case FLOAT64 -> "float64Array";
            case BOOL -> "boolArray";
            default -> throw new UnsupportedOperationException("Unsupported cpu1 fused ASM array dtype " + dataType);
        };
        String descriptor = switch (dataType) {
            case FLOAT32 -> "()[F";
            case FLOAT64 -> "()[D";
            case BOOL -> "()[B";
            default -> throw new UnsupportedOperationException("Unsupported cpu1 fused ASM array dtype " + dataType);
        };
        mv.visitMethodInsn(INVOKEVIRTUAL, Cpu1FusedAsmMethodEmitter.VIEW_INTERNAL_NAME,
                methodName, descriptor, false);
        mv.visitVarInsn(ASTORE, arrayLocal);
    }

    private static void emitBaseAndStrides(MethodVisitor mv, int viewLocal, int baseLocal, int[] strideLocals) {
        mv.visitVarInsn(ALOAD, viewLocal);
        mv.visitMethodInsn(INVOKEVIRTUAL, Cpu1FusedAsmMethodEmitter.VIEW_INTERNAL_NAME,
                "storageOffset", "()I", false);
        mv.visitVarInsn(ISTORE, baseLocal);
        for (int dim = 0; dim < strideLocals.length; dim++) {
            mv.visitVarInsn(ALOAD, viewLocal);
            Cpu1FusedAsmMethodEmitter.pushInt(mv, dim);
            mv.visitMethodInsn(INVOKEVIRTUAL, Cpu1FusedAsmMethodEmitter.VIEW_INTERNAL_NAME,
                    "stride", "(I)I", false);
            mv.visitVarInsn(ISTORE, strideLocals[dim]);
        }
    }

    private static void emitCoordinates(
            MethodVisitor mv,
            int loopIndexLocal,
            int remainderLocal,
            int[] coordinateLocals,
            int[] shape
    ) {
        mv.visitVarInsn(ILOAD, loopIndexLocal);
        mv.visitVarInsn(ISTORE, remainderLocal);
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            mv.visitVarInsn(ILOAD, remainderLocal);
            Cpu1FusedAsmMethodEmitter.pushInt(mv, shape[dim]);
            mv.visitInsn(IREM);
            mv.visitVarInsn(ISTORE, coordinateLocals[dim]);

            mv.visitVarInsn(ILOAD, remainderLocal);
            Cpu1FusedAsmMethodEmitter.pushInt(mv, shape[dim]);
            mv.visitInsn(IDIV);
            mv.visitVarInsn(ISTORE, remainderLocal);
        }
    }

    private static void emitOffset(
            MethodVisitor mv,
            TensorBinding binding,
            int loopIndexLocal,
            int[] coordinateLocals,
            Cpu1FusedCodegenLoopKind loopKind
    ) {
        mv.visitVarInsn(ILOAD, binding.baseLocal());
        if (loopKind == Cpu1FusedCodegenLoopKind.STRIDED_SCALAR) {
            for (int dim = 0; dim < coordinateLocals.length; dim++) {
                mv.visitVarInsn(ILOAD, coordinateLocals[dim]);
                mv.visitVarInsn(ILOAD, binding.strideLocals()[dim]);
                mv.visitInsn(IMUL);
                mv.visitInsn(IADD);
            }
        } else {
            mv.visitVarInsn(ILOAD, loopIndexLocal);
            mv.visitInsn(IADD);
        }
        mv.visitVarInsn(ISTORE, binding.offsetLocal());
    }

    private static void emitStoreOutput(MethodVisitor mv, LoopContext context) {
        Cpu1FusedNodePlan outputNode = context.outputNode();
        mv.visitVarInsn(ALOAD, context.outputBinding().arrayLocal());
        mv.visitVarInsn(ILOAD, context.outputBinding().offsetLocal());
        if (context.usesFloatCompute()) {
            mv.visitVarInsn(FLOAD, context.nodeValueLocal(outputNode.index()));
            mv.visitInsn(FASTORE);
        } else {
            mv.visitVarInsn(DLOAD, context.nodeValueLocal(outputNode.index()));
            mv.visitInsn(DASTORE);
        }
    }

    private static int outputRank(Cpu1FusedCodegenPlan plan) {
        return outputShape(plan).length;
    }

    private static int[] outputShape(Cpu1FusedCodegenPlan plan) {
        if (plan.expressionPlan().inputs().isEmpty()) {
            return new int[0];
        }
        return plan.expressionPlan().inputs().getFirst().logicalOutputShape();
    }

    private static int[] scalarOrdinals(Cpu1FusedCodegenPlan plan) {
        int[] ordinals = new int[plan.expressionPlan().nodeCount()];
        Arrays.fill(ordinals, -1);
        int next = 0;
        for (Cpu1FusedNodePlan node : plan.expressionPlan().nodes()) {
            if (node.scalarParameter().present()) {
                ordinals[node.index()] = next++;
            }
        }
        return ordinals;
    }

    public record TensorBinding(
            DataType dataType,
            int arrayLocal,
            int baseLocal,
            int[] strideLocals,
            int offsetLocal
    ) {
        public TensorBinding {
            strideLocals = strideLocals == null ? new int[0] : strideLocals.clone();
        }

        @Override
        public int[] strideLocals() {
            return strideLocals.clone();
        }
    }

    public static final class LoopContext {
        private final String internalClassName;
        private final TensorBinding[] inputBindings;
        private final TensorBinding outputBinding;
        private final Cpu1FusedNodePlan outputNode;
        private final int[] nodeValueLocals;
        private final int tempScalarLocal;
        private final int[] scalarOrdinals;

        private LoopContext(
                String internalClassName,
                DataType computeType,
                TensorBinding[] inputBindings,
                TensorBinding outputBinding,
                Cpu1FusedNodePlan outputNode,
                int[] nodeValueLocals,
                int tempScalarLocal,
                int[] scalarOrdinals
        ) {
            this.internalClassName = internalClassName;
            this.inputBindings = inputBindings.clone();
            this.outputBinding = outputBinding;
            this.outputNode = outputNode;
            this.nodeValueLocals = nodeValueLocals.clone();
            this.tempScalarLocal = tempScalarLocal;
            this.scalarOrdinals = scalarOrdinals.clone();
            this.computeType = computeType;
        }

        private final DataType computeType;

        String internalClassName() {
            return internalClassName;
        }

        DataType computeType() {
            return computeType;
        }

        boolean usesFloatCompute() {
            return computeType == DataType.FLOAT32;
        }

        int inputCount() {
            return inputBindings.length;
        }

        TensorBinding inputBinding(int index) {
            return inputBindings[index];
        }

        TensorBinding outputBinding() {
            return outputBinding;
        }

        int nodeValueLocal(int nodeIndex) {
            return nodeValueLocals[nodeIndex];
        }

        int tempScalarLocal() {
            return tempScalarLocal;
        }

        int scalarOrdinal(int nodeIndex) {
            return scalarOrdinals[nodeIndex];
        }

        Cpu1FusedNodePlan outputNode() {
            return outputNode;
        }
    }

    private static final class LocalAllocator {
        private int next;

        private LocalAllocator(int firstFreeLocal) {
            this.next = firstFreeLocal;
        }

        int allocateObject() {
            return allocate(1);
        }

        int allocateInt() {
            return allocate(1);
        }

        int allocate(int size) {
            int result = next;
            next += size;
            return result;
        }
    }
}
