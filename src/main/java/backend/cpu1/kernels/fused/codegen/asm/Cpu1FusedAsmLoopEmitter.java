package backend.cpu1.kernels.fused.codegen.asm;

import backend.cpu1.fused.ir.Cpu1FusedInputPlan;
import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenLoopKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenPlan;
import backend.cpu1.kernels.fused.codegen.support.Cpu1FusedGeneratedSupport;
import backend.cpu1.storage.Cpu1StorageKind;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Type;
import tensor.DataType;

import java.util.Arrays;

import static org.objectweb.asm.Opcodes.ALOAD;
import static org.objectweb.asm.Opcodes.ASTORE;
import static org.objectweb.asm.Opcodes.DASTORE;
import static org.objectweb.asm.Opcodes.DLOAD;
import static org.objectweb.asm.Opcodes.FASTORE;
import static org.objectweb.asm.Opcodes.FLOAD;
import static org.objectweb.asm.Opcodes.GOTO;
import static org.objectweb.asm.Opcodes.GETSTATIC;
import static org.objectweb.asm.Opcodes.I2L;
import static org.objectweb.asm.Opcodes.IADD;
import static org.objectweb.asm.Opcodes.IDIV;
import static org.objectweb.asm.Opcodes.IF_ICMPGE;
import static org.objectweb.asm.Opcodes.ILOAD;
import static org.objectweb.asm.Opcodes.IMUL;
import static org.objectweb.asm.Opcodes.INVOKEINTERFACE;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;
import static org.objectweb.asm.Opcodes.INVOKEVIRTUAL;
import static org.objectweb.asm.Opcodes.IREM;
import static org.objectweb.asm.Opcodes.ISTORE;
import static org.objectweb.asm.Opcodes.ISUB;
import static org.objectweb.asm.Opcodes.LMUL;
import static org.objectweb.asm.Opcodes.RETURN;
import static org.objectweb.asm.Opcodes.SALOAD;
import static org.objectweb.asm.Opcodes.SASTORE;

/**
 * Emits the generated fused range loop.
 */
public final class Cpu1FusedAsmLoopEmitter {
    private static final String VECTOR_SPECIES = "jdk/incubator/vector/VectorSpecies";
    private static final String VECTOR_SPECIES_DESC = "Ljdk/incubator/vector/VectorSpecies;";
    private static final String FLOAT_VECTOR = "jdk/incubator/vector/FloatVector";
    private static final String DOUBLE_VECTOR = "jdk/incubator/vector/DoubleVector";
    private static final String MEMORY_SEGMENT = "java/lang/foreign/MemorySegment";
    private static final String VALUE_LAYOUT = "java/lang/foreign/ValueLayout";
    private static final String VALUE_LAYOUT_BYTE_DESC = "Ljava/lang/foreign/ValueLayout$OfByte;";
    private static final String VALUE_LAYOUT_FLOAT_DESC = "Ljava/lang/foreign/ValueLayout$OfFloat;";
    private static final String VALUE_LAYOUT_DOUBLE_DESC = "Ljava/lang/foreign/ValueLayout$OfDouble;";
    private static final String GENERATED_SUPPORT = Type.getInternalName(Cpu1FusedGeneratedSupport.class);

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
                scalarOrdinals,
                plan.useFastExpApprox(),
                plan.useFastTanhApprox()
        );

        mv.visitVarInsn(ILOAD, 2);
        mv.visitVarInsn(ISTORE, loopIndexLocal);

        if (plan.loopKind() == Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR) {
            int speciesLocal = locals.allocateObject();
            int vectorWidthLocal = locals.allocateInt();
            int vectorEndLocal = locals.allocateInt();
            int[] nodeVectorLocals = new int[plan.expressionPlan().nodeCount()];
            for (int i = 0; i < nodeVectorLocals.length; i++) {
                nodeVectorLocals[i] = locals.allocateObject();
            }
            VectorContext vectorContext = new VectorContext(
                    internalClassName,
                    plan.computeType(),
                    inputs,
                    output,
                    plan.expressionPlan().outputNode(),
                    nodeVectorLocals,
                    scalarOrdinals,
                    speciesLocal
            );
            emitVectorLoop(mv, plan, inputs, output, loopIndexLocal, speciesLocal, vectorWidthLocal,
                    vectorEndLocal, vectorContext);
        }

        emitScalarLoop(mv, plan, inputs, output, loopIndexLocal, remainderLocal, coordinateLocals, context);

        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void emitScalarLoop(
            MethodVisitor mv,
            Cpu1FusedCodegenPlan plan,
            TensorBinding[] inputs,
            TensorBinding output,
            int loopIndexLocal,
            int remainderLocal,
            int[] coordinateLocals,
            LoopContext context
    ) {
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
    }

    private static void emitVectorLoop(
            MethodVisitor mv,
            Cpu1FusedCodegenPlan plan,
            TensorBinding[] inputs,
            TensorBinding output,
            int loopIndexLocal,
            int speciesLocal,
            int vectorWidthLocal,
            int vectorEndLocal,
            VectorContext context
    ) {
        emitVectorSpeciesBinding(mv, plan.computeType(), speciesLocal);
        mv.visitVarInsn(ALOAD, speciesLocal);
        mv.visitMethodInsn(INVOKEINTERFACE, VECTOR_SPECIES, "length", "()I", true);
        mv.visitVarInsn(ISTORE, vectorWidthLocal);

        mv.visitVarInsn(ALOAD, speciesLocal);
        mv.visitVarInsn(ILOAD, 3);
        mv.visitVarInsn(ILOAD, 2);
        mv.visitInsn(ISUB);
        mv.visitMethodInsn(INVOKEINTERFACE, VECTOR_SPECIES, "loopBound", "(I)I", true);
        mv.visitVarInsn(ILOAD, 2);
        mv.visitInsn(IADD);
        mv.visitVarInsn(ISTORE, vectorEndLocal);

        Label loopStart = new Label();
        Label loopEnd = new Label();
        mv.visitLabel(loopStart);
        mv.visitVarInsn(ILOAD, loopIndexLocal);
        mv.visitVarInsn(ILOAD, vectorEndLocal);
        mv.visitJumpInsn(IF_ICMPGE, loopEnd);

        for (TensorBinding input : inputs) {
            emitOffset(mv, input, loopIndexLocal, new int[0], plan.loopKind());
        }
        emitOffset(mv, output, loopIndexLocal, new int[0], plan.loopKind());

        for (Cpu1FusedNodePlan node : plan.expressionPlan().nodes()) {
            Cpu1FusedAsmVectorExpressionEmitter.emitNode(mv, context, node);
        }

        emitStoreVectorOutput(mv, context);

        mv.visitVarInsn(ILOAD, loopIndexLocal);
        mv.visitVarInsn(ILOAD, vectorWidthLocal);
        mv.visitInsn(IADD);
        mv.visitVarInsn(ISTORE, loopIndexLocal);
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);
    }

    private static void emitVectorSpeciesBinding(MethodVisitor mv, DataType computeType, int speciesLocal) {
        if (computeType == DataType.FLOAT32) {
            mv.visitFieldInsn(GETSTATIC, FLOAT_VECTOR, "SPECIES_PREFERRED", VECTOR_SPECIES_DESC);
        } else if (computeType == DataType.FLOAT64) {
            mv.visitFieldInsn(GETSTATIC, DOUBLE_VECTOR, "SPECIES_PREFERRED", VECTOR_SPECIES_DESC);
        } else {
            throw new UnsupportedOperationException("Unsupported cpu1 fused vector dtype " + computeType);
        }
        mv.visitVarInsn(ASTORE, speciesLocal);
    }

    private static void emitStoreVectorOutput(MethodVisitor mv, VectorContext context) {
        Cpu1FusedNodePlan outputNode = context.outputNode();
        mv.visitVarInsn(ALOAD, context.nodeVectorLocal(outputNode.index()));
        mv.visitVarInsn(ALOAD, context.outputBinding().storageLocal());
        mv.visitVarInsn(ILOAD, context.outputBinding().offsetLocal());
        if (context.usesFloatCompute()) {
            mv.visitMethodInsn(INVOKEVIRTUAL, FLOAT_VECTOR, "intoArray", "([FI)V", false);
        } else {
            mv.visitMethodInsn(INVOKEVIRTUAL, DOUBLE_VECTOR, "intoArray", "([DI)V", false);
        }
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
            int storageLocal = locals.allocateObject();
            int baseLocal = locals.allocateInt();
            int offsetLocal = locals.allocateInt();
            int[] strideLocals = allocateStrideLocals(locals, rank);

            mv.visitVarInsn(ALOAD, 1);
            Cpu1FusedAsmMethodEmitter.pushInt(mv, i);
            mv.visitMethodInsn(INVOKEVIRTUAL, Cpu1FusedAsmMethodEmitter.ARGS_INTERNAL_NAME,
                    "input", "(I)L" + Cpu1FusedAsmMethodEmitter.VIEW_INTERNAL_NAME + ";", false);
            mv.visitVarInsn(ASTORE, viewLocal);
            emitStorageBinding(mv, input.dataType(), plan.storageKind(), viewLocal, storageLocal);
            emitBaseAndStrides(mv, viewLocal, baseLocal, strideLocals);

            bindings[i] = new TensorBinding(input.dataType(), plan.storageKind(), storageLocal, baseLocal,
                    strideLocals, offsetLocal);
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
        int storageLocal = locals.allocateObject();
        int baseLocal = locals.allocateInt();
        int offsetLocal = locals.allocateInt();
        int[] strideLocals = allocateStrideLocals(locals, rank);

        mv.visitVarInsn(ALOAD, 1);
        mv.visitMethodInsn(INVOKEVIRTUAL, Cpu1FusedAsmMethodEmitter.ARGS_INTERNAL_NAME,
                "output", "()L" + Cpu1FusedAsmMethodEmitter.VIEW_INTERNAL_NAME + ";", false);
        mv.visitVarInsn(ASTORE, viewLocal);
        emitStorageBinding(mv, plan.computeType(), plan.storageKind(), viewLocal, storageLocal);
        emitBaseAndStrides(mv, viewLocal, baseLocal, strideLocals);

        return new TensorBinding(plan.computeType(), plan.storageKind(), storageLocal, baseLocal, strideLocals,
                offsetLocal);
    }

    private static int[] allocateStrideLocals(LocalAllocator locals, int rank) {
        int[] strideLocals = new int[rank];
        for (int dim = 0; dim < rank; dim++) {
            strideLocals[dim] = locals.allocateInt();
        }
        return strideLocals;
    }

    private static void emitStorageBinding(
            MethodVisitor mv,
            DataType dataType,
            Cpu1StorageKind storageKind,
            int viewLocal,
            int storageLocal
    ) {
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            emitSegmentBinding(mv, viewLocal, storageLocal);
        } else {
            emitArrayBinding(mv, dataType, viewLocal, storageLocal);
        }
    }

    private static void emitArrayBinding(MethodVisitor mv, DataType dataType, int viewLocal, int storageLocal) {
        mv.visitVarInsn(ALOAD, viewLocal);
        String methodName = switch (dataType) {
            case FLOAT32 -> "float32Array";
            case FLOAT64 -> "float64Array";
            case BFLOAT16 -> "bfloat16Array";
            case BOOL -> "boolArray";
            default -> throw new UnsupportedOperationException("Unsupported cpu1 fused ASM array dtype " + dataType);
        };
        String descriptor = switch (dataType) {
            case FLOAT32 -> "()[F";
            case FLOAT64 -> "()[D";
            case BFLOAT16 -> "()[S";
            case BOOL -> "()[B";
            default -> throw new UnsupportedOperationException("Unsupported cpu1 fused ASM array dtype " + dataType);
        };
        mv.visitMethodInsn(INVOKEVIRTUAL, Cpu1FusedAsmMethodEmitter.VIEW_INTERNAL_NAME,
                methodName, descriptor, false);
        mv.visitVarInsn(ASTORE, storageLocal);
    }

    private static void emitSegmentBinding(MethodVisitor mv, int viewLocal, int storageLocal) {
        mv.visitVarInsn(ALOAD, viewLocal);
        mv.visitMethodInsn(INVOKEVIRTUAL, Cpu1FusedAsmMethodEmitter.VIEW_INTERNAL_NAME,
                "segment", "()Ljava/lang/foreign/MemorySegment;", false);
        mv.visitVarInsn(ASTORE, storageLocal);
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
        if (context.outputBinding().storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            emitStoreSegmentOutput(mv, context, outputNode);
            return;
        }
        mv.visitVarInsn(ALOAD, context.outputBinding().storageLocal());
        mv.visitVarInsn(ILOAD, context.outputBinding().offsetLocal());
        if (context.outputBinding().dataType() == DataType.BFLOAT16) {
            mv.visitVarInsn(FLOAD, context.nodeValueLocal(outputNode.index()));
            mv.visitMethodInsn(INVOKESTATIC, GENERATED_SUPPORT, "floatToBf16", "(F)S", false);
            mv.visitInsn(SASTORE);
        } else if (context.usesFloatCompute()) {
            mv.visitVarInsn(FLOAD, context.nodeValueLocal(outputNode.index()));
            mv.visitInsn(FASTORE);
        } else {
            mv.visitVarInsn(DLOAD, context.nodeValueLocal(outputNode.index()));
            mv.visitInsn(DASTORE);
        }
    }

    private static void emitStoreSegmentOutput(
            MethodVisitor mv,
            LoopContext context,
            Cpu1FusedNodePlan outputNode
    ) {
        TensorBinding output = context.outputBinding();
        mv.visitVarInsn(ALOAD, output.storageLocal());
        emitValueLayout(mv, output.dataType());
        emitByteOffset(mv, output);
        if (context.usesFloatCompute()) {
            mv.visitVarInsn(FLOAD, context.nodeValueLocal(outputNode.index()));
            mv.visitMethodInsn(INVOKEINTERFACE, MEMORY_SEGMENT, "set",
                    "(" + VALUE_LAYOUT_FLOAT_DESC + "JF)V", true);
        } else {
            mv.visitVarInsn(DLOAD, context.nodeValueLocal(outputNode.index()));
            mv.visitMethodInsn(INVOKEINTERFACE, MEMORY_SEGMENT, "set",
                    "(" + VALUE_LAYOUT_DOUBLE_DESC + "JD)V", true);
        }
    }

    static void emitLoadStorageValue(MethodVisitor mv, TensorBinding binding) {
        if (binding.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            mv.visitVarInsn(ALOAD, binding.storageLocal());
            emitValueLayout(mv, binding.dataType());
            emitByteOffset(mv, binding);
            mv.visitMethodInsn(INVOKEINTERFACE, MEMORY_SEGMENT, "get", segmentGetDescriptor(binding.dataType()), true);
            return;
        }
        mv.visitVarInsn(ALOAD, binding.storageLocal());
        mv.visitVarInsn(ILOAD, binding.offsetLocal());
        if (binding.dataType() == DataType.BOOL) {
            mv.visitInsn(org.objectweb.asm.Opcodes.BALOAD);
        } else if (binding.dataType() == DataType.FLOAT32) {
            mv.visitInsn(org.objectweb.asm.Opcodes.FALOAD);
        } else if (binding.dataType() == DataType.FLOAT64) {
            mv.visitInsn(org.objectweb.asm.Opcodes.DALOAD);
        } else if (binding.dataType() == DataType.BFLOAT16) {
            mv.visitInsn(SALOAD);
            mv.visitMethodInsn(INVOKESTATIC, GENERATED_SUPPORT, "bf16ToFloat", "(S)F", false);
        } else {
            throw new UnsupportedOperationException("Unsupported fused input dtype " + binding.dataType());
        }
    }

    private static void emitValueLayout(MethodVisitor mv, DataType dataType) {
        switch (dataType) {
            case BOOL -> mv.visitFieldInsn(GETSTATIC, VALUE_LAYOUT, "JAVA_BYTE", VALUE_LAYOUT_BYTE_DESC);
            case FLOAT32 -> mv.visitFieldInsn(GETSTATIC, VALUE_LAYOUT, "JAVA_FLOAT", VALUE_LAYOUT_FLOAT_DESC);
            case FLOAT64 -> mv.visitFieldInsn(GETSTATIC, VALUE_LAYOUT, "JAVA_DOUBLE", VALUE_LAYOUT_DOUBLE_DESC);
            default -> throw new UnsupportedOperationException("Unsupported cpu1 fused segment dtype " + dataType);
        }
    }

    private static void emitByteOffset(MethodVisitor mv, TensorBinding binding) {
        mv.visitVarInsn(ILOAD, binding.offsetLocal());
        mv.visitInsn(I2L);
        Cpu1FusedAsmMethodEmitter.pushInt(mv, elementSizeBytes(binding.dataType()));
        mv.visitInsn(I2L);
        mv.visitInsn(LMUL);
    }

    private static String segmentGetDescriptor(DataType dataType) {
        return switch (dataType) {
            case BOOL -> "(" + VALUE_LAYOUT_BYTE_DESC + "J)B";
            case FLOAT32 -> "(" + VALUE_LAYOUT_FLOAT_DESC + "J)F";
            case FLOAT64 -> "(" + VALUE_LAYOUT_DOUBLE_DESC + "J)D";
            default -> throw new UnsupportedOperationException("Unsupported cpu1 fused segment dtype " + dataType);
        };
    }

    private static int elementSizeBytes(DataType dataType) {
        return switch (dataType) {
            case BOOL -> Byte.BYTES;
            case FLOAT32 -> Float.BYTES;
            case FLOAT64 -> Double.BYTES;
            default -> throw new UnsupportedOperationException("Unsupported cpu1 fused segment dtype " + dataType);
        };
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
            Cpu1StorageKind storageKind,
            int storageLocal,
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
        private final boolean useFastExpApprox;
        private final boolean useFastTanhApprox;

        private LoopContext(
                String internalClassName,
                DataType computeType,
                TensorBinding[] inputBindings,
                TensorBinding outputBinding,
                Cpu1FusedNodePlan outputNode,
                int[] nodeValueLocals,
                int tempScalarLocal,
                int[] scalarOrdinals,
                boolean useFastExpApprox,
                boolean useFastTanhApprox
        ) {
            this.internalClassName = internalClassName;
            this.inputBindings = inputBindings.clone();
            this.outputBinding = outputBinding;
            this.outputNode = outputNode;
            this.nodeValueLocals = nodeValueLocals.clone();
            this.tempScalarLocal = tempScalarLocal;
            this.scalarOrdinals = scalarOrdinals.clone();
            this.useFastExpApprox = useFastExpApprox;
            this.useFastTanhApprox = useFastTanhApprox;
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
            return computeType == DataType.FLOAT32 || computeType == DataType.BFLOAT16;
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

        boolean useFastExpApprox() {
            return useFastExpApprox;
        }

        boolean useFastTanhApprox() {
            return useFastTanhApprox;
        }

        Cpu1FusedNodePlan outputNode() {
            return outputNode;
        }
    }

    public static final class VectorContext {
        private final String internalClassName;
        private final DataType computeType;
        private final TensorBinding[] inputBindings;
        private final TensorBinding outputBinding;
        private final Cpu1FusedNodePlan outputNode;
        private final int[] nodeVectorLocals;
        private final int[] scalarOrdinals;
        private final int speciesLocal;

        private VectorContext(
                String internalClassName,
                DataType computeType,
                TensorBinding[] inputBindings,
                TensorBinding outputBinding,
                Cpu1FusedNodePlan outputNode,
                int[] nodeVectorLocals,
                int[] scalarOrdinals,
                int speciesLocal
        ) {
            this.internalClassName = internalClassName;
            this.computeType = computeType;
            this.inputBindings = inputBindings.clone();
            this.outputBinding = outputBinding;
            this.outputNode = outputNode;
            this.nodeVectorLocals = nodeVectorLocals.clone();
            this.scalarOrdinals = scalarOrdinals.clone();
            this.speciesLocal = speciesLocal;
        }

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

        Cpu1FusedNodePlan outputNode() {
            return outputNode;
        }

        int nodeVectorLocal(int nodeIndex) {
            return nodeVectorLocals[nodeIndex];
        }

        int scalarOrdinal(int nodeIndex) {
            return scalarOrdinals[nodeIndex];
        }

        int speciesLocal() {
            return speciesLocal;
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
