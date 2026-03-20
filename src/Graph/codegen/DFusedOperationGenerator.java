package Graph.codegen;

import Operations.Operation;
import Tensor.Tensor;
import Utils.SlotKey;
import Utils.SlotManager;
import org.objectweb.asm.*;

import java.util.*;

public class DFusedOperationGenerator implements Opcodes {

    public static byte[] generate(String internalClassName, List<Tensor> cluster, Tensor outputTensor, List<Tensor> externalInputsInOrder) {
        String className = internalClassName;
        ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_FRAMES);

        cw.visit(
                V1_8,
                ACC_PUBLIC,
                className,
                null,
                "java/lang/Object",
                new String[]{"Operations/Operation", "Operations/FusedCompiledOperation"}
        );

        cw.visitField(ACC_PRIVATE | ACC_FINAL, "expression", "Ljava/lang/String;", null, null).visitEnd();
        cw.visitField(ACC_PRIVATE | ACC_FINAL, "precisionMode", "I", null, null).visitEnd();

        MethodVisitor mv = cw.visitMethod(ACC_PUBLIC, "<init>", "(Ljava/util/List;Ljava/lang/String;I)V", null, null);
        mv.visitCode();
        mv.visitVarInsn(ALOAD, 0);
        mv.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ALOAD, 2);
        mv.visitFieldInsn(PUTFIELD, className, "expression", "Ljava/lang/String;");
        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ILOAD, 3);
        mv.visitFieldInsn(PUTFIELD, className, "precisionMode", "I");
        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();

        generateMetadataMethods(cw, className);
        generateApplyMethod(cw, className);
        generateApplyRangeScalarMethod(cw, className, cluster, outputTensor, externalInputsInOrder);
        generateApplyRangeVectorMethod(cw, className, cluster, outputTensor, externalInputsInOrder);

        cw.visitEnd();
        return cw.toByteArray();
    }

    private static void generateApplyMethod(ClassWriter cw, String className) {
        MethodVisitor mv = cw.visitMethod(ACC_PUBLIC, "apply", "(Ljava/util/List;LTensor/Tensor;)V", null, null);
        mv.visitCode();

        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ALOAD, 1);
        mv.visitVarInsn(ALOAD, 2);
        mv.visitInsn(ICONST_0);
        mv.visitVarInsn(ALOAD, 2);
        mv.visitMethodInsn(INVOKEVIRTUAL, "Tensor/Tensor", "getData", "()[D", false);
        mv.visitInsn(ARRAYLENGTH);
        mv.visitMethodInsn(
                INVOKEVIRTUAL,
                className,
                "applyRangeScalar",
                "(Ljava/util/List;LTensor/Tensor;II)V",
                false
        );

        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void generateApplyRangeVectorMethod(
            ClassWriter cw,
            String className,
            List<Tensor> cluster,
            Tensor outputTensor,
            List<Tensor> externalInputsInOrder
    ) {
        MethodVisitor mv = cw.visitMethod(
                ACC_PUBLIC,
                "applyRangeVector",
                "(Ljava/util/List;LTensor/Tensor;II)V",
                null,
                null
        );
        mv.visitCode();

        List<Tensor> topoCluster = buildTopologicalOrder(outputTensor, cluster);
        Set<Tensor> clusterSet = new HashSet<>(topoCluster);
        if (!clusterSet.contains(outputTensor)) {
            throw new IllegalArgumentException("Output tensor is not part of fused cluster.");
        }
        List<Tensor> externalInputs = externalInputsInOrder != null
                ? new ArrayList<>(externalInputsInOrder)
                : findExternalInputs(topoCluster);
        Map<Tensor, Integer> externalInputIndex = new HashMap<>();
        for (int i = 0; i < externalInputs.size(); i++) {
            externalInputIndex.put(externalInputs.get(i), i);
        }

        SlotManager sm = buildVectorSlotLayout(externalInputs.size(), topoCluster.size());
        Map<Tensor, Integer> nodeVectorSlots = new HashMap<>();
        List<Integer> vecSlots = sm.getGroup(SlotKey.FUSED_NODE_VECTOR_VALUES);
        for (int i = 0; i < topoCluster.size(); i++) {
            nodeVectorSlots.put(topoCluster.get(i), vecSlots.get(i));
        }

        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR));
        mv.visitMethodInsn(INVOKEVIRTUAL, "Tensor/Tensor", "getData", "()[D", false);
        mv.visitVarInsn(ASTORE, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));

        List<Integer> inputSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS);
        for (int i = 0; i < externalInputs.size(); i++) {
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
            mv.visitLdcInsn(i);
            mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
            mv.visitTypeInsn(CHECKCAST, "Tensor/Tensor");
            mv.visitMethodInsn(INVOKEVIRTUAL, "Tensor/Tensor", "getData", "()[D", false);
            mv.visitVarInsn(ASTORE, inputSlots.get(i));
        }

        // width
        mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "width", "()I", false);
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.SECOND_LOOP_COUNTER));

        // upper = end - ((end-start) % width)
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
        mv.visitInsn(ISUB);
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.SECOND_LOOP_COUNTER));
        mv.visitInsn(IREM);
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.RANGE_UPPER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_UPPER));
        mv.visitInsn(ISUB);
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.RANGE_UPPER));

        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.LOOP_COUNTER));

        Label loopStart = new Label();
        Label loopEnd = new Label();
        mv.visitLabel(loopStart);
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_UPPER));
        mv.visitJumpInsn(IF_ICMPGE, loopEnd);

        for (Tensor node : topoCluster) {
            generateNodeEvaluationVectorBytecode(mv, className, node, clusterSet, externalInputIndex, nodeVectorSlots, sm);
            mv.visitVarInsn(ASTORE, nodeVectorSlots.get(node));
        }

        mv.visitVarInsn(ALOAD, nodeVectorSlots.get(outputTensor));
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitMethodInsn(
                INVOKESTATIC,
                "Graph/codegen/FusedVectorOps",
                "intoArray",
                "(Ljava/lang/Object;[DI)V",
                false
        );

        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.SECOND_LOOP_COUNTER));
        mv.visitInsn(IADD);
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);

        // Tail scalar for remaining elements
        Label noTail = new Label();
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_UPPER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitJumpInsn(IF_ICMPGE, noTail);
        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_UPPER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitMethodInsn(
                INVOKEVIRTUAL,
                className,
                "applyRangeScalar",
                "(Ljava/util/List;LTensor/Tensor;II)V",
                false
        );
        mv.visitLabel(noTail);

        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void generateApplyRangeScalarMethod(
            ClassWriter cw,
            String className,
            List<Tensor> cluster,
            Tensor outputTensor,
            List<Tensor> externalInputsInOrder
    ) {
        MethodVisitor mv = cw.visitMethod(
                ACC_PUBLIC,
                "applyRangeScalar",
                "(Ljava/util/List;LTensor/Tensor;II)V",
                null,
                null
        );
        mv.visitCode();

        List<Tensor> topoCluster = buildTopologicalOrder(outputTensor, cluster);
        Set<Tensor> clusterSet = new HashSet<>(topoCluster);
        if (!clusterSet.contains(outputTensor)) {
            throw new IllegalArgumentException("Output tensor is not part of fused cluster.");
        }

        List<Tensor> externalInputs = externalInputsInOrder != null
                ? new ArrayList<>(externalInputsInOrder)
                : findExternalInputs(topoCluster);
        Map<Tensor, Integer> externalInputIndex = new HashMap<>();
        for (int i = 0; i < externalInputs.size(); i++) {
            externalInputIndex.put(externalInputs.get(i), i);
        }

        SlotManager sm = buildRangeSlotLayout(externalInputs.size(), topoCluster.size());
        List<Integer> nodeValueSlots = sm.getGroup(SlotKey.FUSED_NODE_VALUES);
        Map<Tensor, Integer> nodeSlotMap = new HashMap<>();
        for (int i = 0; i < topoCluster.size(); i++) {
            nodeSlotMap.put(topoCluster.get(i), nodeValueSlots.get(i));
        }

        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR));
        mv.visitMethodInsn(INVOKEVIRTUAL, "Tensor/Tensor", "getData", "()[D", false);
        mv.visitVarInsn(ASTORE, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));

        List<Integer> inputSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS);
        for (int i = 0; i < externalInputs.size(); i++) {
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
            mv.visitLdcInsn(i);
            mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
            mv.visitTypeInsn(CHECKCAST, "Tensor/Tensor");
            mv.visitMethodInsn(INVOKEVIRTUAL, "Tensor/Tensor", "getData", "()[D", false);
            mv.visitVarInsn(ASTORE, inputSlots.get(i));
        }

        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.LOOP_COUNTER));

        Label loopStart = new Label();
        Label loopEnd = new Label();
        mv.visitLabel(loopStart);

        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitJumpInsn(IF_ICMPGE, loopEnd);

        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));

        for (Tensor node : topoCluster) {
            generateNodeEvaluationBytecode(mv, className, node, clusterSet, externalInputIndex, nodeSlotMap, sm);
            mv.visitVarInsn(DSTORE, nodeSlotMap.get(node));
        }
        mv.visitVarInsn(DLOAD, nodeSlotMap.get(outputTensor));
        mv.visitInsn(DASTORE);

        mv.visitIincInsn(sm.get(SlotKey.LOOP_COUNTER), 1);
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);
        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void generateNodeEvaluationBytecode(
            MethodVisitor mv,
            String className,
            Tensor current,
            Set<Tensor> clusterSet,
            Map<Tensor, Integer> externalInputIndex,
            Map<Tensor, Integer> nodeSlotMap,
            SlotManager sm
    ) {
        if (!clusterSet.contains(current)) {
            throw new IllegalArgumentException("Tensor is not in fused cluster.");
        }

        List<Tensor> parents = current.getPrevTensors();
        if (parents != null) {
            for (Tensor p : parents) {
                loadTensorValue(mv, p, clusterSet, externalInputIndex, nodeSlotMap, sm);
            }
        }

        if (current.getOperation() == null) {
            throw new UnsupportedOperationException("Fused node without operation is not supported.");
        }

        String op = current.getOperation().getClass().getSimpleName().toLowerCase(Locale.ROOT);
        switch (op) {
            case "add":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "add", "(DDI)D", false);
                break;
            case "sub":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "sub", "(DDI)D", false);
                break;
            case "mul":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "mul", "(DDI)D", false);
                break;
            case "div":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "div", "(DDI)D", false);
                break;
            case "neg":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "neg", "(DI)D", false);
                break;
            case "inv":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "inv", "(DI)D", false);
                break;
            case "log":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "log", "(DI)D", false);
                break;
            case "exp":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "exp", "(DI)D", false);
                break;
            case "tanh":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "tanh", "(DI)D", false);
                break;
            case "pow":
                handlePow(mv, className, current, sm);
                break;
            case "sqrt":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "sqrt", "(DI)D", false);
                break;
            case "mulscalar":
                Operations.mulScalar opInstance = (Operations.mulScalar) current.getOperation();
                mv.visitLdcInsn(opInstance.getScalar());
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "mulScalar", "(DDI)D", false);
                break;
            case "relu":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "relu", "(DI)D", false);
                break;
            case "sigmoid":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "sigmoid", "(DI)D", false);
                break;
            case "noop":
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "noop", "(DI)D", false);
                break;
            default:
                throw new UnsupportedOperationException("Operation " + op + " is not supported for fusing.");
        }
    }

    private static void loadTensorValue(
            MethodVisitor mv,
            Tensor tensor,
            Set<Tensor> clusterSet,
            Map<Tensor, Integer> externalInputIndex,
            Map<Tensor, Integer> nodeSlotMap,
            SlotManager sm
    ) {
        if (clusterSet.contains(tensor)) {
            Integer slot = nodeSlotMap.get(tensor);
            if (slot == null) {
                throw new IllegalArgumentException("Missing local slot for fused node.");
            }
            mv.visitVarInsn(DLOAD, slot);
            return;
        }

        Integer inputIdx = externalInputIndex.get(tensor);
        if (inputIdx == null) {
            throw new IllegalArgumentException("Missing external input mapping for tensor.");
        }
        int inputSlot = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS).get(inputIdx);
        mv.visitVarInsn(ALOAD, inputSlot);
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitInsn(DALOAD);
    }

    private static void handlePow(MethodVisitor mv, String className, Tensor current, SlotManager sm) {
        if (!(current.getOperation() instanceof Operations.pow p)) {
            throw new UnsupportedOperationException("pow operation instance is missing exponent metadata.");
        }
        double exponent = p.getExponent();
        mv.visitLdcInsn(exponent);
        emitPrecisionMode(mv, className);
        mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "pow", "(DDI)D", false);
    }

    private static void generateMetadataMethods(ClassWriter cw, String className) {
        MethodVisitor mv = cw.visitMethod(ACC_PUBLIC, "isElementWise", "()Z", null, null);
        mv.visitInsn(ICONST_1);
        mv.visitInsn(IRETURN);
        mv.visitMaxs(1, 1);
        mv.visitEnd();

        mv = cw.visitMethod(ACC_PUBLIC, "getExpression", "()Ljava/lang/String;", null, null);
        mv.visitVarInsn(ALOAD, 0);
        mv.visitFieldInsn(GETFIELD, className, "expression", "Ljava/lang/String;");
        mv.visitInsn(ARETURN);
        mv.visitMaxs(1, 1);
        mv.visitEnd();

        mv = cw.visitMethod(ACC_PUBLIC, "supportsBackend", "(LBackend/ComputeBackend;)Z", null, null);
        Label notCpu = new Label();
        mv.visitVarInsn(ALOAD, 1);
        mv.visitFieldInsn(GETSTATIC, "Backend/ComputeBackend", "CPU", "LBackend/ComputeBackend;");
        mv.visitJumpInsn(IF_ACMPNE, notCpu);
        mv.visitInsn(ICONST_1);
        Label endSupports = new Label();
        mv.visitJumpInsn(GOTO, endSupports);
        mv.visitLabel(notCpu);
        mv.visitInsn(ICONST_0);
        mv.visitLabel(endSupports);
        mv.visitInsn(IRETURN);
        mv.visitMaxs(2, 2);
        mv.visitEnd();

        mv = cw.visitMethod(ACC_PUBLIC, "getPreferredBackend", "()LBackend/ComputeBackend;", null, null);
        mv.visitFieldInsn(GETSTATIC, "Backend/ComputeBackend", "CPU", "LBackend/ComputeBackend;");
        mv.visitInsn(ARETURN);
        mv.visitMaxs(1, 1);
        mv.visitEnd();

        mv = cw.visitMethod(ACC_PUBLIC, "requiresOutputForGradient", "()Z", null, null);
        mv.visitInsn(ICONST_0);
        mv.visitInsn(IRETURN);
        mv.visitMaxs(1, 1);
        mv.visitEnd();

        mv = cw.visitMethod(ACC_PUBLIC, "opType", "()LOperations/Operation$OpType;", null, null);
        mv.visitFieldInsn(GETSTATIC, "Operations/Operation$OpType", "FUSED", "LOperations/Operation$OpType;");
        mv.visitInsn(ARETURN);
        mv.visitMaxs(1, 1);
        mv.visitEnd();
    }

    private static void generateNodeEvaluationVectorBytecode(
            MethodVisitor mv,
            String className,
            Tensor current,
            Set<Tensor> clusterSet,
            Map<Tensor, Integer> externalInputIndex,
            Map<Tensor, Integer> nodeVectorSlots,
            SlotManager sm
    ) {
        if (!clusterSet.contains(current)) {
            throw new IllegalArgumentException("Tensor is not in fused cluster.");
        }

        List<Tensor> parents = current.getPrevTensors();
        if (parents != null) {
            for (Tensor p : parents) {
                loadTensorVectorValue(mv, p, clusterSet, externalInputIndex, nodeVectorSlots, sm);
            }
        }

        if (current.getOperation() == null) {
            throw new UnsupportedOperationException("Fused node without operation is not supported.");
        }
        String op = current.getOperation().getClass().getSimpleName().toLowerCase(Locale.ROOT);
        switch (op) {
            case "add" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "add", "(Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "sub" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "sub", "(Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "mul" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "mul", "(Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "div" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "div", "(Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "neg" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "neg", "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "inv" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "inv", "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "log" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "log", "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "exp" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "exp", "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "tanh" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "tanh", "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "sqrt" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "sqrt", "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "relu" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "relu", "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "sigmoid" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "sigmoid", "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "noop" -> {
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "noop", "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            case "mulscalar" -> {
                Operations.mulScalar ms = (Operations.mulScalar) current.getOperation();
                mv.visitLdcInsn(ms.getScalar());
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "mulScalar", "(Ljava/lang/Object;DI)Ljava/lang/Object;", false);
            }
            case "pow" -> {
                if (!(current.getOperation() instanceof Operations.pow p)) {
                    throw new UnsupportedOperationException("pow operation instance is missing exponent metadata.");
                }
                mv.visitLdcInsn(p.getExponent());
                emitPrecisionMode(mv, className);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "pow", "(Ljava/lang/Object;DI)Ljava/lang/Object;", false);
            }
            default -> throw new UnsupportedOperationException("Operation " + op + " is not supported for fused vector execution.");
        }
    }

    private static void loadTensorVectorValue(
            MethodVisitor mv,
            Tensor tensor,
            Set<Tensor> clusterSet,
            Map<Tensor, Integer> externalInputIndex,
            Map<Tensor, Integer> nodeVectorSlots,
            SlotManager sm
    ) {
        if (clusterSet.contains(tensor)) {
            Integer slot = nodeVectorSlots.get(tensor);
            if (slot == null) {
                throw new IllegalArgumentException("Missing vector slot for fused node.");
            }
            mv.visitVarInsn(ALOAD, slot);
            return;
        }

        Integer inputIdx = externalInputIndex.get(tensor);
        if (inputIdx == null) {
            throw new IllegalArgumentException("Missing external input mapping for tensor.");
        }
        int inputArraySlot = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS).get(inputIdx);
        mv.visitVarInsn(ALOAD, inputArraySlot);
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitMethodInsn(
                INVOKESTATIC,
                "Graph/codegen/FusedVectorOps",
                "fromArray",
                "([DI)Ljava/lang/Object;",
                false
        );
    }

    private static List<Tensor> findExternalInputs(List<Tensor> cluster) {
        Set<Tensor> clusterSet = new HashSet<>(cluster);
        Set<Tensor> external = new LinkedHashSet<>();
        for (Tensor t : cluster) {
            List<Tensor> parents = t.getPrevTensors();
            if (parents == null) {
                continue;
            }
            for (Tensor p : parents) {
                if (!clusterSet.contains(p)) {
                    external.add(p);
                }
            }
        }
        return new ArrayList<>(external);
    }

    private static List<Tensor> buildTopologicalOrder(Tensor outputTensor, List<Tensor> cluster) {
        Set<Tensor> clusterSet = new HashSet<>(cluster);
        if (!clusterSet.contains(outputTensor)) {
            throw new IllegalArgumentException("Output tensor is not in the provided cluster.");
        }

        Set<Tensor> reachable = new LinkedHashSet<>();
        Deque<Tensor> stack = new ArrayDeque<>();
        stack.push(outputTensor);
        while (!stack.isEmpty()) {
            Tensor current = stack.pop();
            if (!clusterSet.contains(current) || !reachable.add(current)) {
                continue;
            }
            List<Tensor> parents = current.getPrevTensors();
            if (parents != null) {
                for (Tensor prev : parents) {
                    if (clusterSet.contains(prev)) {
                        stack.push(prev);
                    }
                }
            }
        }

        Map<Tensor, Integer> indegree = new HashMap<>();
        Map<Tensor, List<Tensor>> adjacency = new HashMap<>();
        for (Tensor node : reachable) {
            indegree.put(node, 0);
            adjacency.put(node, new ArrayList<>());
        }
        for (Tensor node : reachable) {
            List<Tensor> parents = node.getPrevTensors();
            if (parents == null) {
                continue;
            }
            for (Tensor prev : parents) {
                if (reachable.contains(prev)) {
                    indegree.put(node, indegree.get(node) + 1);
                    adjacency.get(prev).add(node);
                }
            }
        }

        Deque<Tensor> queue = new ArrayDeque<>();
        for (Map.Entry<Tensor, Integer> e : indegree.entrySet()) {
            if (e.getValue() == 0) {
                queue.add(e.getKey());
            }
        }

        List<Tensor> topo = new ArrayList<>(reachable.size());
        while (!queue.isEmpty()) {
            Tensor node = queue.removeFirst();
            topo.add(node);
            for (Tensor child : adjacency.get(node)) {
                int next = indegree.get(child) - 1;
                indegree.put(child, next);
                if (next == 0) {
                    queue.addLast(child);
                }
            }
        }

        if (topo.size() != reachable.size()) {
            throw new IllegalStateException("Cycle detected inside fused cluster.");
        }
        return topo;
    }

    private static SlotManager buildRangeSlotLayout(int externalInputCount, int nodeCount) {
        SlotManager sm = new SlotManager();
        sm.define(SlotKey.CLUSTER_TENSOR_INPUTS);
        sm.define(SlotKey.CLUSTER_TENSOR);
        sm.define(SlotKey.RANGE_START);
        sm.define(SlotKey.RANGE_END);
        sm.define(SlotKey.CLUSTER_TENSOR_VALUES);
        sm.define(SlotKey.LOOP_COUNTER);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.FUSED_NODE_VALUES, nodeCount);
        sm.define(SlotKey.TMP_REGISTER);
        return sm;
    }

    private static SlotManager buildVectorSlotLayout(int externalInputCount, int nodeCount) {
        SlotManager sm = new SlotManager();
        sm.define(SlotKey.CLUSTER_TENSOR_INPUTS);
        sm.define(SlotKey.CLUSTER_TENSOR);
        sm.define(SlotKey.RANGE_START);
        sm.define(SlotKey.RANGE_END);
        sm.define(SlotKey.CLUSTER_TENSOR_VALUES);
        sm.define(SlotKey.LOOP_COUNTER);
        sm.define(SlotKey.SECOND_LOOP_COUNTER);
        sm.define(SlotKey.RANGE_UPPER);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.FUSED_NODE_VECTOR_VALUES, nodeCount);
        return sm;
    }

    private static void emitPrecisionMode(MethodVisitor mv, String className) {
        mv.visitVarInsn(ALOAD, 0);
        mv.visitFieldInsn(GETFIELD, className, "precisionMode", "I");
    }
}
