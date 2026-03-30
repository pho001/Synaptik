package graph.codegen;

import operations.Operation;
import tensor.Tensor;
import utils.SlotKey;
import utils.SlotManager;
import org.objectweb.asm.*;

import java.util.*;

public class HFusedOperationGenerator implements Opcodes {

    public static byte[] generate(
            String internalClassName,
            List<Tensor> cluster,
            Tensor outputTensor,
            List<Tensor> externalInputsInOrder
    ) {
        final int precisionMode = FusedDTypeOps.MODE_F16;
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

        MethodVisitor mv = cw.visitMethod(ACC_PUBLIC, "<init>", "(Ljava/util/List;Ljava/lang/String;I)V", null, null);
        mv.visitCode();
        mv.visitVarInsn(ALOAD, 0);
        mv.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ALOAD, 2);
        mv.visitFieldInsn(PUTFIELD, className, "expression", "Ljava/lang/String;");
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
        mv.visitMethodInsn(INVOKEVIRTUAL, "Tensor/Tensor", "getFlatDataSize", "()I", false);
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
        final int precisionMode = FusedDTypeOps.MODE_F16;
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

        List<Integer> inputSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS);
        List<Integer> cachedInputVectorSlots = sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS);
        for (int i = 0; i < externalInputs.size(); i++) {
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
            mv.visitLdcInsn(i);
            mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
            mv.visitTypeInsn(CHECKCAST, "Tensor/Tensor");
            emitGetRawArrayFromTensorCall(mv, precisionMode);
            mv.visitVarInsn(ASTORE, inputSlots.get(i));
        }
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR));
        emitGetRawArrayFromTensorCall(mv, precisionMode);
        mv.visitVarInsn(ASTORE, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));

        // width
        emitVectorWidthCall(mv, precisionMode);
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

        // Local CSE: load each external input vector once per iteration.
        for (int i = 0; i < externalInputs.size(); i++) {
            mv.visitVarInsn(ALOAD, inputSlots.get(i));
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
            emitLoadVectorFromArrayCall(mv, precisionMode);
            mv.visitVarInsn(ASTORE, cachedInputVectorSlots.get(i));
        }

        for (Tensor node : topoCluster) {
            generateNodeEvaluationVectorBytecode(
                    mv, className, node, clusterSet, externalInputIndex, nodeVectorSlots, sm
            );
            mv.visitVarInsn(ASTORE, nodeVectorSlots.get(node));
        }

        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(ALOAD, nodeVectorSlots.get(outputTensor));
        emitStoreVectorToArrayCall(mv, precisionMode);

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

        List<Integer> inputSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS);
        for (int i = 0; i < externalInputs.size(); i++) {
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
            mv.visitLdcInsn(i);
            mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
            mv.visitTypeInsn(CHECKCAST, "Tensor/Tensor");
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

        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));

        for (Tensor node : topoCluster) {
            generateNodeEvaluationBytecode(mv, className, node, clusterSet, externalInputIndex, nodeSlotMap, sm);
            mv.visitVarInsn(DSTORE, nodeSlotMap.get(node));
        }
        mv.visitVarInsn(DLOAD, nodeSlotMap.get(outputTensor));
        emitPrecisionMode(mv);
        mv.visitMethodInsn(
                INVOKESTATIC,
                "Graph/codegen/FusedStorageOps",
                "storeScalar",
                "(LTensor/Tensor;IDI)V",
                false
        );

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
                loadTensorValue(mv, className, p, clusterSet, externalInputIndex, nodeSlotMap, sm);
            }
        }

        if (current.getOperation() == null) {
            throw new UnsupportedOperationException("Fused node without operation is not supported.");
        }

        String op = current.getOperation().getClass().getSimpleName().toLowerCase(Locale.ROOT);
        switch (op) {
            case "add":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "add", "(DDI)D", false);
                break;
            case "sub":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "sub", "(DDI)D", false);
                break;
            case "mul":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "mul", "(DDI)D", false);
                break;
            case "div":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "div", "(DDI)D", false);
                break;
            case "min":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "min", "(DDI)D", false);
                break;
            case "max":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "max", "(DDI)D", false);
                break;
            case "neg":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "neg", "(DI)D", false);
                break;
            case "inv":
                mv.visitVarInsn(DSTORE, sm.get(SlotKey.TMP_REGISTER));
                mv.visitLdcInsn(1.0d);
                mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
                mv.visitInsn(DDIV);
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
                break;
            case "log":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "log", "(DI)D", false);
                break;
            case "exp":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "exp", "(DI)D", false);
                break;
            case "fastexp":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "fastExp", "(DI)D", false);
                break;
            case "tanh":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "tanh", "(DI)D", false);
                break;
            case "fasttanh":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "fastTanh", "(DI)D", false);
                break;
            case "pow":
                handlePow(mv, className, current, sm);
                break;
            case "sqrt":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "sqrt", "(DI)D", false);
                break;
            case "mulscalar":
                operations.mulScalar opInstance = (operations.mulScalar) current.getOperation();
                mv.visitLdcInsn(opInstance.getScalar());
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "mulScalar", "(DDI)D", false);
                break;
            case "relu":
                mv.visitInsn(DCONST_0);
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(DD)D", false);
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
                break;
            case "sigmoid":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "sigmoid", "(DI)D", false);
                break;
            case "noop":
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "noop", "(DI)D", false);
                break;
            default:
                throw new UnsupportedOperationException("Operation " + op + " is not supported for fusing.");
        }
    }

    private static void loadTensorValue(
            MethodVisitor mv,
            String className,
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
        emitPrecisionMode(mv);
        mv.visitMethodInsn(
                INVOKESTATIC,
                "Graph/codegen/FusedStorageOps",
                "loadScalar",
                "(LTensor/Tensor;II)D",
                false
        );
    }

    private static void handlePow(MethodVisitor mv, String className, Tensor current, SlotManager sm) {
        if (!(current.getOperation() instanceof operations.pow p)) {
            throw new UnsupportedOperationException("pow operation instance is missing exponent metadata.");
        }
        double exponent = p.getExponent();
        if (Double.compare(exponent, 0.0d) == 0) {
            mv.visitInsn(POP2);
            mv.visitInsn(DCONST_1);
            emitPrecisionMode(mv);
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
            return;
        }
        if (Double.compare(exponent, 1.0d) == 0) {
            emitPrecisionMode(mv);
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
            return;
        }
        if (Double.compare(exponent, -1.0d) == 0) {
            mv.visitVarInsn(DSTORE, sm.get(SlotKey.TMP_REGISTER));
            mv.visitLdcInsn(1.0d);
            mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
            mv.visitInsn(DDIV);
            emitPrecisionMode(mv);
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
            return;
        }
        if (Double.compare(exponent, 2.0d) == 0) {
            mv.visitVarInsn(DSTORE, sm.get(SlotKey.TMP_REGISTER));
            mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
            mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
            mv.visitInsn(DMUL);
            emitPrecisionMode(mv);
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
            return;
        }
        mv.visitLdcInsn(exponent);
        mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "pow", "(DD)D", false);
        emitPrecisionMode(mv);
        mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
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
        final int precisionMode = FusedDTypeOps.MODE_F16;
        if (!clusterSet.contains(current)) {
            throw new IllegalArgumentException("Tensor is not in fused cluster.");
        }

        List<Tensor> parents = current.getPrevTensors();
        if (parents != null) {
            for (Tensor p : parents) {
                loadTensorVectorValue(mv, className, p, clusterSet, externalInputIndex, nodeVectorSlots, sm);
            }
        }

        if (current.getOperation() == null) {
            throw new UnsupportedOperationException("Fused node without operation is not supported.");
        }
        String op = current.getOperation().getClass().getSimpleName().toLowerCase(Locale.ROOT);
        switch (op) {
            case "add" -> emitVectorBinaryOpCall(mv, "add", precisionMode);
            case "sub" -> emitVectorBinaryOpCall(mv, "sub", precisionMode);
            case "mul" -> emitVectorBinaryOpCall(mv, "mul", precisionMode);
            case "div" -> emitVectorBinaryOpCall(mv, "div", precisionMode);
            case "min" -> emitVectorBinaryOpCall(mv, "min", precisionMode);
            case "max" -> emitVectorBinaryOpCall(mv, "max", precisionMode);
            case "neg" -> emitVectorUnaryOpCall(mv, "neg", precisionMode);
            case "inv" -> emitVectorUnaryOpCall(mv, "inv", precisionMode);
            case "log" -> emitVectorUnaryOpCall(mv, "log", precisionMode);
            case "exp" -> emitVectorUnaryOpCall(mv, "exp", precisionMode);
            case "fastexp" -> emitVectorUnaryOpCall(mv, "fastExp", precisionMode);
            case "tanh" -> emitVectorUnaryOpCall(mv, "tanh", precisionMode);
            case "fasttanh" -> emitVectorUnaryOpCall(mv, "fastTanh", precisionMode);
            case "sqrt" -> emitVectorUnaryOpCall(mv, "sqrt", precisionMode);
            case "relu" -> emitVectorUnaryOpCall(mv, "relu", precisionMode);
            case "sigmoid" -> emitVectorUnaryOpCall(mv, "sigmoid", precisionMode);
            case "noop" -> emitVectorUnaryOpCall(mv, "noop", precisionMode);
            case "mulscalar" -> {
                operations.mulScalar ms = (operations.mulScalar) current.getOperation();
                if (precisionMode == FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn(ms.getScalarF32());
                } else {
                    mv.visitLdcInsn(ms.getScalar());
                }
                emitVectorMulScalarCall(mv, precisionMode);
            }
            case "pow" -> {
                if (!(current.getOperation() instanceof operations.pow p)) {
                    throw new UnsupportedOperationException("pow operation instance is missing exponent metadata.");
                }
                if (precisionMode == FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn(p.getExponentF32());
                } else {
                    mv.visitLdcInsn(p.getExponent());
                }
                emitVectorPowCall(mv, precisionMode);
            }
            default -> throw new UnsupportedOperationException("Operation " + op + " is not supported for fused vector execution.");
        }
    }

    private static void loadTensorVectorValue(
            MethodVisitor mv,
            String className,
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
        int cachedSlot = sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS).get(inputIdx);
        mv.visitVarInsn(ALOAD, cachedSlot);
    }

    private static void emitGetRawArrayFromTensorCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKEVIRTUAL, "Tensor/Tensor", "getFloat32Data", "()[F", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitMethodInsn(INVOKEVIRTUAL, "Tensor/Tensor", "getFloat64Data", "()[D", false);
        } else {
            mv.visitMethodInsn(INVOKEVIRTUAL, "Tensor/Tensor", "getFloat16Data", "()[S", false);
        }
    }

    private static void emitLoadVectorFromArrayCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitFieldInsn(GETSTATIC, "jdk/incubator/vector/FloatVector", "SPECIES_PREFERRED", "Ljdk/incubator/vector/VectorSpecies;");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKESTATIC, "jdk/incubator/vector/FloatVector", "fromArray", "(Ljdk/incubator/vector/VectorSpecies;[FI)Ljdk/incubator/vector/FloatVector;", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitFieldInsn(GETSTATIC, "jdk/incubator/vector/DoubleVector", "SPECIES_PREFERRED", "Ljdk/incubator/vector/VectorSpecies;");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKESTATIC, "jdk/incubator/vector/DoubleVector", "fromArray", "(Ljdk/incubator/vector/VectorSpecies;[DI)Ljdk/incubator/vector/DoubleVector;", false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedStorageOps", "loadVectorF16Array", "([SI)Ljava/lang/Object;", false);
        }
    }

    private static void emitStoreVectorToArrayCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/FloatVector");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/FloatVector", "intoArray", "([FI)V", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/DoubleVector");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/DoubleVector", "intoArray", "([DI)V", false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedStorageOps", "storeVectorF16Array", "([SILjava/lang/Object;)V", false);
        }
    }

    private static String vectorTypeDesc(int precisionMode) {
        return precisionMode == FusedDTypeOps.MODE_F32
                ? "Ljdk/incubator/vector/FloatVector;"
                : "Ljdk/incubator/vector/DoubleVector;";
    }

    private static void emitVectorWidthCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "widthF32", "()I", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "widthF64", "()I", false);
        } else {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "width", "(I)I", false);
        }
    }

    private static void emitLoadVectorCall(MethodVisitor mv, String className, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedStorageOps", "loadVectorF32", "(LTensor/Tensor;I)Ljdk/incubator/vector/FloatVector;", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedStorageOps", "loadVectorF64", "(LTensor/Tensor;I)Ljdk/incubator/vector/DoubleVector;", false);
        } else {
            emitPrecisionMode(mv);
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedStorageOps", "loadVector", "(LTensor/Tensor;II)Ljava/lang/Object;", false);
        }
    }

    private static void emitStoreVectorCall(MethodVisitor mv, String className, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedStorageOps", "storeVectorF32", "(LTensor/Tensor;ILjdk/incubator/vector/FloatVector;)V", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedStorageOps", "storeVectorF64", "(LTensor/Tensor;ILjdk/incubator/vector/DoubleVector;)V", false);
        } else {
            emitPrecisionMode(mv);
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedStorageOps", "storeVector", "(LTensor/Tensor;ILjava/lang/Object;I)V", false);
        }
    }

    private static void emitVectorBinaryOpCall(MethodVisitor mv, String op, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", op, "(Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
            return;
        }
        String suffix = precisionMode == FusedDTypeOps.MODE_F32 ? "F32" : "F64";
        String vd = vectorTypeDesc(precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", op + suffix, "(" + vd + vd + ")" + vd, false);
    }

    private static void emitVectorUnaryOpCall(MethodVisitor mv, String op, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", op, "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            return;
        }
        String suffix = precisionMode == FusedDTypeOps.MODE_F32 ? "F32" : "F64";
        String vd = vectorTypeDesc(precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", op + suffix, "(" + vd + ")" + vd, false);
    }

    private static void emitVectorMulScalarCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "mulScalar", "(Ljava/lang/Object;DI)Ljava/lang/Object;", false);
            return;
        }
        String vd = vectorTypeDesc(precisionMode);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "mulScalarF32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "mulScalarF64", "(" + vd + "D)" + vd, false);
        }
    }

    private static void emitVectorPowCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "pow", "(Ljava/lang/Object;DI)Ljava/lang/Object;", false);
            return;
        }
        String vd = vectorTypeDesc(precisionMode);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "powF32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "Graph/codegen/FusedVectorOps", "powF64", "(" + vd + "D)" + vd, false);
        }
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
        sm.defineGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.FUSED_NODE_VECTOR_VALUES, nodeCount);
        return sm;
    }

    private static void emitPrecisionMode(MethodVisitor mv) {
        mv.visitLdcInsn(FusedDTypeOps.MODE_F16);
    }
}
