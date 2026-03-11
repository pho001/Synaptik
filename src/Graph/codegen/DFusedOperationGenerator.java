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

        // Definice třídy: implements Operation
        cw.visit(V1_8, ACC_PUBLIC, className, null, "java/lang/Object", new String[]{"Operations/Operation"});

        // Metadata pole
        cw.visitField(ACC_PRIVATE | ACC_FINAL, "expression", "Ljava/lang/String;", null, null).visitEnd();

        // Konstruktor: public FusedOp(List cluster, String expr)
        MethodVisitor mv = cw.visitMethod(ACC_PUBLIC, "<init>", "(Ljava/util/List;Ljava/lang/String;)V", null, null);
        mv.visitCode();
        mv.visitVarInsn(ALOAD, 0);
        mv.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ALOAD, 2);
        mv.visitFieldInsn(PUTFIELD, className, "expression", "Ljava/lang/String;");
        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();

        // Implementace metod rozhraní (isElementWise, atd.)
        generateMetadataMethods(cw, className);

        // Jádro: apply(List<Tensor> inputs, Tensor out)
        generateApplyMethod(cw, className, cluster, outputTensor, externalInputsInOrder);

        cw.visitEnd();
        return cw.toByteArray();
    }

    private static void generateApplyMethod(
            ClassWriter cw,
            String className,
            List<Tensor> cluster,
            Tensor outputTensor,
            List<Tensor> externalInputsInOrder
    ) {
        MethodVisitor mv = cw.visitMethod(ACC_PUBLIC, "apply", "(Ljava/util/List;LTensor/Tensor;)V", null, null);
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

        SlotManager sm = buildSlotLayout(externalInputs.size(), topoCluster.size());
        List<Integer> nodeValueSlots = sm.getGroup(SlotKey.FUSED_NODE_VALUES);
        Map<Tensor, Integer> nodeSlotMap = new HashMap<>();
        for (int i = 0; i < topoCluster.size(); i++) {
            nodeSlotMap.put(topoCluster.get(i), nodeValueSlots.get(i));
        }

        // data pole výstupního tensoru
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR));
        mv.visitMethodInsn(INVOKEVIRTUAL, "Tensor/Tensor", "getData", "()[D", false);
        mv.visitVarInsn(ASTORE, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));

        // data pole externích vstupů
        List<Integer> inputSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS);
        for (int i = 0; i < externalInputs.size(); i++) {
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
            mv.visitLdcInsn(i);
            mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
            mv.visitTypeInsn(CHECKCAST, "Tensor/Tensor");
            mv.visitMethodInsn(INVOKEVIRTUAL, "Tensor/Tensor", "getData", "()[D", false);
            mv.visitVarInsn(ASTORE, inputSlots.get(i));
        }

        // Smyčka: for (int i = 0; i < outData.length; i++)
        mv.visitInsn(ICONST_0);
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.LOOP_COUNTER));

        Label loopStart = new Label();
        Label loopEnd = new Label();
        mv.visitLabel(loopStart);

        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
        mv.visitInsn(ARRAYLENGTH);
        mv.visitJumpInsn(IF_ICMPGE, loopEnd);

        // Adresace pro zápis: outData[i] = ...
        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_VALUES));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));

        // Vyhodnotíme každý uzel clusteru přesně jednou a uložíme ho do lokálního slotu.
        for (Tensor node : topoCluster) {
            generateNodeEvaluationBytecode(mv, node, clusterSet, externalInputIndex, nodeSlotMap, sm);
            mv.visitVarInsn(DSTORE, nodeSlotMap.get(node));
        }
        mv.visitVarInsn(DLOAD, nodeSlotMap.get(outputTensor));

        mv.visitInsn(DASTORE);

        // i++
        mv.visitIincInsn(sm.get(SlotKey.LOOP_COUNTER), 1);
        mv.visitJumpInsn(GOTO, loopStart);
        mv.visitLabel(loopEnd);

        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void generateNodeEvaluationBytecode(
            MethodVisitor mv,
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
            case "add": mv.visitInsn(DADD); break;
            case "sub": mv.visitInsn(DSUB); break;
            case "mul": mv.visitInsn(DMUL); break;
            case "div": mv.visitInsn(DDIV); break;
            case "neg": mv.visitInsn(DNEG); break;
            case "inv":
                mv.visitVarInsn(DSTORE, sm.get(SlotKey.TMP_REGISTER));
                mv.visitInsn(DCONST_1);
                mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
                mv.visitInsn(DDIV);
                break;
            case "log":
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "log", "(D)D", false);
                break;
            case "exp":
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "exp", "(D)D", false);
                break;
            case "tanh":
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "tanh", "(D)D", false);
                break;
            case "pow":
                handlePow(mv, current, sm);
                break;
            case "sqrt":
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                break;
            case "mulscalar":
                Operations.mulScalar opInstance = (Operations.mulScalar) current.getOperation();
                mv.visitLdcInsn(opInstance.getScalar());
                mv.visitInsn(DMUL);
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

    private static void handlePow(MethodVisitor mv, Tensor current, SlotManager sm) {
        if (!(current.getOperation() instanceof Operations.pow p)) {
            throw new UnsupportedOperationException("pow operation instance is missing exponent metadata.");
        }
        double exponent = p.getExponent();

        if (exponent == 0.0) {
            mv.visitInsn(POP2);
            mv.visitInsn(DCONST_1);
        } else if (exponent == 1.0) {
            // no-op
        } else if (exponent == 0.5) {
            mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
        } else if (exponent == -1.0) {
            mv.visitVarInsn(DSTORE, sm.get(SlotKey.TMP_REGISTER));
            mv.visitInsn(DCONST_1);
            mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
            mv.visitInsn(DDIV);
        } else if (exponent == 2.0) {
            mv.visitInsn(DUP2);
            mv.visitInsn(DMUL);
        } else {
            mv.visitLdcInsn(exponent);
            mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "pow", "(DD)D", false);
        }
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

    private static List<Tensor> findExternalInputs(List<Tensor> cluster) {
        Set<Tensor> external = new LinkedHashSet<>();
        for (Tensor t : cluster) {
            List<Tensor> parents = t.getPrevTensors();
            if (parents == null) {
                continue;
            }
            for (Tensor p : parents) {
                if (!cluster.contains(p)) {
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

    private static SlotManager buildSlotLayout(int externalInputCount, int nodeCount) {
        SlotManager sm = new SlotManager();
        sm.define(SlotKey.CLUSTER_TENSOR_INPUTS);
        sm.define(SlotKey.CLUSTER_TENSOR);
        sm.define(SlotKey.CLUSTER_TENSOR_VALUES);
        sm.define(SlotKey.LOOP_COUNTER);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.FUSED_NODE_VALUES, nodeCount);
        sm.define(SlotKey.TMP_REGISTER);
        return sm;
    }
}
