package graph.codegen;

import graph.codegen.FusedDTypeOps;
import org.objectweb.asm.MethodVisitor;
import tensor.Tensor;
import utils.SlotKey;
import utils.SlotManager;

import java.util.*;

import static org.objectweb.asm.Opcodes.*;

final class FusedAsmSupport {
    private FusedAsmSupport() {}

    static List<Tensor> buildTopologicalOrder(Tensor outputTensor, List<Tensor> cluster) {
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

    static SlotManager buildRangeSlotLayout(int externalInputCount, int nodeCount) {
        SlotManager sm = new SlotManager();
        sm.define(SlotKey.CLUSTER_TENSOR_INPUTS);
        sm.define(SlotKey.CLUSTER_TENSOR);
        sm.define(SlotKey.RANGE_START);
        sm.define(SlotKey.RANGE_END);
        sm.define(SlotKey.FUSED_OPTIONS);
        sm.define(SlotKey.CLUSTER_TENSOR_VALUES);
        sm.define(SlotKey.LOOP_COUNTER);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.FUSED_NODE_VALUES, nodeCount);
        sm.define(SlotKey.TMP_REGISTER);
        return sm;
    }

    static SlotManager buildVectorSlotLayout(int externalInputCount, int nodeCount) {
        SlotManager sm = new SlotManager();
        sm.define(SlotKey.CLUSTER_TENSOR_INPUTS);
        sm.define(SlotKey.CLUSTER_TENSOR);
        sm.define(SlotKey.RANGE_START);
        sm.define(SlotKey.RANGE_END);
        sm.define(SlotKey.FUSED_OPTIONS);
        sm.define(SlotKey.CLUSTER_TENSOR_VALUES);
        sm.define(SlotKey.LOOP_COUNTER);
        sm.define(SlotKey.SECOND_LOOP_COUNTER);
        sm.define(SlotKey.RANGE_UPPER);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.FUSED_NODE_VECTOR_VALUES, nodeCount);
        return sm;
    }

    static void emitGetRawArrayFromTensorCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKEVIRTUAL, "tensor/Tensor", "getFloat32Data", "()[F", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(INVOKEVIRTUAL, "tensor/Tensor", "getFloat64Data", "()[D", false);
        } else {
            mv.visitMethodInsn(INVOKEVIRTUAL, "tensor/Tensor", "getFloat16Data", "()[S", false);
        }
    }

    static void emitLoadVectorFromArrayCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitFieldInsn(GETSTATIC, "jdk/incubator/vector/FloatVector", "SPECIES_PREFERRED", "Ljdk/incubator/vector/VectorSpecies;");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKESTATIC, "jdk/incubator/vector/FloatVector", "fromArray", "(Ljdk/incubator/vector/VectorSpecies;[FI)Ljdk/incubator/vector/FloatVector;", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitFieldInsn(GETSTATIC, "jdk/incubator/vector/DoubleVector", "SPECIES_PREFERRED", "Ljdk/incubator/vector/VectorSpecies;");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKESTATIC, "jdk/incubator/vector/DoubleVector", "fromArray", "(Ljdk/incubator/vector/VectorSpecies;[DI)Ljdk/incubator/vector/DoubleVector;", false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedStorageOps", "loadVectorF16Array", "([SI)Ljava/lang/Object;", false);
        }
    }

    static void emitLoadVectorFromCursorCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "graph/codegen/FusedBroadcastVectorOps",
                    "loadVectorF32",
                    "(Lgraph/codegen/FusedBroadcastCursor;[F)Ljdk/incubator/vector/FloatVector;",
                    false
            );
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "graph/codegen/FusedBroadcastVectorOps",
                    "loadVectorF64",
                    "(Lgraph/codegen/FusedBroadcastCursor;[D)Ljdk/incubator/vector/DoubleVector;",
                    false
            );
        } else {
            throw new UnsupportedOperationException("FusedOperationGenerator vector cursor load is supported only for F32/F64.");
        }
    }

    static void emitStoreVectorToArrayCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/FloatVector");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/FloatVector", "intoArray", "([FI)V", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitTypeInsn(CHECKCAST, "jdk/incubator/vector/DoubleVector");
            mv.visitInsn(DUP_X2);
            mv.visitInsn(POP);
            mv.visitMethodInsn(INVOKEVIRTUAL, "jdk/incubator/vector/DoubleVector", "intoArray", "([DI)V", false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedStorageOps", "storeVectorF16Array", "([SILjava/lang/Object;)V", false);
        }
    }

    static String vectorTypeDesc(int precisionMode) {
        return precisionMode == FusedDTypeOps.MODE_F32
                ? "Ljdk/incubator/vector/FloatVector;"
                : "Ljdk/incubator/vector/DoubleVector;";
    }

    static void emitVectorWidthCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", "widthF32", "()I", false);
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", "widthF64", "()I", false);
        } else {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", "width", "(I)I", false);
        }
    }

    static void emitVectorBinaryOpCall(MethodVisitor mv, String op, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", op, "(Ljava/lang/Object;Ljava/lang/Object;I)Ljava/lang/Object;", false);
            return;
        }
        String suffix = precisionMode == FusedDTypeOps.MODE_F32 ? "F32" : "F64";
        String vd = vectorTypeDesc(precisionMode);
        mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", op + suffix, "(" + vd + vd + ")" + vd, false);
    }

    static void emitVectorUnaryOpCall(MethodVisitor mv, String op, int precisionMode) {
        emitVectorUnaryOpCall(mv, op, precisionMode, null);
    }

    static void emitVectorUnaryOpCall(MethodVisitor mv, String op, int precisionMode, SlotManager sm) {
        boolean expOp = "exp".equals(op);
        boolean tanhOp = "tanh".equals(op);
        if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitLdcInsn(precisionMode);
            if (expOp || tanhOp) {
                mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
                mv.visitMethodInsn(
                        INVOKEVIRTUAL,
                        "backend/kernels/cpu/fused/FusedExecutionOptions",
                        expOp ? "useFastExpApprox" : "useFastTanhApprox",
                        "()Z",
                        false
                );
                mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", op, "(Ljava/lang/Object;IZ)Ljava/lang/Object;", false);
            } else {
                mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", op, "(Ljava/lang/Object;I)Ljava/lang/Object;", false);
            }
            return;
        }
        String suffix = precisionMode == FusedDTypeOps.MODE_F32 ? "F32" : "F64";
        String vd = vectorTypeDesc(precisionMode);
        if (expOp || tanhOp) {
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
            mv.visitMethodInsn(
                    INVOKEVIRTUAL,
                    "backend/kernels/cpu/fused/FusedExecutionOptions",
                    expOp ? "useFastExpApprox" : "useFastTanhApprox",
                    "()Z",
                    false
            );
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", op + suffix, "(" + vd + "Z)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", op + suffix, "(" + vd + ")" + vd, false);
        }
    }

    static void emitVectorMulScalarCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", "mulScalar", "(Ljava/lang/Object;DI)Ljava/lang/Object;", false);
            return;
        }
        String vd = vectorTypeDesc(precisionMode);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", "mulScalarF32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", "mulScalarF64", "(" + vd + "D)" + vd, false);
        }
    }

    static void emitVectorPowCall(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F16) {
            mv.visitLdcInsn(precisionMode);
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", "pow", "(Ljava/lang/Object;DI)Ljava/lang/Object;", false);
            return;
        }
        String vd = vectorTypeDesc(precisionMode);
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", "powF32", "(" + vd + "F)" + vd, false);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedVectorOps", "powF64", "(" + vd + "D)" + vd, false);
        }
    }

    static void emitScalarLoadInsn(MethodVisitor mv, int slot, int precisionMode) {
        mv.visitVarInsn(precisionMode == FusedDTypeOps.MODE_F32 ? FLOAD : DLOAD, slot);
    }

    static void emitScalarStoreInsn(MethodVisitor mv, int slot, int precisionMode) {
        mv.visitVarInsn(precisionMode == FusedDTypeOps.MODE_F32 ? FSTORE : DSTORE, slot);
    }

    static void emitScalarArrayLoadInsn(MethodVisitor mv, int precisionMode) {
        mv.visitInsn(precisionMode == FusedDTypeOps.MODE_F32 ? FALOAD : DALOAD);
    }

    static void emitScalarArrayStoreInsn(MethodVisitor mv, int precisionMode) {
        mv.visitInsn(precisionMode == FusedDTypeOps.MODE_F32 ? FASTORE : DASTORE);
    }

    static void emitIntArrayConstant(MethodVisitor mv, int[] values) {
        mv.visitLdcInsn(values.length);
        mv.visitIntInsn(NEWARRAY, T_INT);
        for (int i = 0; i < values.length; i++) {
            mv.visitInsn(DUP);
            mv.visitLdcInsn(i);
            mv.visitLdcInsn(values[i]);
            mv.visitInsn(IASTORE);
        }
    }

    static void handlePow(MethodVisitor mv, Object parameter, SlotManager sm, int precisionMode) {
        double exponentValue = ((Number) parameter).doubleValue();
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            float exponent = (float) exponentValue;

            if (Float.compare(exponent, 0.0f) == 0) {
                mv.visitInsn(POP);
                mv.visitInsn(FCONST_1);
                return;
            }
            if (Float.compare(exponent, 1.0f) == 0) {
                return;
            }
            if (Float.compare(exponent, -1.0f) == 0) {
                mv.visitInsn(FCONST_1);
                mv.visitInsn(SWAP);
                mv.visitInsn(FDIV);
                return;
            }
            if (Float.compare(exponent, 2.0f) == 0) {
                mv.visitInsn(DUP);
                mv.visitInsn(FMUL);
                return;
            }
            if (Float.compare(exponent, 0.5f) == 0) {
                mv.visitInsn(F2D);
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                mv.visitInsn(D2F);
                return;
            }

            mv.visitLdcInsn(exponent);
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedScalarOps", "powF32", "(FF)F", false);
            return;
        }

        double exponent = exponentValue;
        if (Double.compare(exponent, 0.0d) == 0) {
            mv.visitInsn(POP2);
            mv.visitInsn(DCONST_1);
            return;
        }
        if (Double.compare(exponent, 1.0d) == 0) {
            return;
        }
        if (Double.compare(exponent, -1.0d) == 0) {
            emitScalarStoreInsn(mv, sm.get(SlotKey.TMP_REGISTER), precisionMode);
            mv.visitInsn(DCONST_1);
            emitScalarLoadInsn(mv, sm.get(SlotKey.TMP_REGISTER), precisionMode);
            mv.visitInsn(DDIV);
            return;
        }
        if (Double.compare(exponent, 2.0d) == 0) {
            mv.visitInsn(DUP2);
            mv.visitInsn(DMUL);
            return;
        }
        if (Double.compare(exponent, 0.5d) == 0) {
            mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
            return;
        }
        mv.visitLdcInsn(exponent);
        mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "pow", "(DD)D", false);
    }

    static void loadScalarRef(
            MethodVisitor mv,
            int ref,
            int[] nodeValueSlots,
            SlotManager sm,
            int precisionMode,
            java.util.List<FusedExternalInputPlan> inputPlans
    ) {
        if (ref < inputPlans.size()) {
            int inputSlot = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS).get(ref);
            mv.visitVarInsn(ALOAD, inputSlot);
            FusedExternalInputPlan meta = inputPlans.get(ref);
            if (meta.directIndex()) {
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
            } else {
                mv.visitVarInsn(ALOAD, sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS).get(ref));
                mv.visitMethodInsn(INVOKEVIRTUAL, "graph/codegen/FusedBroadcastCursor", "idx", "()I", false);
            }
            emitScalarArrayLoadInsn(mv, precisionMode);
            return;
        }

        int nodeIndex = ref - inputPlans.size();
        if (nodeIndex < 0 || nodeIndex >= nodeValueSlots.length) {
            throw new IllegalArgumentException("Invalid fused scalar ref " + ref);
        }
        emitScalarLoadInsn(mv, nodeValueSlots[nodeIndex], precisionMode);
    }

    static void loadVectorRef(
            MethodVisitor mv,
            int ref,
            int[] nodeVectorSlots,
            SlotManager sm
    ) {
        if (ref < sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS).size()) {
            int cachedSlot = sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS).get(ref);
            mv.visitVarInsn(ALOAD, cachedSlot);
            return;
        }

        int nodeIndex = ref - sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS).size();
        if (nodeIndex < 0 || nodeIndex >= nodeVectorSlots.length) {
            throw new IllegalArgumentException("Invalid fused vector ref " + ref);
        }
        mv.visitVarInsn(ALOAD, nodeVectorSlots[nodeIndex]);
    }
}
