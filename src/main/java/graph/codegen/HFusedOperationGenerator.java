package graph.codegen;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import utils.SlotKey;
import utils.SlotManager;

import java.util.List;

import static org.objectweb.asm.Opcodes.*;

public final class HFusedOperationGenerator {
    private HFusedOperationGenerator() {}

    public static byte[] generate(String internalClassName, FusedExpressionPlan plan) {
        ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_FRAMES);
        cw.visit(
                V1_8,
                ACC_PUBLIC | ACC_FINAL,
                internalClassName,
                null,
                "java/lang/Object",
                new String[]{"backend/kernels/cpu/fused/CompiledFusedKernel"}
        );

        emitConstructor(cw);
        emitApplyRangeScalar(cw, internalClassName, plan);
        emitApplyRangeVector(cw, internalClassName);

        cw.visitEnd();
        return cw.toByteArray();
    }

    private static void emitConstructor(ClassWriter cw) {
        MethodVisitor mv = cw.visitMethod(ACC_PUBLIC, "<init>", "()V", null, null);
        mv.visitCode();
        mv.visitVarInsn(ALOAD, 0);
        mv.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void emitApplyRangeScalar(ClassWriter cw, String className, FusedExpressionPlan plan) {
        MethodVisitor mv = cw.visitMethod(
                ACC_PUBLIC,
                "applyRangeScalar",
                "(Ljava/util/List;Ltensor/Tensor;IILbackend/kernels/cpu/fused/FusedExecutionOptions;)V",
                null,
                null
        );
        mv.visitCode();

        SlotManager sm = buildScalarSlotLayout(plan.inputCount(), plan.nodeCount());
        int[] nodeValueSlots = sm.getGroup(SlotKey.FUSED_NODE_VALUES).stream().mapToInt(Integer::intValue).toArray();

        List<Integer> inputSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS);
        for (int i = 0; i < plan.inputCount(); i++) {
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR_INPUTS));
            mv.visitLdcInsn(i);
            mv.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "get", "(I)Ljava/lang/Object;", true);
            mv.visitTypeInsn(CHECKCAST, "tensor/Tensor");
            mv.visitVarInsn(ASTORE, inputSlots.get(i));
        }

        List<Integer> cursorSlots = sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS);
        for (int i = 0; i < plan.inputCount(); i++) {
            FusedExternalInputPlan input = plan.inputs().get(i);
            if (input.directIndex()) {
                continue;
            }
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
            emitIntArrayConstant(mv, input.outShape());
            emitIntArrayConstant(mv, input.outStrides());
            emitIntArrayConstant(mv, input.effStrides());
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "graph/codegen/FusedBroadcastCursor",
                    "atStart",
                    "(I[I[I[I)Lgraph/codegen/FusedBroadcastCursor;",
                    false
            );
            mv.visitVarInsn(ASTORE, cursorSlots.get(i));
        }

        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_START));
        mv.visitVarInsn(ISTORE, sm.get(SlotKey.LOOP_COUNTER));

        Label loopStart = new Label();
        Label loopEnd = new Label();
        mv.visitLabel(loopStart);
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.RANGE_END));
        mv.visitJumpInsn(IF_ICMPGE, loopEnd);

        for (FusedNodePlan node : plan.nodes()) {
            emitNodeEvaluation(mv, node, plan.inputs(), nodeValueSlots, inputSlots, cursorSlots, sm);
            mv.visitVarInsn(DSTORE, nodeValueSlots[node.index()]);
        }

        mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_TENSOR));
        mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        mv.visitVarInsn(DLOAD, nodeValueSlots[plan.outputRef() - plan.inputCount()]);
        emitPrecisionMode(mv);
        mv.visitMethodInsn(
                INVOKESTATIC,
                "graph/codegen/FusedStorageOps",
                "storeScalar",
                "(Ltensor/Tensor;IDI)V",
                false
        );

        for (int i = 0; i < plan.inputCount(); i++) {
            FusedExternalInputPlan input = plan.inputs().get(i);
            if (input.directIndex()) {
                continue;
            }
            mv.visitVarInsn(ALOAD, cursorSlots.get(i));
            mv.visitMethodInsn(INVOKEVIRTUAL, "graph/codegen/FusedBroadcastCursor", "step", "()V", false);
        }

        mv.visitIincInsn(sm.get(SlotKey.LOOP_COUNTER), 1);
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);
        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void emitApplyRangeVector(ClassWriter cw, String className) {
        MethodVisitor mv = cw.visitMethod(
                ACC_PUBLIC,
                "applyRangeVector",
                "(Ljava/util/List;Ltensor/Tensor;IILbackend/kernels/cpu/fused/FusedExecutionOptions;)V",
                null,
                null
        );
        mv.visitCode();
        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ALOAD, 1);
        mv.visitVarInsn(ALOAD, 2);
        mv.visitVarInsn(ILOAD, 3);
        mv.visitVarInsn(ILOAD, 4);
        mv.visitVarInsn(ALOAD, 5);
        mv.visitMethodInsn(
                INVOKEVIRTUAL,
                className,
                "applyRangeScalar",
                "(Ljava/util/List;Ltensor/Tensor;IILbackend/kernels/cpu/fused/FusedExecutionOptions;)V",
                false
        );
        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void emitNodeEvaluation(
            MethodVisitor mv,
            FusedNodePlan node,
            List<FusedExternalInputPlan> inputPlans,
            int[] nodeValueSlots,
            List<Integer> inputSlots,
            List<Integer> cursorSlots,
            SlotManager sm
    ) {
        for (int ref : node.inputRefs()) {
            loadRef(mv, ref, inputPlans, nodeValueSlots, inputSlots, cursorSlots, sm);
        }

        switch (node.opType()) {
            case ADD -> emitDTypeBinaryCall(mv, "add");
            case SUB -> emitDTypeBinaryCall(mv, "sub");
            case MUL -> emitDTypeBinaryCall(mv, "mul");
            case DIV -> emitDTypeBinaryCall(mv, "div");
            case MIN -> emitDTypeBinaryCall(mv, "min");
            case MAX -> emitDTypeBinaryCall(mv, "max");
            case NEG -> emitDTypeUnaryCall(mv, "neg");
            case INV -> {
                mv.visitVarInsn(DSTORE, sm.get(SlotKey.TMP_REGISTER));
                mv.visitLdcInsn(1.0d);
                mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
                mv.visitInsn(DDIV);
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
            }
            case LOG -> emitDTypeUnaryCall(mv, "log");
            case EXP -> {
                emitPrecisionMode(mv);
                mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
                mv.visitMethodInsn(INVOKEVIRTUAL, "backend/kernels/cpu/fused/FusedExecutionOptions", "useFastExpApprox", "()Z", false);
                mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", "exp", "(DIZ)D", false);
            }
            case FAST_EXP -> emitDTypeUnaryCall(mv, "fastExp");
            case TANH -> {
                emitPrecisionMode(mv);
                mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
                mv.visitMethodInsn(INVOKEVIRTUAL, "backend/kernels/cpu/fused/FusedExecutionOptions", "useFastTanhApprox", "()Z", false);
                mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", "tanh", "(DIZ)D", false);
            }
            case FAST_TANH -> emitDTypeUnaryCall(mv, "fastTanh");
            case POW -> handlePow(mv, node.parameter(), sm);
            case SQRT -> emitDTypeUnaryCall(mv, "sqrt");
            case MUL_SCALAR -> {
                mv.visitLdcInsn(((Number) node.parameter()).doubleValue());
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", "mulScalar", "(DDI)D", false);
            }
            case RELU -> {
                mv.visitInsn(DCONST_0);
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(DD)D", false);
                emitPrecisionMode(mv);
                mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
            }
            case SIGMOID -> emitDTypeUnaryCall(mv, "sigmoid");
            case NOOP -> emitDTypeUnaryCall(mv, "noop");
            default -> throw new UnsupportedOperationException("Operation " + node.opType() + " is not supported for fused F16 execution.");
        }
    }

    private static void loadRef(
            MethodVisitor mv,
            int ref,
            List<FusedExternalInputPlan> inputPlans,
            int[] nodeValueSlots,
            List<Integer> inputSlots,
            List<Integer> cursorSlots,
            SlotManager sm
    ) {
        if (ref < inputPlans.size()) {
            mv.visitVarInsn(ALOAD, inputSlots.get(ref));
            FusedExternalInputPlan input = inputPlans.get(ref);
            if (input.directIndex()) {
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
            } else {
                mv.visitVarInsn(ALOAD, cursorSlots.get(ref));
                mv.visitMethodInsn(INVOKEVIRTUAL, "graph/codegen/FusedBroadcastCursor", "idx", "()I", false);
            }
            emitPrecisionMode(mv);
            mv.visitMethodInsn(
                    INVOKESTATIC,
                    "graph/codegen/FusedStorageOps",
                    "loadScalar",
                    "(Ltensor/Tensor;II)D",
                    false
            );
            return;
        }
        int nodeIndex = ref - inputPlans.size();
        mv.visitVarInsn(DLOAD, nodeValueSlots[nodeIndex]);
    }

    private static void handlePow(MethodVisitor mv, Object parameter, SlotManager sm) {
        double exponent = ((Number) parameter).doubleValue();
        if (Double.compare(exponent, 0.0d) == 0) {
            mv.visitInsn(POP2);
            mv.visitInsn(DCONST_1);
            emitPrecisionMode(mv);
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
            return;
        }
        if (Double.compare(exponent, 1.0d) == 0) {
            emitPrecisionMode(mv);
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
            return;
        }
        if (Double.compare(exponent, -1.0d) == 0) {
            mv.visitVarInsn(DSTORE, sm.get(SlotKey.TMP_REGISTER));
            mv.visitLdcInsn(1.0d);
            mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
            mv.visitInsn(DDIV);
            emitPrecisionMode(mv);
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
            return;
        }
        if (Double.compare(exponent, 2.0d) == 0) {
            mv.visitVarInsn(DSTORE, sm.get(SlotKey.TMP_REGISTER));
            mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
            mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
            mv.visitInsn(DMUL);
            emitPrecisionMode(mv);
            mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
            return;
        }
        mv.visitLdcInsn(exponent);
        mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "pow", "(DD)D", false);
        emitPrecisionMode(mv);
        mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", "cast", "(DI)D", false);
    }

    private static void emitDTypeUnaryCall(MethodVisitor mv, String method) {
        emitPrecisionMode(mv);
        mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", method, "(DI)D", false);
    }

    private static void emitDTypeBinaryCall(MethodVisitor mv, String method) {
        emitPrecisionMode(mv);
        mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedDTypeOps", method, "(DDI)D", false);
    }

    private static SlotManager buildScalarSlotLayout(int externalInputCount, int nodeCount) {
        SlotManager sm = new SlotManager();
        sm.define(SlotKey.CLUSTER_TENSOR_INPUTS);
        sm.define(SlotKey.CLUSTER_TENSOR);
        sm.define(SlotKey.RANGE_START);
        sm.define(SlotKey.RANGE_END);
        sm.define(SlotKey.FUSED_OPTIONS);
        sm.define(SlotKey.LOOP_COUNTER);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.FUSED_NODE_VALUES, nodeCount);
        sm.define(SlotKey.TMP_REGISTER);
        return sm;
    }

    private static void emitPrecisionMode(MethodVisitor mv) {
        mv.visitLdcInsn(FusedDTypeOps.MODE_F16);
    }

    private static void emitIntArrayConstant(MethodVisitor mv, int[] values) {
        mv.visitLdcInsn(values.length);
        mv.visitIntInsn(NEWARRAY, T_INT);
        for (int i = 0; i < values.length; i++) {
            mv.visitInsn(DUP);
            mv.visitLdcInsn(i);
            mv.visitLdcInsn(values[i]);
            mv.visitInsn(IASTORE);
        }
    }
}
