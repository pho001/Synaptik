package backend.cpu.fused.codegen;

import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import utils.SlotKey;
import utils.SlotManager;

import static org.objectweb.asm.Opcodes.*;

/**
 * Internal ASM emitter for scalar expression evaluation in generated fused kernels.
 */
public final class FusedScalarExpressionEmitter {
    private FusedScalarExpressionEmitter() {}

    public static void emitNodeEvaluationBytecode(
            MethodVisitor mv,
            FusedExpressionPlan plan,
            FusedNodePlan current,
            int[] nodeValueSlots,
            int[] nodeBoolSlots,
            SlotManager sm,
            int precisionMode,
            java.util.List<FusedExternalInputPlan> inputAccess
    ) {
        if (current.opType() == operations.Operation.OpType.WHERE) {
            emitWhereNode(mv, plan, current, nodeValueSlots, nodeBoolSlots, sm, precisionMode, inputAccess);
            return;
        }

        for (int ref : current.inputRefs()) {
            FusedAsmSupport.loadScalarRef(
                    mv, ref, plan, nodeValueSlots, nodeBoolSlots, sm, precisionMode, inputAccess
            );
        }

        operations.Operation.OpType opType = current.opType();
        switch (opType) {
            case ADD:
                mv.visitInsn(precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32 ? FADD : DADD);
                break;
            case SUB:
                mv.visitInsn(precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32 ? FSUB : DSUB);
                break;
            case MUL:
                mv.visitInsn(precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32 ? FMUL : DMUL);
                break;
            case DIV:
                mv.visitInsn(precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32 ? FDIV : DDIV);
                break;
            case MIN:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "min", "(FF)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "min", "(DD)D", false);
                }
                break;
            case MAX:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(FF)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(DD)D", false);
                }
                break;
            case NEG:
                mv.visitInsn(precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32 ? FNEG : DNEG);
                break;
            case INV:
                FusedAsmSupport.emitScalarStoreInsn(mv, sm.get(SlotKey.TMP_REGISTER), precisionMode);
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn(1.0f);
                } else {
                    mv.visitLdcInsn(1.0d);
                }
                FusedAsmSupport.emitScalarLoadInsn(mv, sm.get(SlotKey.TMP_REGISTER), precisionMode);
                mv.visitInsn(precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32 ? FDIV : DDIV);
                break;
            case LOG:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/codegen/FusedScalarOps", "logF32", "(F)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "log", "(D)D", false);
                }
                break;
            case EXP:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
                    mv.visitMethodInsn(INVOKEVIRTUAL, "backend/cpu/kernels/fused/FusedExecutionOptions", "useFastExpApprox", "()Z", false);
                    mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/codegen/FusedScalarOps", "expF32", "(FZ)F", false);
                } else {
                    mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
                    mv.visitMethodInsn(INVOKEVIRTUAL, "backend/cpu/kernels/fused/FusedExecutionOptions", "useFastExpApprox", "()Z", false);
                    mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/codegen/FusedScalarOps", "expF64", "(DZ)D", false);
                }
                break;
            case FAST_EXP:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/codegen/FusedScalarOps", "fastExpF32", "(F)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/codegen/FusedScalarOps", "fastExpF64", "(D)D", false);
                }
                break;
            case TANH:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
                    mv.visitMethodInsn(INVOKEVIRTUAL, "backend/cpu/kernels/fused/FusedExecutionOptions", "useFastTanhApprox", "()Z", false);
                    mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/codegen/FusedScalarOps", "tanhF32", "(FZ)F", false);
                } else {
                    mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
                    mv.visitMethodInsn(INVOKEVIRTUAL, "backend/cpu/kernels/fused/FusedExecutionOptions", "useFastTanhApprox", "()Z", false);
                    mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/codegen/FusedScalarOps", "tanhF64", "(DZ)D", false);
                }
                break;
            case FAST_TANH:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/codegen/FusedScalarOps", "fastTanhF32", "(F)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/codegen/FusedScalarOps", "fastTanhF64", "(D)D", false);
                }
                break;
            case POW:
                FusedAsmSupport.handlePow(mv, ((ScalarDoubleAttribute) current.attributes()).value(), sm, precisionMode);
                break;
            case POW_TENSOR:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/codegen/FusedScalarOps", "powF32", "(FF)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/codegen/FusedScalarOps", "powF64", "(DD)D", false);
                }
                break;
            case SQRT:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitInsn(F2D);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                    mv.visitInsn(D2F);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                }
                break;
            case ABS:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "abs", "(F)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "abs", "(D)D", false);
                }
                break;
            case CONST_SCALAR:
                double constant = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) constant);
                } else {
                    mv.visitLdcInsn(constant);
                }
                break;
            case MUL_SCALAR:
                double scalar = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) scalar);
                    mv.visitInsn(FMUL);
                } else {
                    mv.visitLdcInsn(scalar);
                    mv.visitInsn(DMUL);
                }
                break;
            case RELU:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitInsn(FCONST_0);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(FF)F", false);
                } else {
                    mv.visitInsn(DCONST_0);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(DD)D", false);
                }
                break;
            case CLAMP_MIN: {
                double minValue = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) minValue);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(FF)F", false);
                } else {
                    mv.visitLdcInsn(minValue);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(DD)D", false);
                }
                break;
            }
            case CLAMP_MAX: {
                double maxValue = ((ScalarDoubleAttribute) current.attributes()).value();
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) maxValue);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "min", "(FF)F", false);
                } else {
                    mv.visitLdcInsn(maxValue);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "min", "(DD)D", false);
                }
                break;
            }
            case SIGMOID:
                if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitInsn(FNEG);
                    mv.visitInsn(F2D);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "exp", "(D)D", false);
                    mv.visitLdcInsn(1.0d);
                    mv.visitInsn(DADD);
                    mv.visitVarInsn(DSTORE, sm.get(SlotKey.TMP_REGISTER));
                    mv.visitLdcInsn(1.0d);
                    mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
                    mv.visitInsn(DDIV);
                    mv.visitInsn(D2F);
                } else {
                    mv.visitInsn(DNEG);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "exp", "(D)D", false);
                    mv.visitLdcInsn(1.0d);
                    mv.visitInsn(DADD);
                    mv.visitVarInsn(DSTORE, sm.get(SlotKey.TMP_REGISTER));
                    mv.visitLdcInsn(1.0d);
                    mv.visitVarInsn(DLOAD, sm.get(SlotKey.TMP_REGISTER));
                    mv.visitInsn(DDIV);
                }
                break;
            case NOOP:
                break;
            case GT:
            case GE:
            case LT:
            case LE:
            case EQ:
            case NE:
                emitCompareNode(mv, opType, precisionMode);
                break;
            case LOGICAL_AND:
                mv.visitInsn(IAND);
                break;
            case LOGICAL_OR:
                mv.visitInsn(IOR);
                break;
            case LOGICAL_NOT:
                mv.visitInsn(ICONST_1);
                mv.visitInsn(IXOR);
                break;
            default:
                throw new UnsupportedOperationException("Operation " + opType + " is not supported for fusing.");
        }
    }

    private static void emitWhereNode(
            MethodVisitor mv,
            FusedExpressionPlan plan,
            FusedNodePlan current,
            int[] nodeValueSlots,
            int[] nodeBoolSlots,
            SlotManager sm,
            int precisionMode,
            java.util.List<FusedExternalInputPlan> inputAccess
    ) {
        java.util.List<Integer> refs = current.inputRefs();
        if (refs.size() != 3) {
            throw new IllegalArgumentException("WHERE fused node expects exactly 3 inputs");
        }
        int condRef = refs.get(0);
        int trueRef = refs.get(1);
        int falseRef = refs.get(2);

        loadBoolScalarRef(mv, condRef, plan, nodeBoolSlots, sm, inputAccess);
        Label falseBranch = new Label();
        Label done = new Label();
        mv.visitJumpInsn(IFEQ, falseBranch);
        FusedAsmSupport.loadScalarRef(mv, trueRef, plan, nodeValueSlots, nodeBoolSlots, sm, precisionMode, inputAccess);
        mv.visitJumpInsn(GOTO, done);
        mv.visitLabel(falseBranch);
        FusedAsmSupport.loadScalarRef(mv, falseRef, plan, nodeValueSlots, nodeBoolSlots, sm, precisionMode, inputAccess);
        mv.visitLabel(done);
    }

    private static void loadExternalBoolScalarRef(
            MethodVisitor mv,
            int ref,
            SlotManager sm,
            java.util.List<FusedExternalInputPlan> inputPlans
    ) {
        int inputSlot = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS).get(ref);
        mv.visitVarInsn(ALOAD, inputSlot);
        FusedExternalInputPlan meta = inputPlans.get(ref);
        if (meta.isLinearAccess()) {
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
            if (meta.storageOffset() != 0) {
                mv.visitLdcInsn(meta.storageOffset());
                mv.visitInsn(IADD);
            }
        } else {
            mv.visitVarInsn(ALOAD, sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS).get(ref));
            mv.visitMethodInsn(INVOKEVIRTUAL, "backend/cpu/fused/codegen/FusedBroadcastCursor", "idx", "()I", false);
        }
        mv.visitInsn(BALOAD);
    }

    private static void loadBoolScalarRef(
            MethodVisitor mv,
            int ref,
            FusedExpressionPlan plan,
            int[] nodeBoolSlots,
            SlotManager sm,
            java.util.List<FusedExternalInputPlan> inputPlans
    ) {
        if (ref < inputPlans.size()) {
            FusedExternalInputPlan condMeta = inputPlans.get(ref);
            if (condMeta.dataType() != tensor.DataType.BOOL) {
                throw new UnsupportedOperationException("Fused bool ref must use BOOL external input.");
            }
            loadExternalBoolScalarRef(mv, ref, sm, inputPlans);
            return;
        }
        int nodeIndex = ref - inputPlans.size();
        if (nodeIndex < 0 || nodeIndex >= plan.nodeCount()) {
            throw new IllegalArgumentException("Invalid fused bool scalar ref " + ref);
        }
        if (plan.nodes().get(nodeIndex).outputType() != tensor.DataType.BOOL) {
            throw new IllegalArgumentException("Requested bool scalar ref for non-BOOL fused node " + ref);
        }
        FusedAsmSupport.emitBoolScalarLoadInsn(mv, nodeBoolSlots[nodeIndex]);
    }

    private static void emitCompareNode(
            MethodVisitor mv,
            operations.Operation.OpType opType,
            int precisionMode
    ) {
        Label trueLabel = new Label();
        Label endLabel = new Label();
        if (precisionMode == backend.cpu.fused.codegen.FusedDTypeOps.MODE_F32) {
            mv.visitInsn(FCMPL);
        } else {
            mv.visitInsn(DCMPL);
        }
        switch (opType) {
            case GT -> mv.visitJumpInsn(IFGT, trueLabel);
            case GE -> mv.visitJumpInsn(IFGE, trueLabel);
            case LT -> mv.visitJumpInsn(IFLT, trueLabel);
            case LE -> mv.visitJumpInsn(IFLE, trueLabel);
            case EQ -> mv.visitJumpInsn(IFEQ, trueLabel);
            case NE -> mv.visitJumpInsn(IFNE, trueLabel);
            default -> throw new IllegalArgumentException("Unsupported compare op " + opType);
        }
        mv.visitInsn(ICONST_0);
        mv.visitJumpInsn(GOTO, endLabel);
        mv.visitLabel(trueLabel);
        mv.visitInsn(ICONST_1);
        mv.visitLabel(endLabel);
    }
}
