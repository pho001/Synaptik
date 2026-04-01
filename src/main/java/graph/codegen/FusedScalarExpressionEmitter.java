package graph.codegen;

import org.objectweb.asm.MethodVisitor;
import utils.SlotKey;
import utils.SlotManager;

import static org.objectweb.asm.Opcodes.*;

public final class FusedScalarExpressionEmitter {
    private FusedScalarExpressionEmitter() {}

    public static void emitNodeEvaluationBytecode(
            MethodVisitor mv,
            FusedNodePlan current,
            int[] nodeValueSlots,
            SlotManager sm,
            int precisionMode,
            java.util.List<FusedExternalInputPlan> inputAccess
    ) {
        for (int ref : current.inputRefs()) {
            FusedAsmSupport.loadScalarRef(
                    mv, ref, nodeValueSlots, sm, precisionMode, inputAccess
            );
        }

        operations.Operation.OpType opType = current.opType();
        switch (opType) {
            case ADD:
                mv.visitInsn(precisionMode == graph.codegen.FusedDTypeOps.MODE_F32 ? FADD : DADD);
                break;
            case SUB:
                mv.visitInsn(precisionMode == graph.codegen.FusedDTypeOps.MODE_F32 ? FSUB : DSUB);
                break;
            case MUL:
                mv.visitInsn(precisionMode == graph.codegen.FusedDTypeOps.MODE_F32 ? FMUL : DMUL);
                break;
            case DIV:
                mv.visitInsn(precisionMode == graph.codegen.FusedDTypeOps.MODE_F32 ? FDIV : DDIV);
                break;
            case MIN:
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "min", "(FF)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "min", "(DD)D", false);
                }
                break;
            case MAX:
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(FF)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(DD)D", false);
                }
                break;
            case NEG:
                mv.visitInsn(precisionMode == graph.codegen.FusedDTypeOps.MODE_F32 ? FNEG : DNEG);
                break;
            case INV:
                FusedAsmSupport.emitScalarStoreInsn(mv, sm.get(SlotKey.TMP_REGISTER), precisionMode);
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn(1.0f);
                } else {
                    mv.visitLdcInsn(1.0d);
                }
                FusedAsmSupport.emitScalarLoadInsn(mv, sm.get(SlotKey.TMP_REGISTER), precisionMode);
                mv.visitInsn(precisionMode == graph.codegen.FusedDTypeOps.MODE_F32 ? FDIV : DDIV);
                break;
            case LOG:
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedScalarOps", "logF32", "(F)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "log", "(D)D", false);
                }
                break;
            case EXP:
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
                    mv.visitMethodInsn(INVOKEVIRTUAL, "backend/kernels/cpu/fused/FusedExecutionOptions", "useFastExpApprox", "()Z", false);
                    mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedScalarOps", "expF32", "(FZ)F", false);
                } else {
                    mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
                    mv.visitMethodInsn(INVOKEVIRTUAL, "backend/kernels/cpu/fused/FusedExecutionOptions", "useFastExpApprox", "()Z", false);
                    mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedScalarOps", "expF64", "(DZ)D", false);
                }
                break;
            case FAST_EXP:
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedScalarOps", "fastExpF32", "(F)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedScalarOps", "fastExpF64", "(D)D", false);
                }
                break;
            case TANH:
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
                    mv.visitMethodInsn(INVOKEVIRTUAL, "backend/kernels/cpu/fused/FusedExecutionOptions", "useFastTanhApprox", "()Z", false);
                    mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedScalarOps", "tanhF32", "(FZ)F", false);
                } else {
                    mv.visitVarInsn(ALOAD, sm.get(SlotKey.FUSED_OPTIONS));
                    mv.visitMethodInsn(INVOKEVIRTUAL, "backend/kernels/cpu/fused/FusedExecutionOptions", "useFastTanhApprox", "()Z", false);
                    mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedScalarOps", "tanhF64", "(DZ)D", false);
                }
                break;
            case FAST_TANH:
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedScalarOps", "fastTanhF32", "(F)F", false);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "graph/codegen/FusedScalarOps", "fastTanhF64", "(D)D", false);
                }
                break;
            case POW:
                FusedAsmSupport.handlePow(mv, current.parameter(), sm, precisionMode);
                break;
            case SQRT:
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitInsn(F2D);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                    mv.visitInsn(D2F);
                } else {
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                }
                break;
            case MUL_SCALAR:
                double scalar = ((Number) current.parameter()).doubleValue();
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitLdcInsn((float) scalar);
                    mv.visitInsn(FMUL);
                } else {
                    mv.visitLdcInsn(scalar);
                    mv.visitInsn(DMUL);
                }
                break;
            case RELU:
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
                    mv.visitInsn(FCONST_0);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(FF)F", false);
                } else {
                    mv.visitInsn(DCONST_0);
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "max", "(DD)D", false);
                }
                break;
            case SIGMOID:
                if (precisionMode == graph.codegen.FusedDTypeOps.MODE_F32) {
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
            default:
                throw new UnsupportedOperationException("Operation " + opType + " is not supported for fusing.");
        }
    }
}
