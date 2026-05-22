package backend.cpu.fused.asm.emit;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.runtime.FusedDTypeOps;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import utils.SlotKey;
import utils.SlotManager;

import java.util.List;

import static org.objectweb.asm.Opcodes.*;

final class FusedScalarBytecode {
    private FusedScalarBytecode() {}

    static void emitScalarLoadInsn(MethodVisitor mv, int slot, int precisionMode) {
        mv.visitVarInsn(precisionMode == FusedDTypeOps.MODE_F32 ? FLOAD : DLOAD, slot);
    }

    static void emitBoolScalarLoadInsn(MethodVisitor mv, int slot) {
        mv.visitVarInsn(ILOAD, slot);
    }

    static void emitScalarStoreInsn(MethodVisitor mv, int slot, int precisionMode) {
        mv.visitVarInsn(precisionMode == FusedDTypeOps.MODE_F32 ? FSTORE : DSTORE, slot);
    }

    static void emitBoolScalarStoreInsn(MethodVisitor mv, int slot) {
        mv.visitVarInsn(ISTORE, slot);
    }

    static void emitScalarArrayLoadInsn(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitInsn(FALOAD);
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitInsn(DALOAD);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "loadScalarBF16Array", "([SI)D", false);
        }
    }

    static void emitScalarArrayStoreInsn(MethodVisitor mv, int precisionMode) {
        if (precisionMode == FusedDTypeOps.MODE_F32) {
            mv.visitInsn(FASTORE);
        } else if (precisionMode == FusedDTypeOps.MODE_F64) {
            mv.visitInsn(DASTORE);
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedStorageOps", "storeScalarBF16Array", "([SID)V", false);
        }
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

    static void handlePow(MethodVisitor mv, double exponentValue, SlotManager sm, int precisionMode) {
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
            mv.visitMethodInsn(INVOKESTATIC, "backend/cpu/fused/runtime/FusedScalarOps", "powF32", "(FF)F", false);
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
            FusedExpressionPlan plan,
            int[] nodeValueSlots,
            int[] nodeBoolSlots,
            SlotManager sm,
            int precisionMode,
            java.util.List<FusedExternalInputPlan> inputPlans
    ) {
        if (ref < inputPlans.size()) {
            FusedExternalInputPlan meta = inputPlans.get(ref);
            Label done = null;
            if (meta.dataType() == tensor.DataType.BFLOAT16) {
                int continuationSlot = sm.getGroup(SlotKey.CLUSTER_INPUTS_CONTINUATION_ARRAYS).get(ref);
                Label storageLoad = new Label();
                done = new Label();
                mv.visitVarInsn(ALOAD, continuationSlot);
                mv.visitJumpInsn(IFNULL, storageLoad);
                mv.visitVarInsn(ALOAD, continuationSlot);
                if (meta.isLinearAccess()) {
                    mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
                    if (meta.storageOffset() != 0) {
                        mv.visitLdcInsn(meta.storageOffset());
                        mv.visitInsn(IADD);
                    }
                } else {
                    mv.visitVarInsn(ALOAD, sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS).get(ref));
                    mv.visitMethodInsn(INVOKEVIRTUAL, "backend/cpu/fused/runtime/FusedBroadcastCursor", "idx", "()I", false);
                }
                mv.visitInsn(FALOAD);
                if (precisionMode != FusedDTypeOps.MODE_F32) {
                    mv.visitInsn(F2D);
                }
                mv.visitJumpInsn(GOTO, done);
                mv.visitLabel(storageLoad);
            }
            int inputSlot = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS).get(ref);
            mv.visitVarInsn(ALOAD, inputSlot);
            if (meta.isLinearAccess()) {
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
                if (meta.storageOffset() != 0) {
                    mv.visitLdcInsn(meta.storageOffset());
                    mv.visitInsn(IADD);
                }
            } else {
                mv.visitVarInsn(ALOAD, sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS).get(ref));
                mv.visitMethodInsn(INVOKEVIRTUAL, "backend/cpu/fused/runtime/FusedBroadcastCursor", "idx", "()I", false);
            }
            emitScalarArrayLoadInsn(mv, precisionMode);
            if (done != null) {
                mv.visitLabel(done);
            }
            return;
        }

        int nodeIndex = ref - inputPlans.size();
        if (nodeIndex < 0 || nodeIndex >= plan.nodeCount()) {
            throw new IllegalArgumentException("Invalid fused scalar ref " + ref);
        }
        if (plan.nodes().get(nodeIndex).outputType() == tensor.DataType.BOOL) {
            emitBoolScalarLoadInsn(mv, nodeBoolSlots[nodeIndex]);
            return;
        }
        emitScalarLoadInsn(mv, nodeValueSlots[nodeIndex], precisionMode);
    }
}
