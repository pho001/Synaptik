package backend.cpu.fused.asm.emit;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.numeric.FusedNumericContract;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import tensor.DataType;
import utils.SlotKey;
import utils.SlotManager;

import java.util.List;

import static org.objectweb.asm.Opcodes.*;

final class FusedScalarBytecode {
    private static final String TENSOR_DTYPE_OPS = "tensor/dtype/TensorDTypeOps";

    private FusedScalarBytecode() {}

    static void emitScalarLoadInsn(MethodVisitor mv, int slot, FusedNumericContract numericContract) {
        mv.visitVarInsn(numericContract.usesFloatCompute() ? FLOAD : DLOAD, slot);
    }

    static void emitBoolScalarLoadInsn(MethodVisitor mv, int slot) {
        mv.visitVarInsn(ILOAD, slot);
    }

    static void emitScalarStoreInsn(MethodVisitor mv, int slot, FusedNumericContract numericContract) {
        mv.visitVarInsn(numericContract.usesFloatCompute() ? FSTORE : DSTORE, slot);
    }

    static void emitBoolScalarStoreInsn(MethodVisitor mv, int slot) {
        mv.visitVarInsn(ISTORE, slot);
    }

    static void emitScalarArrayLoadInsn(MethodVisitor mv, DataType dataType, FusedNumericContract numericContract) {
        switch (dataType) {
            case FLOAT32 -> {
                mv.visitInsn(FALOAD);
                if (numericContract.usesDoubleCompute()) {
                    mv.visitInsn(F2D);
                }
            }
            case FLOAT64 -> {
                mv.visitInsn(DALOAD);
                if (numericContract.usesFloatCompute()) {
                    mv.visitInsn(D2F);
                }
            }
            case BFLOAT16 -> {
                mv.visitInsn(SALOAD);
                mv.visitMethodInsn(INVOKESTATIC, TENSOR_DTYPE_OPS, "fromBFloat16Bits", "(S)F", false);
                if (numericContract.usesDoubleCompute()) {
                    mv.visitInsn(F2D);
                }
            }
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException(
                    dataType + " scalar array load is not supported for fused numeric values."
            );
        }
    }

    static void emitScalarArrayStoreInsn(MethodVisitor mv, DataType dataType, FusedNumericContract numericContract) {
        switch (dataType) {
            case FLOAT32 -> {
                if (numericContract.usesDoubleCompute()) {
                    mv.visitInsn(D2F);
                }
                mv.visitInsn(FASTORE);
            }
            case FLOAT64 -> {
                if (numericContract.usesFloatCompute()) {
                    mv.visitInsn(F2D);
                }
                mv.visitInsn(DASTORE);
            }
            case BFLOAT16 -> {
                if (numericContract.usesDoubleCompute()) {
                    mv.visitInsn(D2F);
                }
                mv.visitMethodInsn(INVOKESTATIC, TENSOR_DTYPE_OPS, "toBFloat16Bits", "(F)S", false);
                mv.visitInsn(SASTORE);
            }
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException(
                    dataType + " scalar array store is not supported for fused numeric values."
            );
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

    static void handlePow(MethodVisitor mv, double exponentValue, SlotManager sm, FusedNumericContract numericContract) {
        if (numericContract.usesFloatCompute()) {
            float exponent = (float) exponentValue;

            if (Float.compare(exponent, 0.0f) == 0) {
                mv.visitInsn(POP);
                mv.visitInsn(FCONST_1);
                return;
            }
            if (Float.compare(exponent, 1.0f) == 0) {
                return;
            }
            if (Float.compare(exponent, -2.0f) == 0) {
                mv.visitInsn(DUP);
                mv.visitInsn(FMUL);
                mv.visitInsn(FCONST_1);
                mv.visitInsn(SWAP);
                mv.visitInsn(FDIV);
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

            mv.visitInsn(F2D);
            mv.visitLdcInsn((double) exponent);
            mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "pow", "(DD)D", false);
            mv.visitInsn(D2F);
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
        if (Double.compare(exponent, -2.0d) == 0) {
            mv.visitInsn(DUP2);
            mv.visitInsn(DMUL);
            emitScalarStoreInsn(mv, sm.get(SlotKey.TMP_REGISTER), numericContract);
            mv.visitInsn(DCONST_1);
            emitScalarLoadInsn(mv, sm.get(SlotKey.TMP_REGISTER), numericContract);
            mv.visitInsn(DDIV);
            return;
        }
        if (Double.compare(exponent, -1.0d) == 0) {
            emitScalarStoreInsn(mv, sm.get(SlotKey.TMP_REGISTER), numericContract);
            mv.visitInsn(DCONST_1);
            emitScalarLoadInsn(mv, sm.get(SlotKey.TMP_REGISTER), numericContract);
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

    static void handlePowTensor(MethodVisitor mv, SlotManager sm, FusedNumericContract numericContract) {
        if (numericContract.usesFloatCompute()) {
            int exponentSlot = sm.get(SlotKey.TMP_REGISTER);
            mv.visitVarInsn(FSTORE, exponentSlot);
            mv.visitInsn(F2D);
            mv.visitVarInsn(FLOAD, exponentSlot);
            mv.visitInsn(F2D);
            mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "pow", "(DD)D", false);
            mv.visitInsn(D2F);
            return;
        }
        mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "pow", "(DD)D", false);
    }

    static void loadScalarRef(
            MethodVisitor mv,
            int ref,
            FusedExpressionPlan plan,
            int[] nodeValueSlots,
            int[] nodeBoolSlots,
            SlotManager sm,
            FusedNumericContract numericContract,
            java.util.List<FusedExternalInputPlan> inputPlans,
            boolean memorySegmentStorage
    ) {
        if (ref < inputPlans.size()) {
            FusedExternalInputPlan meta = inputPlans.get(ref);
            if (meta.dataType() == tensor.DataType.BOOL) {
                loadExternalBoolScalar(mv, ref, sm, inputPlans, memorySegmentStorage);
                return;
            }
            Label done = null;
            if (!memorySegmentStorage && meta.dataType() == tensor.DataType.BFLOAT16) {
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
                if (numericContract.usesDoubleCompute()) {
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
            if (memorySegmentStorage) {
                FusedRuntimeCalls.emitLoadScalarFromSegmentCall(mv, meta.dataType(), numericContract);
            } else {
                emitScalarArrayLoadInsn(mv, meta.dataType(), numericContract);
            }
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
        emitScalarLoadInsn(mv, nodeValueSlots[nodeIndex], numericContract);
    }

    static void loadExternalBoolScalar(
            MethodVisitor mv,
            int ref,
            SlotManager sm,
            List<FusedExternalInputPlan> inputPlans,
            boolean memorySegmentStorage
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
            mv.visitMethodInsn(INVOKEVIRTUAL, "backend/cpu/fused/runtime/FusedBroadcastCursor", "idx", "()I", false);
        }
        if (memorySegmentStorage) {
            FusedRuntimeCalls.emitLoadBoolFromSegmentCall(mv);
        } else {
            mv.visitInsn(BALOAD);
        }
    }
}
