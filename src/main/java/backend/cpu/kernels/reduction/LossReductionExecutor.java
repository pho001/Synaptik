package backend.cpu.kernels.reduction;

import backend.cpu.storage.CpuStorageView;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;
import backend.cpu.execution.CpuKernelContext;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class LossReductionExecutor {
    private LossReductionExecutor() {}

    static void executeF64(LossReduction reduction, CpuStorageView a, CpuStorageView b, CpuStorageView out, int classDimension, CpuKernelContext context) {
        validate(reduction, a, b, out, context, DataType.FLOAT64);
        int[] shape = a.shape();
        int[] aStrides = a.strides();
        int[] bStrides = b.strides();
        LossReductionTraversal.validateShapes(shape, b.shape(), out.shape(), classDimension, label(reduction));
        double loss;
        if (a.isArray() && b.isArray()) {
            double[] aData = a.requireF64Array();
            double[] bData = b.requireF64Array();
            loss = LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeF64(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        } else if (a.isMemorySegment() && b.isMemorySegment()) {
            MemorySegment aData = a.requireSegment();
            MemorySegment bData = b.requireSegment();
            loss = LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeF64(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        } else if (a.isArray()) {
            double[] aData = a.requireF64Array();
            MemorySegment bData = b.requireSegment();
            loss = LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeF64(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        } else {
            MemorySegment aData = a.requireSegment();
            double[] bData = b.requireF64Array();
            loss = LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeF64(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        }
        writeF64(out, out.storageOffset(), loss);
    }

    static void executeF32(LossReduction reduction, CpuStorageView a, CpuStorageView b, CpuStorageView out, int classDimension, CpuKernelContext context) {
        validate(reduction, a, b, out, context, DataType.FLOAT32);
        int[] shape = a.shape();
        int[] aStrides = a.strides();
        int[] bStrides = b.strides();
        LossReductionTraversal.validateShapes(shape, b.shape(), out.shape(), classDimension, label(reduction));
        double loss;
        if (a.isArray() && b.isArray()) {
            float[] aData = a.requireF32Array();
            float[] bData = b.requireF32Array();
            loss = LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeF32(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        } else if (a.isMemorySegment() && b.isMemorySegment()) {
            MemorySegment aData = a.requireSegment();
            MemorySegment bData = b.requireSegment();
            loss = LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeF32(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        } else if (a.isArray()) {
            float[] aData = a.requireF32Array();
            MemorySegment bData = b.requireSegment();
            loss = LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeF32(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        } else {
            MemorySegment aData = a.requireSegment();
            float[] bData = b.requireF32Array();
            loss = LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeF32(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        }
        writeF32(out, out.storageOffset(), (float) loss);
    }

    static void executeBF16(LossReduction reduction, CpuStorageView a, CpuStorageView b, CpuStorageView out, int classDimension, CpuKernelContext context) {
        validate(reduction, a, b, out, context, DataType.BFLOAT16);
        int[] shape = a.shape();
        int[] aStrides = a.strides();
        int[] bStrides = b.strides();
        LossReductionTraversal.validateShapes(shape, b.shape(), out.shape(), classDimension, label(reduction));
        float loss;
        if (a.isArray() && b.isArray()) {
            short[] aData = a.requireBF16Array();
            short[] bData = b.requireBF16Array();
            loss = (float) LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        } else if (a.isMemorySegment() && b.isMemorySegment()) {
            MemorySegment aData = a.requireSegment();
            MemorySegment bData = b.requireSegment();
            loss = (float) LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        } else if (a.isArray()) {
            short[] aData = a.requireBF16Array();
            MemorySegment bData = b.requireSegment();
            loss = (float) LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        } else {
            MemorySegment aData = a.requireSegment();
            short[] bData = b.requireBF16Array();
            loss = (float) LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, a.storageOffset(), bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        }
        writeBF16(out, out.storageOffset(), TensorDTypeOps.toBFloat16Bits(loss));
    }

    static void executeF32ToBF16(LossReduction reduction, CpuStorageView a, float[] aData, CpuStorageView b, CpuStorageView out, int classDimension, CpuKernelContext context) {
        validate(reduction, a, b, out, context, DataType.BFLOAT16);
        int[] shape = a.shape();
        int[] aStrides = a.strides();
        int[] bStrides = b.strides();
        LossReductionTraversal.validateShapes(shape, b.shape(), out.shape(), classDimension, label(reduction));
        if (aData == null) {
            throw new IllegalArgumentException("Loss F32 continuation requires float logits/logProbs and BF16 targets");
        }
        float loss;
        if (b.isArray()) {
            short[] bData = b.requireBF16Array();
            loss = (float) LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, 0, bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeF32ToBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        } else {
            MemorySegment bData = b.requireSegment();
            loss = (float) LossReductionTraversal.reduceMeanLoss(
                    shape, aStrides, 0, bStrides, b.storageOffset(), classDimension, context,
                    (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                            reduction.computeF32ToBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
            );
        }
        writeBF16(out, out.storageOffset(), TensorDTypeOps.toBFloat16Bits(loss));
    }

    private static void validate(LossReduction reduction, CpuStorageView a, CpuStorageView b, CpuStorageView out, CpuKernelContext context, DataType dtype) {
        if (reduction == null || a == null || b == null || out == null || context == null) {
            throw new IllegalArgumentException("loss reduction execution arguments cannot be null");
        }
        if (a.dtype() != dtype || b.dtype() != dtype || out.dtype() != dtype) {
            throw new IllegalArgumentException(label(reduction) + " requires " + dtype
                    + " storage views, logits/logProbs=" + a.dtype() + ", targets=" + b.dtype()
                    + ", output=" + out.dtype());
        }
    }

    private static String label(LossReduction reduction) {
        return reduction == LossReduction.NLL ? "NLL loss" : "Cross entropy";
    }

    private static void writeF64(CpuStorageView view, int offset, double value) {
        if (view.isArray()) {
            view.requireF64Array()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
        }
    }

    private static void writeF32(CpuStorageView view, int offset, float value) {
        if (view.isArray()) {
            view.requireF32Array()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
        }
    }

    private static void writeBF16(CpuStorageView view, int offset, short value) {
        if (view.isArray()) {
            view.requireBF16Array()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_SHORT, (long) offset * Short.BYTES, value);
        }
    }
}
