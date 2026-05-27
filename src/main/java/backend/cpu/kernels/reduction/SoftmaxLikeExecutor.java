package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.storage.CpuStorageView;
import tensor.DataType;

import java.lang.foreign.MemorySegment;

final class SoftmaxLikeExecutor {
    private SoftmaxLikeExecutor() {}

    static void executeF64(SoftmaxLikeReduction reduction, CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        validate(reduction, input, output, context, DataType.FLOAT64);
        int[] shape = input.shape();
        int[] inStrides = input.strides();
        int[] outStrides = output.strides();
        SoftmaxLikeTraversal.validateShapes(shape, output.shape(), dimension, label(reduction));
        if (input.isArray() && output.isArray()) {
            double[] in = input.requireF64Array();
            double[] out = output.requireF64Array();
            SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                    (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                            reduction.computeF64(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
            return;
        }
        if (input.isMemorySegment() && output.isMemorySegment()) {
            MemorySegment in = input.requireSegment();
            MemorySegment out = output.requireSegment();
            SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                    (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                            reduction.computeF64(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
            return;
        }
        if (input.isArray()) {
            double[] in = input.requireF64Array();
            MemorySegment out = output.requireSegment();
            SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                    (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                            reduction.computeF64(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
            return;
        }
        MemorySegment in = input.requireSegment();
        double[] out = output.requireF64Array();
        SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                        reduction.computeF64(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
    }

    static void executeF32(SoftmaxLikeReduction reduction, CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        validate(reduction, input, output, context, DataType.FLOAT32);
        int[] shape = input.shape();
        int[] inStrides = input.strides();
        int[] outStrides = output.strides();
        SoftmaxLikeTraversal.validateShapes(shape, output.shape(), dimension, label(reduction));
        if (input.isArray() && output.isArray()) {
            float[] in = input.requireF32Array();
            float[] out = output.requireF32Array();
            SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                    (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                            reduction.computeF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
            return;
        }
        if (input.isMemorySegment() && output.isMemorySegment()) {
            MemorySegment in = input.requireSegment();
            MemorySegment out = output.requireSegment();
            SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                    (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                            reduction.computeF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
            return;
        }
        if (input.isArray()) {
            float[] in = input.requireF32Array();
            MemorySegment out = output.requireSegment();
            SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                    (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                            reduction.computeF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
            return;
        }
        MemorySegment in = input.requireSegment();
        float[] out = output.requireF32Array();
        SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                        reduction.computeF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
    }

    static void executeBF16(SoftmaxLikeReduction reduction, CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        validate(reduction, input, output, context, DataType.BFLOAT16);
        int[] shape = input.shape();
        int[] inStrides = input.strides();
        int[] outStrides = output.strides();
        SoftmaxLikeTraversal.validateShapes(shape, output.shape(), dimension, label(reduction));
        if (input.isArray() && output.isArray()) {
            short[] in = input.requireBF16Array();
            short[] out = output.requireBF16Array();
            SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                    (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                            reduction.computeBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
            return;
        }
        if (input.isMemorySegment() && output.isMemorySegment()) {
            MemorySegment in = input.requireSegment();
            MemorySegment out = output.requireSegment();
            SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                    (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                            reduction.computeBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
            return;
        }
        if (input.isArray()) {
            short[] in = input.requireBF16Array();
            MemorySegment out = output.requireSegment();
            SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                    (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                            reduction.computeBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
            return;
        }
        MemorySegment in = input.requireSegment();
        short[] out = output.requireBF16Array();
        SoftmaxLikeTraversal.runGroups(shape, inStrides, input.storageOffset(), outStrides, output.storageOffset(), dimension, context,
                (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                        reduction.computeBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
    }

    static void executeF32ToBF16(SoftmaxLikeReduction reduction, CpuStorageView input, float[] in, CpuStorageView output, int dimension, CpuKernelContext context) {
        validateContinuation(reduction, input, output, context, DataType.BFLOAT16);
        int[] shape = input.shape();
        int[] inStrides = input.strides();
        int[] outStrides = output.strides();
        SoftmaxLikeTraversal.validateShapes(shape, output.shape(), dimension, label(reduction));
        if (in == null) {
            throw new IllegalArgumentException("Float continuation input cannot be null");
        }
        if (output.isArray()) {
            short[] out = output.requireBF16Array();
            SoftmaxLikeTraversal.runGroups(shape, inStrides, 0, outStrides, output.storageOffset(), dimension, context,
                    (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                            reduction.computeF32ToBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
            return;
        }
        MemorySegment out = output.requireSegment();
        SoftmaxLikeTraversal.runGroups(shape, inStrides, 0, outStrides, output.storageOffset(), dimension, context,
                (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                        reduction.computeF32ToBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
    }

    static void executeF32ToFloat(SoftmaxLikeReduction reduction, CpuStorageView input, float[] in, float[] out, int dimension, CpuKernelContext context) {
        validate(reduction, input, input, context, input.dtype());
        int[] shape = input.shape();
        int[] strides = input.strides();
        SoftmaxLikeTraversal.validateShapes(shape, shape, dimension, label(reduction));
        if (in == null || out == null) {
            throw new IllegalArgumentException("Float continuation buffers cannot be null");
        }
        SoftmaxLikeTraversal.runGroups(shape, strides, 0, strides, 0, dimension, context,
                (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                        reduction.computeF32ToFloat(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
    }

    private static void validate(SoftmaxLikeReduction reduction, CpuStorageView input, CpuStorageView output, CpuKernelContext context, DataType dtype) {
        if (reduction == null || input == null || output == null || context == null) {
            throw new IllegalArgumentException("softmax-like execution arguments cannot be null");
        }
        if (input.dtype() != dtype || output.dtype() != dtype) {
            throw new IllegalArgumentException(label(reduction) + " requires " + dtype
                    + " storage views, input=" + input.dtype() + ", output=" + output.dtype());
        }
    }

    private static void validateContinuation(SoftmaxLikeReduction reduction, CpuStorageView input, CpuStorageView output, CpuKernelContext context, DataType outputDType) {
        if (reduction == null || input == null || output == null || context == null) {
            throw new IllegalArgumentException("softmax-like execution arguments cannot be null");
        }
        if (input.dtype() != DataType.BFLOAT16 || output.dtype() != outputDType) {
            throw new IllegalArgumentException(label(reduction) + " float continuation requires BF16 input view and "
                    + outputDType + " output view, input=" + input.dtype() + ", output=" + output.dtype());
        }
    }

    private static String label(SoftmaxLikeReduction reduction) {
        return reduction == SoftmaxLikeReduction.SOFTMAX ? "Softmax" : "LogSoftmax";
    }
}
