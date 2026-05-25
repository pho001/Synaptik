package backend.cpu.kernels.reduction;

import tensor.TensorInternalAccess;

import backend.cpu.execution.CpuKernelContext;
import tensor.Tensor;

final class SoftmaxLikeExecutor {
    private SoftmaxLikeExecutor() {}

    static void executeF64(SoftmaxLikeReduction reduction, Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validate(reduction, input, node, context);
        SoftmaxLikeTraversal.validateShapes(input.getShapeUnsafe(), node.getShapeUnsafe(), dimension, label(reduction));
        double[] in = TensorInternalAccess.float64Data(input);
        double[] out = TensorInternalAccess.float64Data(node);
        SoftmaxLikeTraversal.runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, context,
                (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                        reduction.computeF64(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
    }

    static void executeF32(SoftmaxLikeReduction reduction, Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validate(reduction, input, node, context);
        SoftmaxLikeTraversal.validateShapes(input.getShapeUnsafe(), node.getShapeUnsafe(), dimension, label(reduction));
        float[] in = TensorInternalAccess.float32Data(input);
        float[] out = TensorInternalAccess.float32Data(node);
        SoftmaxLikeTraversal.runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, context,
                (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                        reduction.computeF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
    }

    static void executeBF16(SoftmaxLikeReduction reduction, Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validate(reduction, input, node, context);
        SoftmaxLikeTraversal.validateShapes(input.getShapeUnsafe(), node.getShapeUnsafe(), dimension, label(reduction));
        short[] in = TensorInternalAccess.bfloat16Data(input);
        short[] out = TensorInternalAccess.bfloat16Data(node);
        SoftmaxLikeTraversal.runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, context,
                (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                        reduction.computeBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
    }

    static void executeF32ToBF16(SoftmaxLikeReduction reduction, Tensor input, float[] in, Tensor node, int dimension, CpuKernelContext context) {
        validate(reduction, input, node, context);
        SoftmaxLikeTraversal.validateShapes(input.getShapeUnsafe(), node.getShapeUnsafe(), dimension, label(reduction));
        short[] out = TensorInternalAccess.bfloat16Data(node);
        if (in == null) {
            throw new IllegalArgumentException("Float continuation input cannot be null");
        }
        SoftmaxLikeTraversal.runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), 0, node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, context,
                (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                        reduction.computeF32ToBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
    }

    static void executeF32ToFloat(SoftmaxLikeReduction reduction, Tensor input, float[] in, float[] out, int dimension, CpuKernelContext context) {
        validate(reduction, input, input, context);
        SoftmaxLikeTraversal.validateShapes(input.getShapeUnsafe(), input.getShapeUnsafe(), dimension, label(reduction));
        if (in == null || out == null) {
            throw new IllegalArgumentException("Float continuation buffers cannot be null");
        }
        SoftmaxLikeTraversal.runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), 0, input.getStridesUnsafe(), 0, dimension, context,
                (baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize) ->
                        reduction.computeF32ToFloat(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize));
    }

    private static void validate(SoftmaxLikeReduction reduction, Tensor input, Tensor node, CpuKernelContext context) {
        if (reduction == null || input == null || node == null || context == null) {
            throw new IllegalArgumentException("softmax-like execution arguments cannot be null");
        }
    }

    private static String label(SoftmaxLikeReduction reduction) {
        return reduction == SoftmaxLikeReduction.SOFTMAX ? "Softmax" : "LogSoftmax";
    }
}
