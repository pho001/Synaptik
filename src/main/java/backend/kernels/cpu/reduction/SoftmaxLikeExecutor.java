package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernelContext;
import tensor.Tensor;

final class SoftmaxLikeExecutor {
    private SoftmaxLikeExecutor() {}

    static void executeF64(SoftmaxLikeReduction reduction, Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validate(reduction, input, node, context);
        SoftmaxLikeTraversal.validateShapes(input.getShapeUnsafe(), node.getShapeUnsafe(), dimension, label(reduction));
        double[] in = input.getFloat64Data();
        double[] out = node.getFloat64Data();
        SoftmaxLikeTraversal.runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, context,
                group -> reduction.computeF64(in, out, group.baseIn(), group.baseOut(), group.axisStrideIn(), group.axisStrideOut(), group.axisSize()));
    }

    static void executeF32(SoftmaxLikeReduction reduction, Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validate(reduction, input, node, context);
        SoftmaxLikeTraversal.validateShapes(input.getShapeUnsafe(), node.getShapeUnsafe(), dimension, label(reduction));
        float[] in = input.getFloat32Data();
        float[] out = node.getFloat32Data();
        SoftmaxLikeTraversal.runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, context,
                group -> reduction.computeF32(in, out, group.baseIn(), group.baseOut(), group.axisStrideIn(), group.axisStrideOut(), group.axisSize()));
    }

    static void executeBF16(SoftmaxLikeReduction reduction, Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validate(reduction, input, node, context);
        SoftmaxLikeTraversal.validateShapes(input.getShapeUnsafe(), node.getShapeUnsafe(), dimension, label(reduction));
        short[] in = input.getBFloat16Data();
        short[] out = node.getBFloat16Data();
        SoftmaxLikeTraversal.runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, context,
                group -> reduction.computeBF16(in, out, group.baseIn(), group.baseOut(), group.axisStrideIn(), group.axisStrideOut(), group.axisSize()));
    }

    static void executeF32ToBF16(SoftmaxLikeReduction reduction, Tensor input, float[] in, Tensor node, int dimension, CpuKernelContext context) {
        validate(reduction, input, node, context);
        SoftmaxLikeTraversal.validateShapes(input.getShapeUnsafe(), node.getShapeUnsafe(), dimension, label(reduction));
        short[] out = node.getBFloat16Data();
        if (in == null) {
            throw new IllegalArgumentException("Float continuation input cannot be null");
        }
        SoftmaxLikeTraversal.runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), 0, node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, context,
                group -> reduction.computeF32ToBF16(in, out, group.baseIn(), group.baseOut(), group.axisStrideIn(), group.axisStrideOut(), group.axisSize()));
    }

    static void executeF32ToFloat(SoftmaxLikeReduction reduction, Tensor input, float[] in, float[] out, int dimension, CpuKernelContext context) {
        validate(reduction, input, input, context);
        SoftmaxLikeTraversal.validateShapes(input.getShapeUnsafe(), input.getShapeUnsafe(), dimension, label(reduction));
        if (in == null || out == null) {
            throw new IllegalArgumentException("Float continuation buffers cannot be null");
        }
        SoftmaxLikeTraversal.runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), 0, input.getStridesUnsafe(), 0, dimension, context,
                group -> reduction.computeF32ToFloat(in, out, group.baseIn(), group.baseOut(), group.axisStrideIn(), group.axisStrideOut(), group.axisSize()));
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
