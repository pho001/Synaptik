package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedReductionHints;
import tensor.TensorMetadata;

final class SoftmaxLikeTraversal {
    private SoftmaxLikeTraversal() {}

    static void validateShapes(int[] shape, int[] outShape, int dimension, String label) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
        if (outShape.length != shape.length) {
            throw new IllegalArgumentException(label + " output rank must match input rank");
        }
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != outShape[i]) {
                throw new IllegalArgumentException(label + " output shape must match input shape");
            }
        }
    }

    static void runGroups(
            int[] shape,
            int[] inStrides,
            int inBaseOffset,
            int[] outStrides,
            int outBaseOffset,
            int axis,
            CpuKernelContext context,
            GroupComputer computer
    ) {
        int[] reducedShape = reduceShape(shape, axis);
        int[] reducedDenseStrides = TensorMetadata.computeStrides(reducedShape);
        int groupCount = logicalSize(reducedShape);
        int axisSize = shape[axis];
        int axisStrideIn = inStrides[axis];
        int axisStrideOut = outStrides[axis];
        ResolvedReductionHints hints = context.reductionHints();

        if (canUseDenseContiguousLastAxisFastPath(shape, inStrides, outStrides, axis, axisStrideIn, axisStrideOut)) {
            runDenseContiguousLastAxisGroups(groupCount, axisSize, inBaseOffset, outBaseOffset, hints, computer);
            return;
        }

        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                for (int group = start; group < end; group++) {
                    runGenericGroup(group, shape, inStrides, inBaseOffset, outStrides, outBaseOffset, axis, reducedDenseStrides, axisSize, axisStrideIn, axisStrideOut, computer);
                }
            });
            return;
        }

        for (int group = 0; group < groupCount; group++) {
            runGenericGroup(group, shape, inStrides, inBaseOffset, outStrides, outBaseOffset, axis, reducedDenseStrides, axisSize, axisStrideIn, axisStrideOut, computer);
        }
    }

    private static void runDenseContiguousLastAxisGroups(
            int groupCount,
            int axisSize,
            int inBaseOffset,
            int outBaseOffset,
            ResolvedReductionHints hints,
            GroupComputer computer
    ) {
        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                for (int group = start; group < end; group++) {
                    int baseIn = inBaseOffset + group * axisSize;
                    int baseOut = outBaseOffset + group * axisSize;
                    computer.compute(baseIn, baseOut, 1, 1, axisSize);
                }
            });
            return;
        }
        for (int group = 0; group < groupCount; group++) {
            int baseIn = inBaseOffset + group * axisSize;
            int baseOut = outBaseOffset + group * axisSize;
            computer.compute(baseIn, baseOut, 1, 1, axisSize);
        }
    }

    private static void runGenericGroup(
            int reducedIndex,
            int[] shape,
            int[] inStrides,
            int inBaseOffset,
            int[] outStrides,
            int outBaseOffset,
            int axis,
            int[] reducedDenseStrides,
            int axisSize,
            int axisStrideIn,
            int axisStrideOut,
            GroupComputer computer
    ) {
        int rem = reducedIndex;
        int baseIn = inBaseOffset;
        int baseOut = outBaseOffset;
        for (int d = 0, rd = 0; d < shape.length; d++) {
            if (d == axis) {
                continue;
            }
            int coord = rem / reducedDenseStrides[rd];
            rem %= reducedDenseStrides[rd];
            baseIn += coord * inStrides[d];
            baseOut += coord * outStrides[d];
            rd++;
        }
        computer.compute(baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
    }

    private static boolean canUseDenseContiguousLastAxisFastPath(
            int[] shape,
            int[] inStrides,
            int[] outStrides,
            int axis,
            int axisStrideIn,
            int axisStrideOut
    ) {
        return axis == shape.length - 1
                && axisStrideIn == 1
                && axisStrideOut == 1
                && isDenseContiguous(shape, inStrides)
                && isDenseContiguous(shape, outStrides);
    }

    private static boolean isDenseContiguous(int[] shape, int[] strides) {
        if (shape.length != strides.length) {
            return false;
        }
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expected) {
                return false;
            }
            expected *= shape[i];
        }
        return true;
    }

    private static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    @FunctionalInterface
    interface GroupComputer {
        void compute(int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize);
    }
}
