package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedReductionHints;
import tensor.TensorMetadata;

final class SoftmaxGradTraversal {
    private SoftmaxGradTraversal() {}

    static void validateShapes(int[] primaryShape, int[] outGradShape, int[] outShape, int dimension, String label) {
        validateSameShape(primaryShape, outShape, label + " output shape must match primary input");
        validateSameShape(primaryShape, outGradShape, label + " gradient input shape must match primary input");
        if (dimension < 0 || dimension >= primaryShape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
    }

    static void runGroups(
            int[] shape,
            int[] primaryStrides,
            int primaryBaseOffset,
            int[] gradStrides,
            int gradBaseOffset,
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
        int primaryAxisStride = primaryStrides[axis];
        int gradAxisStride = gradStrides[axis];
        int outAxisStride = outStrides[axis];
        ResolvedReductionHints hints = context.reductionHints();

        if (canUseDenseContiguousLastAxisFastPath(shape, primaryStrides, gradStrides, outStrides, axis, primaryAxisStride, gradAxisStride, outAxisStride)) {
            runDenseContiguousLastAxisGroups(groupCount, axisSize, primaryBaseOffset, gradBaseOffset, outBaseOffset, hints, computer);
            return;
        }

        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                for (int group = start; group < end; group++) {
                    runGenericGroup(group, shape, primaryStrides, primaryBaseOffset, gradStrides, gradBaseOffset, outStrides, outBaseOffset,
                            axis, reducedDenseStrides, axisSize, primaryAxisStride, gradAxisStride, outAxisStride, computer);
                }
            });
            return;
        }

        for (int group = 0; group < groupCount; group++) {
            runGenericGroup(group, shape, primaryStrides, primaryBaseOffset, gradStrides, gradBaseOffset, outStrides, outBaseOffset,
                    axis, reducedDenseStrides, axisSize, primaryAxisStride, gradAxisStride, outAxisStride, computer);
        }
    }

    private static void runDenseContiguousLastAxisGroups(
            int groupCount,
            int axisSize,
            int primaryBaseOffset,
            int gradBaseOffset,
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
                    int primaryBase = primaryBaseOffset + group * axisSize;
                    int gradBase = gradBaseOffset + group * axisSize;
                    int outBase = outBaseOffset + group * axisSize;
                    computer.compute(primaryBase, gradBase, outBase, 1, 1, 1, axisSize);
                }
            });
            return;
        }
        for (int group = 0; group < groupCount; group++) {
            int primaryBase = primaryBaseOffset + group * axisSize;
            int gradBase = gradBaseOffset + group * axisSize;
            int outBase = outBaseOffset + group * axisSize;
            computer.compute(primaryBase, gradBase, outBase, 1, 1, 1, axisSize);
        }
    }

    private static void runGenericGroup(
            int reducedIndex,
            int[] shape,
            int[] primaryStrides,
            int primaryBaseOffset,
            int[] gradStrides,
            int gradBaseOffset,
            int[] outStrides,
            int outBaseOffset,
            int axis,
            int[] reducedDenseStrides,
            int axisSize,
            int primaryAxisStride,
            int gradAxisStride,
            int outAxisStride,
            GroupComputer computer
    ) {
        int rem = reducedIndex;
        int primaryBase = primaryBaseOffset;
        int gradBase = gradBaseOffset;
        int outBase = outBaseOffset;
        for (int d = 0, rd = 0; d < shape.length; d++) {
            if (d == axis) {
                continue;
            }
            int coord = rem / reducedDenseStrides[rd];
            rem %= reducedDenseStrides[rd];
            primaryBase += coord * primaryStrides[d];
            gradBase += coord * gradStrides[d];
            outBase += coord * outStrides[d];
            rd++;
        }
        computer.compute(primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize);
    }

    private static boolean canUseDenseContiguousLastAxisFastPath(
            int[] shape,
            int[] primaryStrides,
            int[] gradStrides,
            int[] outStrides,
            int axis,
            int primaryAxisStride,
            int gradAxisStride,
            int outAxisStride
    ) {
        return axis == shape.length - 1
                && primaryAxisStride == 1
                && gradAxisStride == 1
                && outAxisStride == 1
                && isDenseContiguous(shape, primaryStrides)
                && isDenseContiguous(shape, gradStrides)
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

    private static void validateSameShape(int[] actual, int[] expected, String message) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(message);
        }
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != expected[i]) {
                throw new IllegalArgumentException(message);
            }
        }
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
        void compute(int primaryBase, int gradBase, int outBase, int primaryAxisStride, int gradAxisStride, int outAxisStride, int axisSize);
    }
}
