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

        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                for (int group = start; group < end; group++) {
                    GroupState state = groupState(group, shape, inStrides, inBaseOffset, outStrides, outBaseOffset, axis, reducedDenseStrides, axisSize, axisStrideIn, axisStrideOut);
                    computer.compute(state);
                }
            });
            return;
        }

        for (int group = 0; group < groupCount; group++) {
            GroupState state = groupState(group, shape, inStrides, inBaseOffset, outStrides, outBaseOffset, axis, reducedDenseStrides, axisSize, axisStrideIn, axisStrideOut);
            computer.compute(state);
        }
    }

    private static GroupState groupState(
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
            int axisStrideOut
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
        return new GroupState(baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
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
        void compute(GroupState state);
    }

    record GroupState(int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
    }
}
