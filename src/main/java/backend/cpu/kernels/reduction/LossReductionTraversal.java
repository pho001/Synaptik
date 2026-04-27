package backend.cpu.kernels.reduction;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuThreadPool;
import backend.cpu.kernels.reduction.plan.ResolvedReductionHints;
import tensor.TensorMetadata;

final class LossReductionTraversal {
    private LossReductionTraversal() {}

    static void validateShapes(int[] shape, int[] targetShape, int[] outShape, int classDimension, String label) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException(label + " input shape must not be empty");
        }
        if (classDimension < 0 || classDimension >= shape.length) {
            throw new IllegalArgumentException("Class dimension out of bounds: " + classDimension);
        }
        if (targetShape.length != shape.length) {
            throw new IllegalArgumentException("Targets shape rank must match input rank");
        }
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != targetShape[i]) {
                throw new IllegalArgumentException("Targets shape must match input shape");
            }
        }
        if (outShape.length != 1 || outShape[0] != 1) {
            throw new IllegalArgumentException(label + " output must be scalar-shaped [1]");
        }
    }

    static double reduceMeanLoss(
            int[] shape,
            int[] aStrides,
            int aBaseOffset,
            int[] bStrides,
            int bBaseOffset,
            int classDimension,
            CpuKernelContext context,
            GroupComputer computer
    ) {
        int[] reducedShape = reduceShape(shape, classDimension);
        int[] reducedDenseStrides = TensorMetadata.computeStrides(reducedShape);
        int groupCount = logicalSize(reducedShape);
        int axisSize = shape[classDimension];
        int axisStrideA = aStrides[classDimension];
        int axisStrideB = bStrides[classDimension];

        if (groupCount == 0) {
            return 0.0d;
        }

        ResolvedReductionHints hints = context.reductionHints();
        if (canUseDenseContiguousLastAxisFastPath(shape, aStrides, bStrides, classDimension, axisStrideA, axisStrideB)) {
            return reduceDenseContiguousLastAxis(groupCount, axisSize, aBaseOffset, bBaseOffset, hints, computer);
        }
        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                double partial = 0.0d;
                for (int group = start; group < end; group++) {
                    partial += computeGenericGroup(group, shape, aStrides, aBaseOffset, bStrides, bBaseOffset, classDimension, reducedDenseStrides, axisSize, axisStrideA, axisStrideB, computer);
                }
                partials[chunk] = partial;
            });
            double total = 0.0d;
            for (double partial : partials) {
                total += partial;
            }
            return total / groupCount;
        }

        double total = 0.0d;
        for (int group = 0; group < groupCount; group++) {
            total += computeGenericGroup(group, shape, aStrides, aBaseOffset, bStrides, bBaseOffset, classDimension, reducedDenseStrides, axisSize, axisStrideA, axisStrideB, computer);
        }
        return total / groupCount;
    }

    private static double reduceDenseContiguousLastAxis(
            int groupCount,
            int axisSize,
            int aBaseOffset,
            int bBaseOffset,
            ResolvedReductionHints hints,
            GroupComputer computer
    ) {
        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                double partial = 0.0d;
                for (int group = start; group < end; group++) {
                    int baseA = aBaseOffset + group * axisSize;
                    int baseB = bBaseOffset + group * axisSize;
                    partial += computer.compute(baseA, baseB, 1, 1, axisSize);
                }
                partials[chunk] = partial;
            });
            double total = 0.0d;
            for (double partial : partials) {
                total += partial;
            }
            return total / groupCount;
        }

        double total = 0.0d;
        for (int group = 0; group < groupCount; group++) {
            int baseA = aBaseOffset + group * axisSize;
            int baseB = bBaseOffset + group * axisSize;
            total += computer.compute(baseA, baseB, 1, 1, axisSize);
        }
        return total / groupCount;
    }

    private static double computeGenericGroup(
            int reducedIndex,
            int[] shape,
            int[] aStrides,
            int aBaseOffset,
            int[] bStrides,
            int bBaseOffset,
            int classDimension,
            int[] reducedDenseStrides,
            int axisSize,
            int axisStrideA,
            int axisStrideB,
            GroupComputer computer
    ) {
        int rem = reducedIndex;
        int baseA = aBaseOffset;
        int baseB = bBaseOffset;
        for (int d = 0, rd = 0; d < shape.length; d++) {
            if (d == classDimension) {
                continue;
            }
            int coord = rem / reducedDenseStrides[rd];
            rem %= reducedDenseStrides[rd];
            baseA += coord * aStrides[d];
            baseB += coord * bStrides[d];
            rd++;
        }
        return computer.compute(baseA, baseB, axisStrideA, axisStrideB, axisSize);
    }

    private static boolean canUseDenseContiguousLastAxisFastPath(
            int[] shape,
            int[] aStrides,
            int[] bStrides,
            int classDimension,
            int axisStrideA,
            int axisStrideB
    ) {
        return classDimension == shape.length - 1
                && axisStrideA == 1
                && axisStrideB == 1
                && isDenseContiguous(shape, aStrides)
                && isDenseContiguous(shape, bStrides);
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
        double compute(int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize);
    }
}
