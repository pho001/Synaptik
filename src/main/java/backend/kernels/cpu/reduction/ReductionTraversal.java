package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.reduction.plan.ResolvedReductionHints;
import tensor.TensorMetadata;

final class ReductionTraversal {
    private ReductionTraversal() {}

    static void validateDimension(int[] shape, int dimension) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
    }

    static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    static int logicalToOffset(int logicalIndex, int[] shape, int[] strides, int[] denseStrides, int baseOffset) {
        int idx = logicalIndex;
        int offset = baseOffset;
        for (int d = 0; d < shape.length; d++) {
            int coord = idx / denseStrides[d];
            idx %= denseStrides[d];
            offset += coord * strides[d];
        }
        return offset;
    }

    static void forEachAxisGroup(
            int[] inputShape,
            int[] inputStrides,
            int inputBaseOffset,
            int[] outShape,
            int reducedDimension,
            CpuKernelContext context,
            GroupVisitor visitor
    ) {
        int groups = logicalSize(outShape);
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        int reducedSize = inputShape[reducedDimension];
        int reducedStride = inputStrides[reducedDimension];
        if (shouldParallelize(groups, context)) {
            parallelFor(groups, context, (start, end) -> {
                for (int outIndex = start; outIndex < end; outIndex++) {
                    int baseOffset = reductionBaseOffset(outIndex, outShape, outDenseStrides, inputStrides, reducedDimension, inputBaseOffset);
                    visitor.visit(outIndex, baseOffset, reducedSize, reducedStride);
                }
            });
            return;
        }
        for (int outIndex = 0; outIndex < groups; outIndex++) {
            int baseOffset = reductionBaseOffset(outIndex, outShape, outDenseStrides, inputStrides, reducedDimension, inputBaseOffset);
            visitor.visit(outIndex, baseOffset, reducedSize, reducedStride);
        }
    }

    static void forEachAxisGroup(
            int[] inputShape,
            int[] inputStrides,
            int inputBaseOffset,
            int[] outShape,
            int reducedDimension,
            GroupVisitor visitor
    ) {
        int groups = logicalSize(outShape);
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        int reducedSize = inputShape[reducedDimension];
        int reducedStride = inputStrides[reducedDimension];
        for (int outIndex = 0; outIndex < groups; outIndex++) {
            int baseOffset = reductionBaseOffset(outIndex, outShape, outDenseStrides, inputStrides, reducedDimension, inputBaseOffset);
            visitor.visit(outIndex, baseOffset, reducedSize, reducedStride);
        }
    }

    private static int reductionBaseOffset(
            int outIndex,
            int[] outShape,
            int[] outDenseStrides,
            int[] inputStrides,
            int reducedDimension,
            int inputBaseOffset
    ) {
        int idx = outIndex;
        int baseOffset = inputBaseOffset;
        int inputRank = inputStrides.length;
        if (outShape.length == inputRank) {
            for (int outDim = 0; outDim < outShape.length; outDim++) {
                int coord = idx / outDenseStrides[outDim];
                idx %= outDenseStrides[outDim];
                if (outDim == reducedDimension) {
                    continue;
                }
                baseOffset += coord * inputStrides[outDim];
            }
            return baseOffset;
        }
        for (int outDim = 0; outDim < outShape.length; outDim++) {
            int coord = idx / outDenseStrides[outDim];
            idx %= outDenseStrides[outDim];
            int inputDim = outDim < reducedDimension ? outDim : outDim + 1;
            baseOffset += coord * inputStrides[inputDim];
        }
        return baseOffset;
    }

    private static boolean shouldParallelize(int logicalSize, CpuKernelContext context) {
        ResolvedReductionHints hints = context.reductionHints();
        return hints != null && hints.parallel() && logicalSize > 1;
    }

    private static int chunkSize(CpuKernelContext context) {
        ResolvedReductionHints hints = context.reductionHints();
        return hints == null ? 1 : hints.chunkSize();
    }

    private static int workers(CpuKernelContext context) {
        ResolvedReductionHints hints = context.reductionHints();
        return hints == null ? 1 : hints.plannedWorkers();
    }

    private static void parallelFor(int logicalSize, CpuKernelContext context, RangeBody body) {
        int chunkSize = chunkSize(context);
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, logicalSize);
            body.run(start, end);
        });
    }

    @FunctionalInterface
    interface GroupVisitor {
        void visit(int outIndex, int baseOffset, int reducedSize, int reducedStride);
    }

    @FunctionalInterface
    private interface RangeBody {
        void run(int start, int end);
    }
}
