package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedReductionHints;
import tensor.Tensor;
import tensor.TensorMetadata;

final class MinMaxReduceLoops {
    private MinMaxReduceLoops() {}

    static void execute(Tensor input, Tensor node, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = input.getShapeUnsafe();
        validateDimension(shape, dimension);

        double[] out = node.getFloat64Data();
        if (out == null) {
            throw new IllegalStateException("F64 output storage is missing");
        }

        if (dimension == -1) {
            out[0] = reduceAllF64(input.getData(), shape, input.getStridesUnsafe(), input.isContiguous(), context, isMax);
            return;
        }
        reduceAxisF64(input.getData(), shape, input.getStridesUnsafe(), out, node.getShapeUnsafe(), dimension, context, isMax);
    }

    static void executeF32(Tensor input, Tensor node, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = input.getShapeUnsafe();
        validateDimension(shape, dimension);

        float[] in = input.getFloat32Data();
        float[] out = node.getFloat32Data();
        if (in == null || out == null) {
            throw new IllegalStateException("F32 storage is missing");
        }

        if (dimension == -1) {
            out[0] = (float) reduceAllF32(in, shape, input.getStridesUnsafe(), input.isContiguous(), context, isMax);
            return;
        }
        reduceAxisF32(in, shape, input.getStridesUnsafe(), out, node.getShapeUnsafe(), dimension, context, isMax);
    }

    static void executeF16(Tensor input, Tensor node, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = input.getShapeUnsafe();
        validateDimension(shape, dimension);

        short[] in = input.getFloat16Data();
        short[] out = node.getFloat16Data();
        if (in == null || out == null) {
            throw new IllegalStateException("F16 storage is missing");
        }

        if (dimension == -1) {
            out[0] = CpuDTypeOps.toHalfBits((float) reduceAllF16(in, shape, input.getStridesUnsafe(), input.isContiguous(), context, isMax));
            return;
        }
        reduceAxisF16(in, shape, input.getStridesUnsafe(), out, node.getShapeUnsafe(), dimension, context, isMax);
    }

    private static void reduceAxisF64(
            double[] in,
            int[] inputShape,
            int[] inputStrides,
            double[] out,
            int[] outShape,
            int dimension,
            CpuKernelContext context,
            boolean isMax
    ) {
        int groups = out.length;
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        if (shouldParallelize(groups, context)) {
            parallelFor(groups, context, (start, end) ->
                    reduceAxisRangeF64(in, inputShape, inputStrides, out, outShape, outDenseStrides, dimension, isMax, start, end)
            );
            return;
        }
        reduceAxisRangeF64(in, inputShape, inputStrides, out, outShape, outDenseStrides, dimension, isMax, 0, groups);
    }

    private static void reduceAxisRangeF64(
            double[] in,
            int[] inputShape,
            int[] inputStrides,
            double[] out,
            int[] outShape,
            int[] outDenseStrides,
            int dimension,
            boolean isMax,
            int start,
            int end
    ) {
        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outIndex = start; outIndex < end; outIndex++) {
            int baseOffset = reductionBaseOffset(outIndex, outShape, outDenseStrides, inputStrides, dimension);
            double best = in[baseOffset];
            for (int r = 1; r < reducedSize; r++) {
                double value = in[baseOffset + r * reducedStride];
                best = isMax ? Math.max(best, value) : Math.min(best, value);
            }
            out[outIndex] = best;
        }
    }

    private static void reduceAxisF32(
            float[] in,
            int[] inputShape,
            int[] inputStrides,
            float[] out,
            int[] outShape,
            int dimension,
            CpuKernelContext context,
            boolean isMax
    ) {
        int groups = out.length;
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        if (shouldParallelize(groups, context)) {
            parallelFor(groups, context, (start, end) ->
                    reduceAxisRangeF32(in, inputShape, inputStrides, out, outShape, outDenseStrides, dimension, isMax, start, end)
            );
            return;
        }
        reduceAxisRangeF32(in, inputShape, inputStrides, out, outShape, outDenseStrides, dimension, isMax, 0, groups);
    }

    private static void reduceAxisRangeF32(
            float[] in,
            int[] inputShape,
            int[] inputStrides,
            float[] out,
            int[] outShape,
            int[] outDenseStrides,
            int dimension,
            boolean isMax,
            int start,
            int end
    ) {
        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outIndex = start; outIndex < end; outIndex++) {
            int baseOffset = reductionBaseOffset(outIndex, outShape, outDenseStrides, inputStrides, dimension);
            float best = in[baseOffset];
            for (int r = 1; r < reducedSize; r++) {
                float value = in[baseOffset + r * reducedStride];
                best = isMax ? Math.max(best, value) : Math.min(best, value);
            }
            out[outIndex] = best;
        }
    }

    private static void reduceAxisF16(
            short[] in,
            int[] inputShape,
            int[] inputStrides,
            short[] out,
            int[] outShape,
            int dimension,
            CpuKernelContext context,
            boolean isMax
    ) {
        int groups = out.length;
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        if (shouldParallelize(groups, context)) {
            parallelFor(groups, context, (start, end) ->
                    reduceAxisRangeF16(in, inputShape, inputStrides, out, outShape, outDenseStrides, dimension, isMax, start, end)
            );
            return;
        }
        reduceAxisRangeF16(in, inputShape, inputStrides, out, outShape, outDenseStrides, dimension, isMax, 0, groups);
    }

    private static void reduceAxisRangeF16(
            short[] in,
            int[] inputShape,
            int[] inputStrides,
            short[] out,
            int[] outShape,
            int[] outDenseStrides,
            int dimension,
            boolean isMax,
            int start,
            int end
    ) {
        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outIndex = start; outIndex < end; outIndex++) {
            int baseOffset = reductionBaseOffset(outIndex, outShape, outDenseStrides, inputStrides, dimension);
            float best = CpuDTypeOps.fromHalfBits(in[baseOffset]);
            for (int r = 1; r < reducedSize; r++) {
                float value = CpuDTypeOps.fromHalfBits(in[baseOffset + r * reducedStride]);
                best = isMax ? Math.max(best, value) : Math.min(best, value);
            }
            out[outIndex] = CpuDTypeOps.toHalfBits(best);
        }
    }

    private static double reduceAllF64(
            double[] in,
            int[] shape,
            int[] strides,
            boolean contiguous,
            CpuKernelContext context,
            boolean isMax
    ) {
        int logicalSize = logicalSize(shape);
        if (contiguous) {
            if (shouldParallelize(logicalSize, context)) {
                int chunks = chunkCount(logicalSize, context);
                double[] partials = new double[chunks];
                CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                    int start = chunk * chunkSize(context);
                    int end = Math.min(start + chunkSize(context), logicalSize);
                    partials[chunk] = reduceContiguousRangeF64(in, start, end, isMax);
                });
                return mergePartialsF64(partials, isMax);
            }
            return reduceContiguousRangeF64(in, 0, logicalSize, isMax);
        }
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeF64(in, shape, strides, denseStrides, start, end, isMax);
            });
            return mergePartialsF64(partials, isMax);
        }
        return reduceStridedRangeF64(in, shape, strides, denseStrides, 0, logicalSize, isMax);
    }

    private static double reduceAllF32(
            float[] in,
            int[] shape,
            int[] strides,
            boolean contiguous,
            CpuKernelContext context,
            boolean isMax
    ) {
        int logicalSize = logicalSize(shape);
        if (contiguous) {
            if (shouldParallelize(logicalSize, context)) {
                int chunks = chunkCount(logicalSize, context);
                double[] partials = new double[chunks];
                CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                    int start = chunk * chunkSize(context);
                    int end = Math.min(start + chunkSize(context), logicalSize);
                    partials[chunk] = reduceContiguousRangeF32(in, start, end, isMax);
                });
                return mergePartialsF64(partials, isMax);
            }
            return reduceContiguousRangeF32(in, 0, logicalSize, isMax);
        }
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeF32(in, shape, strides, denseStrides, start, end, isMax);
            });
            return mergePartialsF64(partials, isMax);
        }
        return reduceStridedRangeF32(in, shape, strides, denseStrides, 0, logicalSize, isMax);
    }

    private static double reduceAllF16(
            short[] in,
            int[] shape,
            int[] strides,
            boolean contiguous,
            CpuKernelContext context,
            boolean isMax
    ) {
        int logicalSize = logicalSize(shape);
        if (contiguous) {
            if (shouldParallelize(logicalSize, context)) {
                int chunks = chunkCount(logicalSize, context);
                double[] partials = new double[chunks];
                CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                    int start = chunk * chunkSize(context);
                    int end = Math.min(start + chunkSize(context), logicalSize);
                    partials[chunk] = reduceContiguousRangeF16(in, start, end, isMax);
                });
                return mergePartialsF64(partials, isMax);
            }
            return reduceContiguousRangeF16(in, 0, logicalSize, isMax);
        }
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeF16(in, shape, strides, denseStrides, start, end, isMax);
            });
            return mergePartialsF64(partials, isMax);
        }
        return reduceStridedRangeF16(in, shape, strides, denseStrides, 0, logicalSize, isMax);
    }

    private static double reduceContiguousRangeF64(double[] in, int start, int end, boolean isMax) {
        double best = in[start];
        for (int i = start + 1; i < end; i++) {
            best = isMax ? Math.max(best, in[i]) : Math.min(best, in[i]);
        }
        return best;
    }

    private static double reduceContiguousRangeF32(float[] in, int start, int end, boolean isMax) {
        float best = in[start];
        for (int i = start + 1; i < end; i++) {
            best = isMax ? Math.max(best, in[i]) : Math.min(best, in[i]);
        }
        return best;
    }

    private static double reduceContiguousRangeF16(short[] in, int start, int end, boolean isMax) {
        float best = CpuDTypeOps.fromHalfBits(in[start]);
        for (int i = start + 1; i < end; i++) {
            float value = CpuDTypeOps.fromHalfBits(in[i]);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static double reduceStridedRangeF64(double[] in, int[] shape, int[] strides, int[] denseStrides, int start, int end, boolean isMax) {
        int offset = logicalToOffset(start, shape, strides, denseStrides);
        double best = in[offset];
        for (int logical = start + 1; logical < end; logical++) {
            best = isMax
                    ? Math.max(best, in[logicalToOffset(logical, shape, strides, denseStrides)])
                    : Math.min(best, in[logicalToOffset(logical, shape, strides, denseStrides)]);
        }
        return best;
    }

    private static double reduceStridedRangeF32(float[] in, int[] shape, int[] strides, int[] denseStrides, int start, int end, boolean isMax) {
        int offset = logicalToOffset(start, shape, strides, denseStrides);
        float best = in[offset];
        for (int logical = start + 1; logical < end; logical++) {
            float value = in[logicalToOffset(logical, shape, strides, denseStrides)];
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static double reduceStridedRangeF16(short[] in, int[] shape, int[] strides, int[] denseStrides, int start, int end, boolean isMax) {
        int offset = logicalToOffset(start, shape, strides, denseStrides);
        float best = CpuDTypeOps.fromHalfBits(in[offset]);
        for (int logical = start + 1; logical < end; logical++) {
            float value = CpuDTypeOps.fromHalfBits(in[logicalToOffset(logical, shape, strides, denseStrides)]);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static double mergePartialsF64(double[] partials, boolean isMax) {
        double best = partials[0];
        for (int i = 1; i < partials.length; i++) {
            best = isMax ? Math.max(best, partials[i]) : Math.min(best, partials[i]);
        }
        return best;
    }

    private static int reductionBaseOffset(
            int outIndex,
            int[] outShape,
            int[] outDenseStrides,
            int[] inputStrides,
            int reducedDimension
    ) {
        int idx = outIndex;
        int baseOffset = 0;
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

    private static int logicalToOffset(int logicalIndex, int[] shape, int[] strides, int[] denseStrides) {
        int idx = logicalIndex;
        int offset = 0;
        for (int d = 0; d < shape.length; d++) {
            int coord = idx / denseStrides[d];
            idx %= denseStrides[d];
            offset += coord * strides[d];
        }
        return offset;
    }

    private static void validateDimension(int[] shape, int dimension) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
    }

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
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

    private static int chunkCount(int logicalSize, CpuKernelContext context) {
        return (logicalSize + chunkSize(context) - 1) / chunkSize(context);
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
    private interface RangeBody {
        void run(int start, int end);
    }
}
