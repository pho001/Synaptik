package backend.cpu.kernels.reduction;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.*;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuThreadPool;
import backend.cpu.kernels.reduction.plan.ResolvedReductionHints;
import tensor.Tensor;
import tensor.TensorMetadata;

final class MinMaxReduceLoops {
    private MinMaxReduceLoops() {}

    static void execute(Tensor input, Tensor node, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = input.getShapeUnsafe();
        ReductionTraversal.validateDimension(shape, dimension);

        double[] out = TensorInternalAccess.float64Data(node);
        if (out == null) {
            throw new IllegalStateException("F64 output storage is missing");
        }

        if (dimension == -1) {
            out[node.getStorageOffsetUnsafe()] = reduceAllF64(
                    TensorInternalAccess.float64Data(input),
                    shape,
                    input.getStridesUnsafe(),
                    input.getStorageOffsetUnsafe(),
                    input.isContiguous() && !input.hasStorageOffset(),
                    context,
                    isMax
            );
            return;
        }
        reduceAxisF64(
                TensorInternalAccess.float64Data(input),
                shape,
                input.getStridesUnsafe(),
                input.getStorageOffsetUnsafe(),
                out,
                node.getShapeUnsafe(),
                node.getStorageOffsetUnsafe(),
                dimension,
                context,
                isMax
        );
    }

    static void executeF32(Tensor input, Tensor node, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = input.getShapeUnsafe();
        ReductionTraversal.validateDimension(shape, dimension);

        float[] in = TensorInternalAccess.float32Data(input);
        float[] out = TensorInternalAccess.float32Data(node);
        if (in == null || out == null) {
            throw new IllegalStateException("F32 storage is missing");
        }

        if (dimension == -1) {
            out[node.getStorageOffsetUnsafe()] = (float) reduceAllF32(
                    in,
                    shape,
                    input.getStridesUnsafe(),
                    input.getStorageOffsetUnsafe(),
                    input.isContiguous() && !input.hasStorageOffset(),
                    context,
                    isMax
            );
            return;
        }
        reduceAxisF32(in, shape, input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), out, node.getShapeUnsafe(), node.getStorageOffsetUnsafe(), dimension, context, isMax);
    }

    static void executeBF16(Tensor input, Tensor node, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = input.getShapeUnsafe();
        ReductionTraversal.validateDimension(shape, dimension);

        short[] in = TensorInternalAccess.bfloat16Data(input);
        short[] out = TensorInternalAccess.bfloat16Data(node);
        if (in == null || out == null) {
            throw new IllegalStateException("BF16 storage is missing");
        }

        if (dimension == -1) {
            out[node.getStorageOffsetUnsafe()] = CpuDTypeOps.toBFloat16Bits((float) reduceAllF16(
                    in,
                    shape,
                    input.getStridesUnsafe(),
                    input.getStorageOffsetUnsafe(),
                    input.isContiguous() && !input.hasStorageOffset(),
                    context,
                    isMax
            ));
            return;
        }
        reduceAxisF16(in, shape, input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), out, node.getShapeUnsafe(), node.getStorageOffsetUnsafe(), dimension, context, isMax);
    }

    private static void reduceAxisF64(
            double[] in,
            int[] inputShape,
            int[] inputStrides,
            int inputBaseOffset,
            double[] out,
            int[] outShape,
            int outBaseOffset,
            int dimension,
            CpuKernelContext context,
            boolean isMax
    ) {
        ReductionTraversal.forEachAxisGroup(inputShape, inputStrides, inputBaseOffset, outShape, dimension, context, (outIndex, baseOffset, reducedSize, reducedStride) -> {
            double best = in[baseOffset];
            for (int r = 1; r < reducedSize; r++) {
                double value = in[baseOffset + r * reducedStride];
                best = isMax ? Math.max(best, value) : Math.min(best, value);
            }
            out[outBaseOffset + outIndex] = best;
        });
    }

    private static void reduceAxisF32(
            float[] in,
            int[] inputShape,
            int[] inputStrides,
            int inputBaseOffset,
            float[] out,
            int[] outShape,
            int outBaseOffset,
            int dimension,
            CpuKernelContext context,
            boolean isMax
    ) {
        ReductionTraversal.forEachAxisGroup(inputShape, inputStrides, inputBaseOffset, outShape, dimension, context, (outIndex, baseOffset, reducedSize, reducedStride) -> {
            float best = in[baseOffset];
            for (int r = 1; r < reducedSize; r++) {
                float value = in[baseOffset + r * reducedStride];
                best = isMax ? Math.max(best, value) : Math.min(best, value);
            }
            out[outBaseOffset + outIndex] = best;
        });
    }

    private static void reduceAxisF16(
            short[] in,
            int[] inputShape,
            int[] inputStrides,
            int inputBaseOffset,
            short[] out,
            int[] outShape,
            int outBaseOffset,
            int dimension,
            CpuKernelContext context,
            boolean isMax
    ) {
        ReductionTraversal.forEachAxisGroup(inputShape, inputStrides, inputBaseOffset, outShape, dimension, context, (outIndex, baseOffset, reducedSize, reducedStride) -> {
            float best = CpuDTypeOps.fromBFloat16Bits(in[baseOffset]);
            for (int r = 1; r < reducedSize; r++) {
                float value = CpuDTypeOps.fromBFloat16Bits(in[baseOffset + r * reducedStride]);
                best = isMax ? Math.max(best, value) : Math.min(best, value);
            }
            out[outBaseOffset + outIndex] = CpuDTypeOps.toBFloat16Bits(best);
        });
    }

    private static double reduceAllF64(
            double[] in,
            int[] shape,
            int[] strides,
            int baseOffset,
            boolean contiguous,
            CpuKernelContext context,
            boolean isMax
    ) {
        int logicalSize = ReductionTraversal.logicalSize(shape);
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
            return reduceContiguousRangeF64(in, baseOffset, baseOffset + logicalSize, isMax);
        }
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeF64(in, shape, strides, denseStrides, baseOffset, start, end, isMax);
            });
            return mergePartialsF64(partials, isMax);
        }
        return reduceStridedRangeF64(in, shape, strides, denseStrides, baseOffset, 0, logicalSize, isMax);
    }

    private static double reduceAllF32(
            float[] in,
            int[] shape,
            int[] strides,
            int baseOffset,
            boolean contiguous,
            CpuKernelContext context,
            boolean isMax
    ) {
        int logicalSize = ReductionTraversal.logicalSize(shape);
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
            return reduceContiguousRangeF32(in, baseOffset, baseOffset + logicalSize, isMax);
        }
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeF32(in, shape, strides, denseStrides, baseOffset, start, end, isMax);
            });
            return mergePartialsF64(partials, isMax);
        }
        return reduceStridedRangeF32(in, shape, strides, denseStrides, baseOffset, 0, logicalSize, isMax);
    }

    private static double reduceAllF16(
            short[] in,
            int[] shape,
            int[] strides,
            int baseOffset,
            boolean contiguous,
            CpuKernelContext context,
            boolean isMax
    ) {
        int logicalSize = ReductionTraversal.logicalSize(shape);
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
            return reduceContiguousRangeF16(in, baseOffset, baseOffset + logicalSize, isMax);
        }
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeF16(in, shape, strides, denseStrides, baseOffset, start, end, isMax);
            });
            return mergePartialsF64(partials, isMax);
        }
        return reduceStridedRangeF16(in, shape, strides, denseStrides, baseOffset, 0, logicalSize, isMax);
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
        float best = CpuDTypeOps.fromBFloat16Bits(in[start]);
        for (int i = start + 1; i < end; i++) {
            float value = CpuDTypeOps.fromBFloat16Bits(in[i]);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static double reduceStridedRangeF64(double[] in, int[] shape, int[] strides, int[] denseStrides, int baseOffset, int start, int end, boolean isMax) {
        int offset = ReductionTraversal.logicalToOffset(start, shape, strides, denseStrides, baseOffset);
        double best = in[offset];
        for (int logical = start + 1; logical < end; logical++) {
            best = isMax
                    ? Math.max(best, in[ReductionTraversal.logicalToOffset(logical, shape, strides, denseStrides, baseOffset)])
                    : Math.min(best, in[ReductionTraversal.logicalToOffset(logical, shape, strides, denseStrides, baseOffset)]);
        }
        return best;
    }

    private static double reduceStridedRangeF32(float[] in, int[] shape, int[] strides, int[] denseStrides, int baseOffset, int start, int end, boolean isMax) {
        int offset = ReductionTraversal.logicalToOffset(start, shape, strides, denseStrides, baseOffset);
        float best = in[offset];
        for (int logical = start + 1; logical < end; logical++) {
            float value = in[ReductionTraversal.logicalToOffset(logical, shape, strides, denseStrides, baseOffset)];
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static double reduceStridedRangeF16(short[] in, int[] shape, int[] strides, int[] denseStrides, int baseOffset, int start, int end, boolean isMax) {
        int offset = ReductionTraversal.logicalToOffset(start, shape, strides, denseStrides, baseOffset);
        float best = CpuDTypeOps.fromBFloat16Bits(in[offset]);
        for (int logical = start + 1; logical < end; logical++) {
            float value = CpuDTypeOps.fromBFloat16Bits(in[ReductionTraversal.logicalToOffset(logical, shape, strides, denseStrides, baseOffset)]);
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

}
