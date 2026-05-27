package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.execution.CpuThreadPool;
import backend.cpu.plan.reduction.ResolvedReductionHints;
import backend.cpu.storage.CpuStorageView;
import tensor.TensorMetadata;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

final class MinMaxReduceLoops {
    private MinMaxReduceLoops() {
    }

    static void execute(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = validateShapeAndDimension(input, dimension);
        validateOutputSize(output, shape, input.logicalSize(), dimension);

        if (input.isArray() && output.isArray() && isExactDenseArrayOutput(output)) {
            double[] in = input.requireF64Array();
            double[] out = output.requireF64Array();
            if (dimension == -1) {
                out[0] = reduceAllF64(input, in, context, isMax);
                return;
            }
            reduceAxisF64(in, shape, input.strides(), input.storageOffset(), out, output.shape(), 0, dimension, context, isMax);
            return;
        }
        executeStorageF64(input, output, dimension, isMax);
    }

    static void executeF32(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = validateShapeAndDimension(input, dimension);
        validateOutputSize(output, shape, input.logicalSize(), dimension);

        if (input.isArray() && output.isArray() && isExactDenseArrayOutput(output)) {
            float[] in = input.requireF32Array();
            float[] out = output.requireF32Array();
            if (dimension == -1) {
                out[0] = (float) reduceAllF32(input, in, context, isMax);
                return;
            }
            reduceAxisF32(in, shape, input.strides(), input.storageOffset(), out, output.shape(), 0, dimension, context, isMax);
            return;
        }
        executeStorageF32(input, output, dimension, isMax);
    }

    static void executeBF16(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = validateShapeAndDimension(input, dimension);
        validateOutputSize(output, shape, input.logicalSize(), dimension);

        if (input.isArray() && output.isArray() && isExactDenseArrayOutput(output)) {
            short[] in = input.requireBF16Array();
            short[] out = output.requireBF16Array();
            if (dimension == -1) {
                out[0] = TensorDTypeOps.toBFloat16Bits((float) reduceAllBF16(input, in, context, isMax));
                return;
            }
            reduceAxisBF16(in, shape, input.strides(), input.storageOffset(), out, output.shape(), 0, dimension, context, isMax);
            return;
        }
        executeStorageBF16(input, output, dimension, isMax);
    }

    static void executeI32(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = validateShapeAndDimension(input, dimension);
        validateOutputSize(output, shape, input.logicalSize(), dimension);

        if (input.isArray() && output.isArray() && isExactDenseArrayOutput(output)) {
            int[] in = input.requireI32Array();
            int[] out = output.requireI32Array();
            if (dimension == -1) {
                out[0] = reduceAllI32(input, in, context, isMax);
                return;
            }
            reduceAxisI32(in, shape, input.strides(), input.storageOffset(), out, output.shape(), 0, dimension, context, isMax);
            return;
        }
        executeStorageI32(input, output, dimension, isMax);
    }

    static void executeI64(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = validateShapeAndDimension(input, dimension);
        validateOutputSize(output, shape, input.logicalSize(), dimension);

        if (input.isArray() && output.isArray() && isExactDenseArrayOutput(output)) {
            long[] in = input.requireI64Array();
            long[] out = output.requireI64Array();
            if (dimension == -1) {
                out[0] = reduceAllI64(input, in, context, isMax);
                return;
            }
            reduceAxisI64(in, shape, input.strides(), input.storageOffset(), out, output.shape(), 0, dimension, context, isMax);
            return;
        }
        executeStorageI64(input, output, dimension, isMax);
    }

    static void executeBOOL(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context, boolean isMax) {
        int[] shape = validateShapeAndDimension(input, dimension);
        validateOutputSize(output, shape, input.logicalSize(), dimension);

        if (input.isArray() && output.isArray() && isExactDenseArrayOutput(output)) {
            byte[] in = input.requireBoolArray();
            byte[] out = output.requireBoolArray();
            if (dimension == -1) {
                out[0] = reduceAllBOOL(input, in, context, isMax);
                return;
            }
            reduceAxisBOOL(in, shape, input.strides(), input.storageOffset(), out, output.shape(), 0, dimension, context, isMax);
            return;
        }
        executeStorageBOOL(input, output, dimension, isMax);
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

    private static void reduceAxisBF16(
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
            float best = TensorDTypeOps.fromBFloat16Bits(in[baseOffset]);
            for (int r = 1; r < reducedSize; r++) {
                float value = TensorDTypeOps.fromBFloat16Bits(in[baseOffset + r * reducedStride]);
                best = isMax ? Math.max(best, value) : Math.min(best, value);
            }
            out[outBaseOffset + outIndex] = TensorDTypeOps.toBFloat16Bits(best);
        });
    }

    private static void reduceAxisI32(
            int[] in,
            int[] inputShape,
            int[] inputStrides,
            int inputBaseOffset,
            int[] out,
            int[] outShape,
            int outBaseOffset,
            int dimension,
            CpuKernelContext context,
            boolean isMax
    ) {
        ReductionTraversal.forEachAxisGroup(inputShape, inputStrides, inputBaseOffset, outShape, dimension, context, (outIndex, baseOffset, reducedSize, reducedStride) -> {
            int best = in[baseOffset];
            for (int r = 1; r < reducedSize; r++) {
                int value = in[baseOffset + r * reducedStride];
                best = isMax ? Math.max(best, value) : Math.min(best, value);
            }
            out[outBaseOffset + outIndex] = best;
        });
    }

    private static void reduceAxisI64(
            long[] in,
            int[] inputShape,
            int[] inputStrides,
            int inputBaseOffset,
            long[] out,
            int[] outShape,
            int outBaseOffset,
            int dimension,
            CpuKernelContext context,
            boolean isMax
    ) {
        ReductionTraversal.forEachAxisGroup(inputShape, inputStrides, inputBaseOffset, outShape, dimension, context, (outIndex, baseOffset, reducedSize, reducedStride) -> {
            long best = in[baseOffset];
            for (int r = 1; r < reducedSize; r++) {
                long value = in[baseOffset + r * reducedStride];
                best = isMax ? Math.max(best, value) : Math.min(best, value);
            }
            out[outBaseOffset + outIndex] = best;
        });
    }

    private static void reduceAxisBOOL(
            byte[] in,
            int[] inputShape,
            int[] inputStrides,
            int inputBaseOffset,
            byte[] out,
            int[] outShape,
            int outBaseOffset,
            int dimension,
            CpuKernelContext context,
            boolean isMax
    ) {
        ReductionTraversal.forEachAxisGroup(inputShape, inputStrides, inputBaseOffset, outShape, dimension, context, (outIndex, baseOffset, reducedSize, reducedStride) -> {
            boolean best = in[baseOffset] != 0;
            for (int r = 1; r < reducedSize; r++) {
                boolean value = in[baseOffset + r * reducedStride] != 0;
                best = isMax ? best || value : best && value;
            }
            out[outBaseOffset + outIndex] = best ? (byte) 1 : (byte) 0;
        });
    }

    private static double reduceAllF64(CpuStorageView input, double[] data, CpuKernelContext context, boolean isMax) {
        int logicalSize = input.logicalSize();
        if (isDenseContiguous(input)) {
            if (shouldParallelize(logicalSize, context)) {
                int chunks = chunkCount(logicalSize, context);
                double[] partials = new double[chunks];
                CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                    int start = chunk * chunkSize(context);
                    int end = Math.min(start + chunkSize(context), logicalSize);
                    partials[chunk] = reduceContiguousRangeF64(data, start, end, isMax);
                });
                return mergePartialsF64(partials, isMax);
            }
            return reduceContiguousRangeF64(data, 0, logicalSize, isMax);
        }
        int[] shape = input.shape();
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeF64(data, shape, input.strides(), denseStrides, input.storageOffset(), start, end, isMax);
            });
            return mergePartialsF64(partials, isMax);
        }
        return reduceStridedRangeF64(data, shape, input.strides(), denseStrides, input.storageOffset(), 0, logicalSize, isMax);
    }

    private static double reduceAllF32(CpuStorageView input, float[] data, CpuKernelContext context, boolean isMax) {
        int logicalSize = input.logicalSize();
        if (isDenseContiguous(input)) {
            if (shouldParallelize(logicalSize, context)) {
                int chunks = chunkCount(logicalSize, context);
                double[] partials = new double[chunks];
                CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                    int start = chunk * chunkSize(context);
                    int end = Math.min(start + chunkSize(context), logicalSize);
                    partials[chunk] = reduceContiguousRangeF32(data, start, end, isMax);
                });
                return mergePartialsF64(partials, isMax);
            }
            return reduceContiguousRangeF32(data, 0, logicalSize, isMax);
        }
        int[] shape = input.shape();
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeF32(data, shape, input.strides(), denseStrides, input.storageOffset(), start, end, isMax);
            });
            return mergePartialsF64(partials, isMax);
        }
        return reduceStridedRangeF32(data, shape, input.strides(), denseStrides, input.storageOffset(), 0, logicalSize, isMax);
    }

    private static double reduceAllBF16(CpuStorageView input, short[] data, CpuKernelContext context, boolean isMax) {
        int logicalSize = input.logicalSize();
        if (isDenseContiguous(input)) {
            if (shouldParallelize(logicalSize, context)) {
                int chunks = chunkCount(logicalSize, context);
                double[] partials = new double[chunks];
                CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                    int start = chunk * chunkSize(context);
                    int end = Math.min(start + chunkSize(context), logicalSize);
                    partials[chunk] = reduceContiguousRangeBF16(data, start, end, isMax);
                });
                return mergePartialsF64(partials, isMax);
            }
            return reduceContiguousRangeBF16(data, 0, logicalSize, isMax);
        }
        int[] shape = input.shape();
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeBF16(data, shape, input.strides(), denseStrides, input.storageOffset(), start, end, isMax);
            });
            return mergePartialsF64(partials, isMax);
        }
        return reduceStridedRangeBF16(data, shape, input.strides(), denseStrides, input.storageOffset(), 0, logicalSize, isMax);
    }

    private static int reduceAllI32(CpuStorageView input, int[] data, CpuKernelContext context, boolean isMax) {
        int logicalSize = input.logicalSize();
        if (isDenseContiguous(input)) {
            if (shouldParallelize(logicalSize, context)) {
                int chunks = chunkCount(logicalSize, context);
                int[] partials = new int[chunks];
                CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                    int start = chunk * chunkSize(context);
                    int end = Math.min(start + chunkSize(context), logicalSize);
                    partials[chunk] = reduceContiguousRangeI32(data, start, end, isMax);
                });
                return mergePartialsI32(partials, isMax);
            }
            return reduceContiguousRangeI32(data, 0, logicalSize, isMax);
        }
        int[] shape = input.shape();
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            int[] partials = new int[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeI32(data, shape, input.strides(), denseStrides, input.storageOffset(), start, end, isMax);
            });
            return mergePartialsI32(partials, isMax);
        }
        return reduceStridedRangeI32(data, shape, input.strides(), denseStrides, input.storageOffset(), 0, logicalSize, isMax);
    }

    private static long reduceAllI64(CpuStorageView input, long[] data, CpuKernelContext context, boolean isMax) {
        int logicalSize = input.logicalSize();
        if (isDenseContiguous(input)) {
            if (shouldParallelize(logicalSize, context)) {
                int chunks = chunkCount(logicalSize, context);
                long[] partials = new long[chunks];
                CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                    int start = chunk * chunkSize(context);
                    int end = Math.min(start + chunkSize(context), logicalSize);
                    partials[chunk] = reduceContiguousRangeI64(data, start, end, isMax);
                });
                return mergePartialsI64(partials, isMax);
            }
            return reduceContiguousRangeI64(data, 0, logicalSize, isMax);
        }
        int[] shape = input.shape();
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            long[] partials = new long[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeI64(data, shape, input.strides(), denseStrides, input.storageOffset(), start, end, isMax);
            });
            return mergePartialsI64(partials, isMax);
        }
        return reduceStridedRangeI64(data, shape, input.strides(), denseStrides, input.storageOffset(), 0, logicalSize, isMax);
    }

    private static byte reduceAllBOOL(CpuStorageView input, byte[] data, CpuKernelContext context, boolean isMax) {
        int logicalSize = input.logicalSize();
        if (isDenseContiguous(input)) {
            if (shouldParallelize(logicalSize, context)) {
                int chunks = chunkCount(logicalSize, context);
                byte[] partials = new byte[chunks];
                CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                    int start = chunk * chunkSize(context);
                    int end = Math.min(start + chunkSize(context), logicalSize);
                    partials[chunk] = reduceContiguousRangeBOOL(data, start, end, isMax);
                });
                return mergePartialsBOOL(partials, isMax);
            }
            return reduceContiguousRangeBOOL(data, 0, logicalSize, isMax);
        }
        int[] shape = input.shape();
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (shouldParallelize(logicalSize, context)) {
            int chunks = chunkCount(logicalSize, context);
            byte[] partials = new byte[chunks];
            CpuThreadPool.runChunks(chunks, workers(context), chunk -> {
                int start = chunk * chunkSize(context);
                int end = Math.min(start + chunkSize(context), logicalSize);
                partials[chunk] = reduceStridedRangeBOOL(data, shape, input.strides(), denseStrides, input.storageOffset(), start, end, isMax);
            });
            return mergePartialsBOOL(partials, isMax);
        }
        return reduceStridedRangeBOOL(data, shape, input.strides(), denseStrides, input.storageOffset(), 0, logicalSize, isMax);
    }

    private static void executeStorageF64(CpuStorageView input, CpuStorageView output, int dimension, boolean isMax) {
        double[] inArray = ReductionStorageAccess.f64Array(input);
        MemorySegment inSegment = ReductionStorageAccess.f64Segment(input);
        double[] outArray = ReductionStorageAccess.f64Array(output);
        MemorySegment outSegment = ReductionStorageAccess.f64Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();

        if (dimension == -1) {
            double best = reduceStorageAllF64(inArray, inSegment, inputShape, inputStrides, input.storageOffset(), input.logicalSize(), isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF64(outArray, outSegment, outOffset, best);
            return;
        }

        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < output.logicalSize(); outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            double best = reduceStorageFixedBaseF64(inArray, inSegment, inputBase, reducedStride, reducedSize, isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF64(outArray, outSegment, outOffset, best);
        }
    }

    private static void executeStorageF32(CpuStorageView input, CpuStorageView output, int dimension, boolean isMax) {
        float[] inArray = ReductionStorageAccess.f32Array(input);
        MemorySegment inSegment = ReductionStorageAccess.f32Segment(input);
        float[] outArray = ReductionStorageAccess.f32Array(output);
        MemorySegment outSegment = ReductionStorageAccess.f32Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();

        if (dimension == -1) {
            float best = reduceStorageAllF32(inArray, inSegment, inputShape, inputStrides, input.storageOffset(), input.logicalSize(), isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF32(outArray, outSegment, outOffset, best);
            return;
        }

        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < output.logicalSize(); outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            float best = reduceStorageFixedBaseF32(inArray, inSegment, inputBase, reducedStride, reducedSize, isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF32(outArray, outSegment, outOffset, best);
        }
    }

    private static void executeStorageBF16(CpuStorageView input, CpuStorageView output, int dimension, boolean isMax) {
        short[] inArray = ReductionStorageAccess.bf16Array(input);
        MemorySegment inSegment = ReductionStorageAccess.bf16Segment(input);
        short[] outArray = ReductionStorageAccess.bf16Array(output);
        MemorySegment outSegment = ReductionStorageAccess.bf16Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();

        if (dimension == -1) {
            float best = reduceStorageAllBF16(inArray, inSegment, inputShape, inputStrides, input.storageOffset(), input.logicalSize(), isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeBF16(outArray, outSegment, outOffset, TensorDTypeOps.toBFloat16Bits(best));
            return;
        }

        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < output.logicalSize(); outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            float best = reduceStorageFixedBaseBF16(inArray, inSegment, inputBase, reducedStride, reducedSize, isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeBF16(outArray, outSegment, outOffset, TensorDTypeOps.toBFloat16Bits(best));
        }
    }

    private static void executeStorageI32(CpuStorageView input, CpuStorageView output, int dimension, boolean isMax) {
        int[] inArray = ReductionStorageAccess.i32Array(input);
        MemorySegment inSegment = ReductionStorageAccess.i32Segment(input);
        int[] outArray = ReductionStorageAccess.i32Array(output);
        MemorySegment outSegment = ReductionStorageAccess.i32Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();

        if (dimension == -1) {
            int best = reduceStorageAllI32(inArray, inSegment, inputShape, inputStrides, input.storageOffset(), input.logicalSize(), isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeI32(outArray, outSegment, outOffset, best);
            return;
        }

        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < output.logicalSize(); outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            int best = reduceStorageFixedBaseI32(inArray, inSegment, inputBase, reducedStride, reducedSize, isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeI32(outArray, outSegment, outOffset, best);
        }
    }

    private static void executeStorageI64(CpuStorageView input, CpuStorageView output, int dimension, boolean isMax) {
        long[] inArray = ReductionStorageAccess.i64Array(input);
        MemorySegment inSegment = ReductionStorageAccess.i64Segment(input);
        long[] outArray = ReductionStorageAccess.i64Array(output);
        MemorySegment outSegment = ReductionStorageAccess.i64Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();

        if (dimension == -1) {
            long best = reduceStorageAllI64(inArray, inSegment, inputShape, inputStrides, input.storageOffset(), input.logicalSize(), isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeI64(outArray, outSegment, outOffset, best);
            return;
        }

        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < output.logicalSize(); outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            long best = reduceStorageFixedBaseI64(inArray, inSegment, inputBase, reducedStride, reducedSize, isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeI64(outArray, outSegment, outOffset, best);
        }
    }

    private static void executeStorageBOOL(CpuStorageView input, CpuStorageView output, int dimension, boolean isMax) {
        byte[] inArray = ReductionStorageAccess.boolArray(input);
        MemorySegment inSegment = ReductionStorageAccess.boolSegment(input);
        byte[] outArray = ReductionStorageAccess.boolArray(output);
        MemorySegment outSegment = ReductionStorageAccess.boolSegment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();

        if (dimension == -1) {
            byte best = reduceStorageAllBOOL(inArray, inSegment, inputShape, inputStrides, input.storageOffset(), input.logicalSize(), isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeBool(outArray, outSegment, outOffset, best);
            return;
        }

        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < output.logicalSize(); outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            byte best = reduceStorageFixedBaseBOOL(inArray, inSegment, inputBase, reducedStride, reducedSize, isMax);
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeBool(outArray, outSegment, outOffset, best);
        }
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

    private static double reduceContiguousRangeBF16(short[] in, int start, int end, boolean isMax) {
        float best = TensorDTypeOps.fromBFloat16Bits(in[start]);
        for (int i = start + 1; i < end; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in[i]);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static int reduceContiguousRangeI32(int[] in, int start, int end, boolean isMax) {
        int best = in[start];
        for (int i = start + 1; i < end; i++) {
            best = isMax ? Math.max(best, in[i]) : Math.min(best, in[i]);
        }
        return best;
    }

    private static long reduceContiguousRangeI64(long[] in, int start, int end, boolean isMax) {
        long best = in[start];
        for (int i = start + 1; i < end; i++) {
            best = isMax ? Math.max(best, in[i]) : Math.min(best, in[i]);
        }
        return best;
    }

    private static byte reduceContiguousRangeBOOL(byte[] in, int start, int end, boolean isMax) {
        boolean best = in[start] != 0;
        for (int i = start + 1; i < end; i++) {
            boolean value = in[i] != 0;
            best = isMax ? best || value : best && value;
        }
        return best ? (byte) 1 : (byte) 0;
    }

    private static double reduceStridedRangeF64(double[] in, int[] shape, int[] strides, int[] denseStrides, int baseOffset, int start, int end, boolean isMax) {
        double best = in[ReductionTraversal.logicalToOffset(start, shape, strides, denseStrides, baseOffset)];
        for (int logical = start + 1; logical < end; logical++) {
            double value = in[ReductionTraversal.logicalToOffset(logical, shape, strides, denseStrides, baseOffset)];
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static double reduceStridedRangeF32(float[] in, int[] shape, int[] strides, int[] denseStrides, int baseOffset, int start, int end, boolean isMax) {
        float best = in[ReductionTraversal.logicalToOffset(start, shape, strides, denseStrides, baseOffset)];
        for (int logical = start + 1; logical < end; logical++) {
            float value = in[ReductionTraversal.logicalToOffset(logical, shape, strides, denseStrides, baseOffset)];
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static double reduceStridedRangeBF16(short[] in, int[] shape, int[] strides, int[] denseStrides, int baseOffset, int start, int end, boolean isMax) {
        float best = TensorDTypeOps.fromBFloat16Bits(in[ReductionTraversal.logicalToOffset(start, shape, strides, denseStrides, baseOffset)]);
        for (int logical = start + 1; logical < end; logical++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in[ReductionTraversal.logicalToOffset(logical, shape, strides, denseStrides, baseOffset)]);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static int reduceStridedRangeI32(int[] in, int[] shape, int[] strides, int[] denseStrides, int baseOffset, int start, int end, boolean isMax) {
        int best = in[ReductionTraversal.logicalToOffset(start, shape, strides, denseStrides, baseOffset)];
        for (int logical = start + 1; logical < end; logical++) {
            int value = in[ReductionTraversal.logicalToOffset(logical, shape, strides, denseStrides, baseOffset)];
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static long reduceStridedRangeI64(long[] in, int[] shape, int[] strides, int[] denseStrides, int baseOffset, int start, int end, boolean isMax) {
        long best = in[ReductionTraversal.logicalToOffset(start, shape, strides, denseStrides, baseOffset)];
        for (int logical = start + 1; logical < end; logical++) {
            long value = in[ReductionTraversal.logicalToOffset(logical, shape, strides, denseStrides, baseOffset)];
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static byte reduceStridedRangeBOOL(byte[] in, int[] shape, int[] strides, int[] denseStrides, int baseOffset, int start, int end, boolean isMax) {
        boolean best = in[ReductionTraversal.logicalToOffset(start, shape, strides, denseStrides, baseOffset)] != 0;
        for (int logical = start + 1; logical < end; logical++) {
            boolean value = in[ReductionTraversal.logicalToOffset(logical, shape, strides, denseStrides, baseOffset)] != 0;
            best = isMax ? best || value : best && value;
        }
        return best ? (byte) 1 : (byte) 0;
    }

    private static double reduceStorageAllF64(double[] array, MemorySegment segment, int[] shape, int[] strides, int baseOffset, int logicalSize, boolean isMax) {
        int bestOffset = ReductionStorageAccess.logicalToOffset(0, shape, strides, baseOffset);
        double best = ReductionStorageAccess.readF64(array, segment, bestOffset);
        for (int logical = 1; logical < logicalSize; logical++) {
            int offset = ReductionStorageAccess.logicalToOffset(logical, shape, strides, baseOffset);
            double value = ReductionStorageAccess.readF64(array, segment, offset);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static float reduceStorageAllF32(float[] array, MemorySegment segment, int[] shape, int[] strides, int baseOffset, int logicalSize, boolean isMax) {
        int bestOffset = ReductionStorageAccess.logicalToOffset(0, shape, strides, baseOffset);
        float best = ReductionStorageAccess.readF32(array, segment, bestOffset);
        for (int logical = 1; logical < logicalSize; logical++) {
            int offset = ReductionStorageAccess.logicalToOffset(logical, shape, strides, baseOffset);
            float value = ReductionStorageAccess.readF32(array, segment, offset);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static float reduceStorageAllBF16(short[] array, MemorySegment segment, int[] shape, int[] strides, int baseOffset, int logicalSize, boolean isMax) {
        int bestOffset = ReductionStorageAccess.logicalToOffset(0, shape, strides, baseOffset);
        float best = TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(array, segment, bestOffset));
        for (int logical = 1; logical < logicalSize; logical++) {
            int offset = ReductionStorageAccess.logicalToOffset(logical, shape, strides, baseOffset);
            float value = TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(array, segment, offset));
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static int reduceStorageAllI32(int[] array, MemorySegment segment, int[] shape, int[] strides, int baseOffset, int logicalSize, boolean isMax) {
        int bestOffset = ReductionStorageAccess.logicalToOffset(0, shape, strides, baseOffset);
        int best = ReductionStorageAccess.readI32(array, segment, bestOffset);
        for (int logical = 1; logical < logicalSize; logical++) {
            int offset = ReductionStorageAccess.logicalToOffset(logical, shape, strides, baseOffset);
            int value = ReductionStorageAccess.readI32(array, segment, offset);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static long reduceStorageAllI64(long[] array, MemorySegment segment, int[] shape, int[] strides, int baseOffset, int logicalSize, boolean isMax) {
        int bestOffset = ReductionStorageAccess.logicalToOffset(0, shape, strides, baseOffset);
        long best = ReductionStorageAccess.readI64(array, segment, bestOffset);
        for (int logical = 1; logical < logicalSize; logical++) {
            int offset = ReductionStorageAccess.logicalToOffset(logical, shape, strides, baseOffset);
            long value = ReductionStorageAccess.readI64(array, segment, offset);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static byte reduceStorageAllBOOL(byte[] array, MemorySegment segment, int[] shape, int[] strides, int baseOffset, int logicalSize, boolean isMax) {
        int bestOffset = ReductionStorageAccess.logicalToOffset(0, shape, strides, baseOffset);
        boolean best = ReductionStorageAccess.readBool(array, segment, bestOffset) != 0;
        for (int logical = 1; logical < logicalSize; logical++) {
            int offset = ReductionStorageAccess.logicalToOffset(logical, shape, strides, baseOffset);
            boolean value = ReductionStorageAccess.readBool(array, segment, offset) != 0;
            best = isMax ? best || value : best && value;
        }
        return best ? (byte) 1 : (byte) 0;
    }

    private static double reduceStorageFixedBaseF64(double[] array, MemorySegment segment, int baseOffset, int reducedStride, int reducedSize, boolean isMax) {
        double best = ReductionStorageAccess.readF64(array, segment, baseOffset);
        for (int r = 1; r < reducedSize; r++) {
            double value = ReductionStorageAccess.readF64(array, segment, baseOffset + r * reducedStride);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static float reduceStorageFixedBaseF32(float[] array, MemorySegment segment, int baseOffset, int reducedStride, int reducedSize, boolean isMax) {
        float best = ReductionStorageAccess.readF32(array, segment, baseOffset);
        for (int r = 1; r < reducedSize; r++) {
            float value = ReductionStorageAccess.readF32(array, segment, baseOffset + r * reducedStride);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static float reduceStorageFixedBaseBF16(short[] array, MemorySegment segment, int baseOffset, int reducedStride, int reducedSize, boolean isMax) {
        float best = TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(array, segment, baseOffset));
        for (int r = 1; r < reducedSize; r++) {
            float value = TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(array, segment, baseOffset + r * reducedStride));
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static int reduceStorageFixedBaseI32(int[] array, MemorySegment segment, int baseOffset, int reducedStride, int reducedSize, boolean isMax) {
        int best = ReductionStorageAccess.readI32(array, segment, baseOffset);
        for (int r = 1; r < reducedSize; r++) {
            int value = ReductionStorageAccess.readI32(array, segment, baseOffset + r * reducedStride);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static long reduceStorageFixedBaseI64(long[] array, MemorySegment segment, int baseOffset, int reducedStride, int reducedSize, boolean isMax) {
        long best = ReductionStorageAccess.readI64(array, segment, baseOffset);
        for (int r = 1; r < reducedSize; r++) {
            long value = ReductionStorageAccess.readI64(array, segment, baseOffset + r * reducedStride);
            best = isMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    private static byte reduceStorageFixedBaseBOOL(byte[] array, MemorySegment segment, int baseOffset, int reducedStride, int reducedSize, boolean isMax) {
        boolean best = ReductionStorageAccess.readBool(array, segment, baseOffset) != 0;
        for (int r = 1; r < reducedSize; r++) {
            boolean value = ReductionStorageAccess.readBool(array, segment, baseOffset + r * reducedStride) != 0;
            best = isMax ? best || value : best && value;
        }
        return best ? (byte) 1 : (byte) 0;
    }

    private static double mergePartialsF64(double[] partials, boolean isMax) {
        double best = partials[0];
        for (int i = 1; i < partials.length; i++) {
            best = isMax ? Math.max(best, partials[i]) : Math.min(best, partials[i]);
        }
        return best;
    }

    private static int mergePartialsI32(int[] partials, boolean isMax) {
        int best = partials[0];
        for (int i = 1; i < partials.length; i++) {
            best = isMax ? Math.max(best, partials[i]) : Math.min(best, partials[i]);
        }
        return best;
    }

    private static long mergePartialsI64(long[] partials, boolean isMax) {
        long best = partials[0];
        for (int i = 1; i < partials.length; i++) {
            best = isMax ? Math.max(best, partials[i]) : Math.min(best, partials[i]);
        }
        return best;
    }

    private static byte mergePartialsBOOL(byte[] partials, boolean isMax) {
        boolean best = partials[0] != 0;
        for (int i = 1; i < partials.length; i++) {
            boolean value = partials[i] != 0;
            best = isMax ? best || value : best && value;
        }
        return best ? (byte) 1 : (byte) 0;
    }

    private static int[] validateShapeAndDimension(CpuStorageView input, int dimension) {
        int[] shape = input.shape();
        if (shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
        return shape;
    }

    private static void validateOutputSize(CpuStorageView output, int[] inputShape, int logicalSize, int dimension) {
        int expectedOut = dimension == -1 ? 1 : logicalSize / inputShape[dimension];
        if (output.logicalSize() != expectedOut) {
            throw new IllegalArgumentException("Output tensor has wrong size for min/max reduction");
        }
    }

    private static boolean isDenseContiguous(CpuStorageView view) {
        return view.storageOffset() == 0 && isDenseLayout(view.shape(), view.strides());
    }

    private static boolean isExactDenseArrayOutput(CpuStorageView output) {
        return isDenseContiguous(output) && arrayLength(output.requireArray()) == output.logicalSize();
    }

    private static int arrayLength(Object array) {
        if (array instanceof double[] data) {
            return data.length;
        }
        if (array instanceof float[] data) {
            return data.length;
        }
        if (array instanceof short[] data) {
            return data.length;
        }
        if (array instanceof int[] data) {
            return data.length;
        }
        if (array instanceof long[] data) {
            return data.length;
        }
        if (array instanceof byte[] data) {
            return data.length;
        }
        throw new IllegalArgumentException("Unsupported min/max output array type: " + array.getClass().getSimpleName());
    }

    private static boolean isDenseLayout(int[] shape, int[] strides) {
        int expected = 1;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            if (strides[dim] != expected) {
                return false;
            }
            expected *= shape[dim];
        }
        return true;
    }

    private static int axisBaseOffset(
            int outputLogical,
            int[] inputShape,
            int[] inputStrides,
            int inputStorageOffset,
            int[] outputShape,
            int reducedAxis
    ) {
        int remaining = outputLogical;
        int offset = inputStorageOffset;
        if (outputShape.length == inputShape.length) {
            for (int outDim = outputShape.length - 1; outDim >= 0; outDim--) {
                int coord = remaining % outputShape[outDim];
                remaining /= outputShape[outDim];
                if (outDim != reducedAxis) {
                    offset += coord * inputStrides[outDim];
                }
            }
            return offset;
        }
        for (int outDim = outputShape.length - 1; outDim >= 0; outDim--) {
            int coord = remaining % outputShape[outDim];
            remaining /= outputShape[outDim];
            int inputDim = outDim < reducedAxis ? outDim : outDim + 1;
            offset += coord * inputStrides[inputDim];
        }
        return offset;
    }

    private static boolean shouldParallelize(int logicalSize, CpuKernelContext context) {
        ResolvedReductionHints hints = context == null ? null : context.reductionHints();
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
