package backend.kernels.cpu.reduction;

import backend.kernels.cpu.*;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.ResolvedReductionHints;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuThreadPool;
import config.backend.SumAccuracyMode;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorRemap;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class SumLoops {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private SumLoops() {}

    public static void execute(Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        int[] shape = input.getShapeUnsafe();
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }

        int logicalSize = logicalSize(shape);
        int expectedOut = (dimension == -1) ? 1 : (logicalSize / shape[dimension]);
        double[] out = node.getFloat64Data();
        if (out == null || out.length != expectedOut) {
            throw new IllegalArgumentException("Output tensor has wrong size for sum reduction");
        }

        if (dimension == -1) {
            out[node.getStorageOffsetUnsafe()] = sumAll(input, logicalSize, context);
            return;
        }
        sumAxis(input, out, logicalSize, dimension, context, node.getStorageOffsetUnsafe());
    }

    public static void executeF32(Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        int[] shape = input.getShapeUnsafe();
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }

        int logicalSize = logicalSize(shape);
        int expectedOut = (dimension == -1) ? 1 : (logicalSize / shape[dimension]);
        float[] out = node.getFloat32Data();
        if (out == null || out.length != expectedOut) {
            throw new IllegalArgumentException("Output tensor has wrong size for sum reduction");
        }

        float[] in = input.getFloat32Data();
        if (in == null) {
            throw new IllegalStateException("F32 input storage is missing");
        }

        if (dimension == -1) {
            out[0] = (float) sumAllF32(input, in, logicalSize, context);
            return;
        }
        sumAxisF32(input, in, out, logicalSize, dimension, context, node.getStorageOffsetUnsafe());
    }

    public static void executeBF16(Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        int[] shape = input.getShapeUnsafe();
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }

        int logicalSize = logicalSize(shape);
        int expectedOut = (dimension == -1) ? 1 : (logicalSize / shape[dimension]);
        short[] out = node.getBFloat16Data();
        if (out == null || out.length != expectedOut) {
            throw new IllegalArgumentException("Output tensor has wrong size for sum reduction");
        }

        short[] in = input.getBFloat16Data();
        if (in == null) {
            throw new IllegalStateException("BF16 input storage is missing");
        }

        if (dimension == -1) {
            out[0] = CpuDTypeOps.toBFloat16Bits((float) sumAllBF16(input, in, logicalSize, context));
            return;
        }
        sumAxisBF16(input, in, out, logicalSize, dimension, context, node.getStorageOffsetUnsafe());
    }

    public static void executeF32ToBF16(Tensor input, float[] data, Tensor node, int dimension, CpuKernelContext context) {
        int[] shape = input.getShapeUnsafe();
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }

        int logicalSize = logicalSize(shape);
        int expectedOut = (dimension == -1) ? 1 : (logicalSize / shape[dimension]);
        short[] out = node.getBFloat16Data();
        if (out == null || out.length != expectedOut) {
            throw new IllegalArgumentException("Output tensor has wrong size for sum reduction");
        }
        if (data == null || data.length < logicalSize) {
            throw new IllegalArgumentException("Float continuation input is missing or too small");
        }

        if (dimension == -1) {
            out[0] = CpuDTypeOps.toBFloat16Bits((float) sumAllContiguousF32(data, logicalSize, context));
            return;
        }

        float[] tmp = new float[expectedOut];
        sumAxisContiguousF32(data, shape, input.getStridesUnsafe(), 0, tmp, 0, dimension, context);
        int baseOffset = node.getStorageOffsetUnsafe();
        for (int i = 0; i < expectedOut; i++) {
            out[baseOffset + i] = CpuDTypeOps.toBFloat16Bits(tmp[i]);
        }
    }

    private static double sumAll(Tensor input, int logicalSize, CpuKernelContext context) {
        if (!input.isContiguous() || input.hasStorageOffset()) {
            if (logicalSize >= materializeThreshold(context)) {
                Tensor contiguous = materializeContiguous(input);
                return sumAllContiguous(contiguous.getFloat64Data(), logicalSize, context);
            }
            return sumAllStrided(input, logicalSize, context);
        }
        return sumAllContiguous(input.getFloat64Data(), logicalSize, context);
    }

    private static double sumAllContiguous(double[] data, int logicalSize, CpuKernelContext context) {
        CpuExecutionMode mode = reductionMode(context);
        SumAccuracyMode accuracy = reductionAccuracy(context);
        return switch (mode) {
            case SCALAR -> accumulateScalar(data, 0, logicalSize, accuracy);
            case VECTOR -> (accuracy == SumAccuracyMode.FAST)
                    ? accumulateVectorFast(data, 0, logicalSize)
                    : accumulateScalar(data, 0, logicalSize, accuracy);
            case PARALLEL -> parallelSumContiguous(data, logicalSize, context, false);
            case PARALLEL_VECTOR -> parallelSumContiguous(data, logicalSize, context, true);
        };
    }

    private static double parallelSumContiguous(double[] data, int logicalSize, CpuKernelContext context, boolean preferVector) {
        SumAccuracyMode accuracy = reductionAccuracy(context);
        boolean useVector = preferVector && accuracy == SumAccuracyMode.FAST && reductionVectorWidth(context) > 1;
        int chunkSize = reductionChunkSize(context);
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        double[] partials = new double[chunks];

        CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, logicalSize);
            partials[chunk] = useVector
                    ? accumulateVectorFast(data, start, end)
                    : accumulateScalar(data, start, end, accuracy);
        });

        return mergePartials(partials, accuracy);
    }

    private static double sumAllStrided(Tensor input, int logicalSize, CpuKernelContext context) {
        int[] shape = input.getShapeUnsafe();
        int[] strides = input.getStridesUnsafe();
        int[] denseStrides = denseStrides(shape);
        double[] data = input.getFloat64Data();
        int baseOffset = input.getStorageOffsetUnsafe();
        SumAccuracyMode accuracy = reductionAccuracy(context);
        CpuExecutionMode mode = reductionMode(context);

        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = reductionChunkSize(context);
            int chunks = (logicalSize + chunkSize - 1) / chunkSize;
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, logicalSize);
                partials[chunk] = accumulateStridedRange(data, start, end, shape, strides, denseStrides, baseOffset, accuracy);
            });
            return mergePartials(partials, accuracy);
        }

        return accumulateStridedRange(data, 0, logicalSize, shape, strides, denseStrides, baseOffset, accuracy);
    }

    private static double sumAllF32(Tensor input, float[] data, int logicalSize, CpuKernelContext context) {
        if (!input.isContiguous() || input.hasStorageOffset()) {
            if (logicalSize >= materializeThreshold(context)) {
                Tensor contiguous = materializeContiguousTyped(input, context, DataType.FLOAT32);
                float[] c = contiguous.getFloat32Data();
                return sumAllContiguousF32(c, logicalSize, context);
            }
            return sumAllStridedF32(input, data, logicalSize, context);
        }
        return sumAllContiguousF32(data, logicalSize, context);
    }

    private static double sumAllContiguousF32(float[] data, int logicalSize, CpuKernelContext context) {
        CpuExecutionMode mode = reductionMode(context);
        SumAccuracyMode accuracy = reductionAccuracy(context);
        return switch (mode) {
            case SCALAR -> accumulateScalarF32(data, 0, logicalSize, accuracy);
            case VECTOR -> (accuracy == SumAccuracyMode.FAST)
                    ? accumulateVectorFastF32(data, 0, logicalSize)
                    : accumulateScalarF32(data, 0, logicalSize, accuracy);
            case PARALLEL -> parallelSumContiguousF32(data, logicalSize, context, false);
            case PARALLEL_VECTOR -> parallelSumContiguousF32(data, logicalSize, context, true);
        };
    }

    private static double parallelSumContiguousF32(float[] data, int logicalSize, CpuKernelContext context, boolean preferVector) {
        SumAccuracyMode accuracy = reductionAccuracy(context);
        boolean useVector = preferVector && accuracy == SumAccuracyMode.FAST && reductionVectorWidth(context) > 1;
        int chunkSize = reductionChunkSize(context);
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        double[] partials = new double[chunks];

        CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, logicalSize);
            partials[chunk] = useVector
                    ? accumulateVectorFastF32(data, start, end)
                    : accumulateScalarF32(data, start, end, accuracy);
        });
        return mergePartials(partials, accuracy);
    }

    private static double sumAllStridedF32(Tensor input, float[] data, int logicalSize, CpuKernelContext context) {
        int[] shape = input.getShapeUnsafe();
        int[] strides = input.getStridesUnsafe();
        int[] denseStrides = denseStrides(shape);
        int baseOffset = input.getStorageOffsetUnsafe();
        SumAccuracyMode accuracy = reductionAccuracy(context);
        CpuExecutionMode mode = reductionMode(context);

        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = reductionChunkSize(context);
            int chunks = (logicalSize + chunkSize - 1) / chunkSize;
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, logicalSize);
                partials[chunk] = accumulateStridedRangeF32(data, start, end, shape, strides, denseStrides, baseOffset, accuracy);
            });
            return mergePartials(partials, accuracy);
        }
        return accumulateStridedRangeF32(data, 0, logicalSize, shape, strides, denseStrides, baseOffset, accuracy);
    }

    private static double sumAllBF16(Tensor input, short[] data, int logicalSize, CpuKernelContext context) {
        if (!input.isContiguous() || input.hasStorageOffset()) {
            if (logicalSize >= materializeThreshold(context)) {
                Tensor contiguous = materializeContiguousTyped(input, context, DataType.BFLOAT16);
                short[] c = contiguous.getBFloat16Data();
                return sumAllContiguousBF16(c, logicalSize, context);
            }
            return sumAllStridedBF16(input, data, logicalSize, context);
        }
        return sumAllContiguousBF16(data, logicalSize, context);
    }

    private static double sumAllContiguousBF16(short[] data, int logicalSize, CpuKernelContext context) {
        CpuExecutionMode mode = reductionMode(context);
        SumAccuracyMode accuracy = reductionAccuracy(context);
        return switch (mode) {
            case SCALAR -> accumulateScalarBF16(data, 0, logicalSize, accuracy);
            case VECTOR -> accuracy == SumAccuracyMode.FAST && reductionVectorWidth(context) > 1
                    ? accumulateVectorFastBF16(data, 0, logicalSize)
                    : accumulateScalarBF16(data, 0, logicalSize, accuracy);
            case PARALLEL, PARALLEL_VECTOR -> parallelSumContiguousBF16(data, logicalSize, context, mode == CpuExecutionMode.PARALLEL_VECTOR);
        };
    }

    private static double parallelSumContiguousBF16(short[] data, int logicalSize, CpuKernelContext context, boolean preferVector) {
        SumAccuracyMode accuracy = reductionAccuracy(context);
        boolean useVector = preferVector && accuracy == SumAccuracyMode.FAST && reductionVectorWidth(context) > 1;
        int chunkSize = reductionChunkSize(context);
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        double[] partials = new double[chunks];
        CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, logicalSize);
            partials[chunk] = useVector
                    ? accumulateVectorFastBF16(data, start, end)
                    : accumulateScalarBF16(data, start, end, accuracy);
        });
        return mergePartials(partials, accuracy);
    }

    private static double sumAllStridedBF16(Tensor input, short[] data, int logicalSize, CpuKernelContext context) {
        int[] shape = input.getShapeUnsafe();
        int[] strides = input.getStridesUnsafe();
        int[] denseStrides = denseStrides(shape);
        int baseOffset = input.getStorageOffsetUnsafe();
        SumAccuracyMode accuracy = reductionAccuracy(context);
        CpuExecutionMode mode = reductionMode(context);
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = reductionChunkSize(context);
            int chunks = (logicalSize + chunkSize - 1) / chunkSize;
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, logicalSize);
                partials[chunk] = accumulateStridedRangeBF16(data, start, end, shape, strides, denseStrides, baseOffset, accuracy);
            });
            return mergePartials(partials, accuracy);
        }
        return accumulateStridedRangeBF16(data, 0, logicalSize, shape, strides, denseStrides, baseOffset, accuracy);
    }

    private static void sumAxis(Tensor input, double[] out, int logicalSize, int dimension, CpuKernelContext context, int outBaseOffset) {
        int[] shape = input.getShapeUnsafe();
        int reducedDim = shape[dimension];
        int outSize = logicalSize / reducedDim;
        if (out.length != outSize) {
            throw new IllegalArgumentException("Output length does not match reduction size");
        }

        if (!input.isContiguous() || input.hasStorageOffset()) {
            if (logicalSize >= materializeThreshold(context)) {
                Tensor contiguous = materializeContiguous(input);
                sumAxisContiguous(contiguous, out, dimension, context, outBaseOffset);
                return;
            }
            sumAxisStrided(input, out, dimension, context, outBaseOffset);
            return;
        }
        sumAxisContiguous(input, out, dimension, context, outBaseOffset);
    }

    private static void sumAxisContiguous(Tensor input, double[] out, int dimension, CpuKernelContext context, int outBaseOffset) {
        int[] shape = input.getShapeUnsafe();
        int[] strides = input.getStridesUnsafe();
        int reducedDim = shape[dimension];
        int outSize = out.length;
        double[] data = input.getFloat64Data();
        int inputBaseOffset = input.getStorageOffsetUnsafe();

        boolean canVectorizeLastDim = (dimension == shape.length - 1)
                && reductionAccuracy(context) == SumAccuracyMode.FAST
                && reductionVectorWidth(context) > 1;
        CpuExecutionMode mode = reductionMode(context);

        switch (mode) {
            case PARALLEL, PARALLEL_VECTOR -> {
                boolean useVector = mode == CpuExecutionMode.PARALLEL_VECTOR && canVectorizeLastDim;
                int chunkSize = reductionChunkSize(context);
                int chunks = (outSize + chunkSize - 1) / chunkSize;
                CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
                    int start = chunk * chunkSize;
                    int end = Math.min(start + chunkSize, outSize);
                    reduceOutputRange(data, out, start, end, shape, strides, inputBaseOffset, dimension, reducedDim, useVector, reductionAccuracy(context), outBaseOffset);
                });
            }
            case VECTOR -> reduceOutputRange(data, out, 0, outSize, shape, strides, inputBaseOffset, dimension, reducedDim, canVectorizeLastDim, reductionAccuracy(context), outBaseOffset);
            case SCALAR -> reduceOutputRange(data, out, 0, outSize, shape, strides, inputBaseOffset, dimension, reducedDim, false, reductionAccuracy(context), outBaseOffset);
        }
    }

    private static void sumAxisStrided(Tensor input, double[] out, int dimension, CpuKernelContext context, int outBaseOffset) {
        int[] shape = input.getShapeUnsafe();
        int[] strides = input.getStridesUnsafe();
        int reducedDim = shape[dimension];
        int outSize = out.length;
        CpuExecutionMode mode = reductionMode(context);

        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = reductionChunkSize(context);
            int chunks = (outSize + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, outSize);
                reduceOutputRange(input.getFloat64Data(), out, start, end, shape, strides, input.getStorageOffsetUnsafe(), dimension, reducedDim, false, reductionAccuracy(context), outBaseOffset);
            });
            return;
        }
        reduceOutputRange(input.getFloat64Data(), out, 0, outSize, shape, strides, input.getStorageOffsetUnsafe(), dimension, reducedDim, false, reductionAccuracy(context), outBaseOffset);
    }

    private static void sumAxisF32(Tensor input, float[] data, float[] out, int logicalSize, int dimension, CpuKernelContext context, int outBaseOffset) {
        int[] shape = input.getShapeUnsafe();
        int reducedDim = shape[dimension];
        int outSize = logicalSize / reducedDim;
        if (out.length != outSize) {
            throw new IllegalArgumentException("Output length does not match reduction size");
        }
        if (!input.isContiguous() || input.hasStorageOffset()) {
            if (logicalSize >= materializeThreshold(context)) {
                Tensor contiguous = materializeContiguousTyped(input, context, DataType.FLOAT32);
                sumAxisContiguousF32(contiguous.getFloat32Data(), contiguous.getShapeUnsafe(), contiguous.getStridesUnsafe(), contiguous.getStorageOffsetUnsafe(), out, outBaseOffset, dimension, context);
                return;
            }
            sumAxisStridedF32(data, shape, input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), out, outBaseOffset, dimension, context);
            return;
        }
        sumAxisContiguousF32(data, shape, input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), out, outBaseOffset, dimension, context);
    }

    private static void sumAxisContiguousF32(float[] data, int[] shape, int[] strides, int inputBaseOffset, float[] out, int outBaseOffset, int dimension, CpuKernelContext context) {
        int reducedDim = shape[dimension];
        int outSize = out.length;
        boolean canVectorizeLastDim = (dimension == shape.length - 1)
                && reductionAccuracy(context) == SumAccuracyMode.FAST
                && reductionVectorWidth(context) > 1;
        CpuExecutionMode mode = reductionMode(context);

        switch (mode) {
            case PARALLEL, PARALLEL_VECTOR -> {
                boolean useVector = mode == CpuExecutionMode.PARALLEL_VECTOR && canVectorizeLastDim;
                int chunkSize = reductionChunkSize(context);
                int chunks = (outSize + chunkSize - 1) / chunkSize;
                CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
                    int start = chunk * chunkSize;
                    int end = Math.min(start + chunkSize, outSize);
                    reduceOutputRangeF32(data, out, start, end, shape, strides, inputBaseOffset, dimension, reducedDim, useVector, reductionAccuracy(context), outBaseOffset);
                });
            }
            case VECTOR -> reduceOutputRangeF32(data, out, 0, outSize, shape, strides, inputBaseOffset, dimension, reducedDim, canVectorizeLastDim, reductionAccuracy(context), outBaseOffset);
            case SCALAR -> reduceOutputRangeF32(data, out, 0, outSize, shape, strides, inputBaseOffset, dimension, reducedDim, false, reductionAccuracy(context), outBaseOffset);
        }
    }

    private static void sumAxisStridedF32(float[] data, int[] shape, int[] strides, int inputBaseOffset, float[] out, int outBaseOffset, int dimension, CpuKernelContext context) {
        int reducedDim = shape[dimension];
        int outSize = out.length;
        CpuExecutionMode mode = reductionMode(context);
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = reductionChunkSize(context);
            int chunks = (outSize + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, outSize);
                reduceOutputRangeF32(data, out, start, end, shape, strides, inputBaseOffset, dimension, reducedDim, false, reductionAccuracy(context), outBaseOffset);
            });
            return;
        }
        reduceOutputRangeF32(data, out, 0, outSize, shape, strides, inputBaseOffset, dimension, reducedDim, false, reductionAccuracy(context), outBaseOffset);
    }

    private static void sumAxisBF16(Tensor input, short[] data, short[] out, int logicalSize, int dimension, CpuKernelContext context, int outBaseOffset) {
        int[] shape = input.getShapeUnsafe();
        int reducedDim = shape[dimension];
        int outSize = logicalSize / reducedDim;
        if (out.length != outSize) {
            throw new IllegalArgumentException("Output length does not match reduction size");
        }
        if (!input.isContiguous() || input.hasStorageOffset()) {
            if (logicalSize >= materializeThreshold(context)) {
                Tensor contiguous = materializeContiguousTyped(input, context, DataType.BFLOAT16);
                sumAxisContiguousBF16(contiguous.getBFloat16Data(), contiguous.getShapeUnsafe(), contiguous.getStridesUnsafe(), contiguous.getStorageOffsetUnsafe(), out, outBaseOffset, dimension, context);
                return;
            }
            sumAxisStridedBF16(data, shape, input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), out, outBaseOffset, dimension, context);
            return;
        }
        sumAxisContiguousBF16(data, shape, input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), out, outBaseOffset, dimension, context);
    }

    private static void sumAxisContiguousBF16(short[] data, int[] shape, int[] strides, int inputBaseOffset, short[] out, int outBaseOffset, int dimension, CpuKernelContext context) {
        int reducedDim = shape[dimension];
        int outSize = out.length;
        boolean canVectorizeLastDim = (dimension == shape.length - 1)
                && reductionAccuracy(context) == SumAccuracyMode.FAST
                && reductionVectorWidth(context) > 1;
        CpuExecutionMode mode = reductionMode(context);
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = reductionChunkSize(context);
            int chunks = (outSize + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, outSize);
                reduceOutputRangeBF16(
                        data, out, start, end, shape, strides, inputBaseOffset, dimension, reducedDim,
                        mode == CpuExecutionMode.PARALLEL_VECTOR && canVectorizeLastDim,
                        reductionAccuracy(context), outBaseOffset
                );
            });
            return;
        }
        reduceOutputRangeBF16(
                data, out, 0, outSize, shape, strides, inputBaseOffset, dimension, reducedDim,
                mode == CpuExecutionMode.VECTOR && canVectorizeLastDim,
                reductionAccuracy(context), outBaseOffset
        );
    }

    private static void sumAxisStridedBF16(short[] data, int[] shape, int[] strides, int inputBaseOffset, short[] out, int outBaseOffset, int dimension, CpuKernelContext context) {
        int reducedDim = shape[dimension];
        int outSize = out.length;
        CpuExecutionMode mode = reductionMode(context);
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = reductionChunkSize(context);
            int chunks = (outSize + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, outSize);
                reduceOutputRangeBF16(data, out, start, end, shape, strides, inputBaseOffset, dimension, reducedDim, false, reductionAccuracy(context), outBaseOffset);
            });
            return;
        }
        reduceOutputRangeBF16(data, out, 0, outSize, shape, strides, inputBaseOffset, dimension, reducedDim, false, reductionAccuracy(context), outBaseOffset);
    }

    private static void reduceOutputRange(
            double[] data,
            double[] out,
            int fromOut,
            int toOut,
            int[] shape,
            int[] strides,
            int inputBaseOffset,
            int dimension,
            int reducedDim,
            boolean useVectorLastDim,
            SumAccuracyMode accuracy,
            int outBaseOffset
    ) {
        int[] outDenseStrides = denseStridesExcludingDim(shape, dimension);
        int axisStride = strides[dimension];
        int rank = shape.length;

        for (int outIndex = fromOut; outIndex < toOut; outIndex++) {
            int rem = outIndex;
            int base = inputBaseOffset;
            int outAxis = 0;
            for (int d = 0; d < rank; d++) {
                if (d == dimension) continue;
                int coord = rem / outDenseStrides[outAxis];
                rem %= outDenseStrides[outAxis];
                base += coord * strides[d];
                outAxis++;
            }

            double acc;
            if (useVectorLastDim && axisStride == 1) {
                acc = accumulateVectorFast(data, base, base + reducedDim);
            } else {
                acc = accumulateStridedFixedBase(data, base, axisStride, reducedDim, accuracy);
            }
            out[outBaseOffset + outIndex] = acc;
        }
    }

    private static void reduceOutputRangeF32(
            float[] data,
            float[] out,
            int fromOut,
            int toOut,
            int[] shape,
            int[] strides,
            int inputBaseOffset,
            int dimension,
            int reducedDim,
            boolean useVectorLastDim,
            SumAccuracyMode accuracy,
            int outBaseOffset
    ) {
        int[] outDenseStrides = denseStridesExcludingDim(shape, dimension);
        int axisStride = strides[dimension];
        int rank = shape.length;

        for (int outIndex = fromOut; outIndex < toOut; outIndex++) {
            int rem = outIndex;
            int base = inputBaseOffset;
            int outAxis = 0;
            for (int d = 0; d < rank; d++) {
                if (d == dimension) continue;
                int coord = rem / outDenseStrides[outAxis];
                rem %= outDenseStrides[outAxis];
                base += coord * strides[d];
                outAxis++;
            }

            double acc;
            if (useVectorLastDim && axisStride == 1) {
                acc = accumulateVectorFastF32(data, base, base + reducedDim);
            } else {
                acc = accumulateStridedFixedBaseF32(data, base, axisStride, reducedDim, accuracy);
            }
            out[outBaseOffset + outIndex] = (float) acc;
        }
    }

    private static void reduceOutputRangeBF16(
            short[] data,
            short[] out,
            int fromOut,
            int toOut,
            int[] shape,
            int[] strides,
            int inputBaseOffset,
            int dimension,
            int reducedDim,
            boolean useVectorLastDim,
            SumAccuracyMode accuracy,
            int outBaseOffset
    ) {
        int[] outDenseStrides = denseStridesExcludingDim(shape, dimension);
        int axisStride = strides[dimension];
        int rank = shape.length;

        for (int outIndex = fromOut; outIndex < toOut; outIndex++) {
            int rem = outIndex;
            int base = inputBaseOffset;
            int outAxis = 0;
            for (int d = 0; d < rank; d++) {
                if (d == dimension) continue;
                int coord = rem / outDenseStrides[outAxis];
                rem %= outDenseStrides[outAxis];
                base += coord * strides[d];
                outAxis++;
            }
            double acc = useVectorLastDim && axisStride == 1
                    ? accumulateVectorFastBF16(data, base, base + reducedDim)
                    : accumulateStridedFixedBaseBF16(data, base, axisStride, reducedDim, accuracy);
            out[outBaseOffset + outIndex] = CpuDTypeOps.toBFloat16Bits((float) acc);
        }
    }

    private static double accumulateStridedFixedBase(double[] data, int base, int step, int count, SumAccuracyMode accuracy) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) sum += data[idx];
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double y = data[idx] - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0;
                double c = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double x = data[idx];
                    double t = sum + x;
                    if (Math.abs(sum) >= Math.abs(x)) {
                        c += (sum - t) + x;
                    } else {
                        c += (x - t) + sum;
                    }
                    sum = t;
                }
                yield sum + c;
            }
        };
    }

    private static double accumulateStridedRange(
            double[] data,
            int startLogical,
            int endLogical,
            int[] shape,
            int[] strides,
            int[] denseStrides,
            int baseOffset,
            SumAccuracyMode accuracy
    ) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    sum += data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)];
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    double y = data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)] - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    double x = data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)];
                    double t = sum + x;
                    if (Math.abs(sum) >= Math.abs(x)) {
                        c += (sum - t) + x;
                    } else {
                        c += (x - t) + sum;
                    }
                    sum = t;
                }
                yield sum + c;
            }
        };
    }

    private static double accumulateStridedRangeF32(
            float[] data,
            int startLogical,
            int endLogical,
            int[] shape,
            int[] strides,
            int[] denseStrides,
            int baseOffset,
            SumAccuracyMode accuracy
    ) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    sum += data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)];
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    double y = data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)] - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    double x = data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)];
                    double t = sum + x;
                    if (Math.abs(sum) >= Math.abs(x)) {
                        c += (sum - t) + x;
                    } else {
                        c += (x - t) + sum;
                    }
                    sum = t;
                }
                yield sum + c;
            }
        };
    }

    private static double accumulateStridedRangeBF16(
            short[] data,
            int startLogical,
            int endLogical,
            int[] shape,
            int[] strides,
            int[] denseStrides,
            int baseOffset,
            SumAccuracyMode accuracy
    ) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    sum += CpuDTypeOps.fromBFloat16Bits(data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)]);
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    double y = CpuDTypeOps.fromBFloat16Bits(data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)]) - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    double x = CpuDTypeOps.fromBFloat16Bits(data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)]);
                    double t = sum + x;
                    if (Math.abs(sum) >= Math.abs(x)) {
                        c += (sum - t) + x;
                    } else {
                        c += (x - t) + sum;
                    }
                    sum = t;
                }
                yield sum + c;
            }
        };
    }

    private static double accumulateScalar(double[] data, int start, int end, SumAccuracyMode accuracy) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                for (int i = start; i < end; i++) sum += data[i];
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = start; i < end; i++) {
                    double y = data[i] - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = start; i < end; i++) {
                    double x = data[i];
                    double t = sum + x;
                    if (Math.abs(sum) >= Math.abs(x)) {
                        c += (sum - t) + x;
                    } else {
                        c += (x - t) + sum;
                    }
                    sum = t;
                }
                yield sum + c;
            }
        };
    }

    private static double accumulateScalarF32(float[] data, int start, int end, SumAccuracyMode accuracy) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                for (int i = start; i < end; i++) sum += data[i];
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = start; i < end; i++) {
                    double y = data[i] - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = start; i < end; i++) {
                    double x = data[i];
                    double t = sum + x;
                    if (Math.abs(sum) >= Math.abs(x)) {
                        c += (sum - t) + x;
                    } else {
                        c += (x - t) + sum;
                    }
                    sum = t;
                }
                yield sum + c;
            }
        };
    }

    private static double accumulateScalarBF16(short[] data, int start, int end, SumAccuracyMode accuracy) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                for (int i = start; i < end; i++) sum += CpuDTypeOps.fromBFloat16Bits(data[i]);
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = start; i < end; i++) {
                    double y = CpuDTypeOps.fromBFloat16Bits(data[i]) - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = start; i < end; i++) {
                    double x = CpuDTypeOps.fromBFloat16Bits(data[i]);
                    double t = sum + x;
                    if (Math.abs(sum) >= Math.abs(x)) {
                        c += (sum - t) + x;
                    } else {
                        c += (x - t) + sum;
                    }
                    sum = t;
                }
                yield sum + c;
            }
        };
    }

    private static double accumulateVectorFast(double[] data, int start, int end) {
        int width = SPECIES.length();
        int i = start;
        int upper = end - ((end - start) % width);
        DoubleVector vectorAcc = DoubleVector.zero(SPECIES);
        for (; i < upper; i += width) {
            vectorAcc = vectorAcc.add(DoubleVector.fromArray(SPECIES, data, i));
        }
        double sum = vectorAcc.reduceLanes(VectorOperators.ADD);
        for (; i < end; i++) sum += data[i];
        return sum;
    }

    private static double accumulateVectorFastF32(float[] data, int start, int end) {
        int width = FLOAT_SPECIES.length();
        int i = start;
        int upper = end - ((end - start) % width);
        FloatVector vectorAcc = FloatVector.zero(FLOAT_SPECIES);
        for (; i < upper; i += width) {
            vectorAcc = vectorAcc.add(FloatVector.fromArray(FLOAT_SPECIES, data, i));
        }
        double sum = vectorAcc.reduceLanes(VectorOperators.ADD);
        for (; i < end; i++) sum += data[i];
        return sum;
    }

    private static double accumulateVectorFastBF16(short[] data, int start, int end) {
        int width = FLOAT_SPECIES.length();
        int i = start;
        int upper = end - ((end - start) % width);
        FloatVector vectorAcc = FloatVector.zero(FLOAT_SPECIES);
        float[] lanes = new float[width];
        for (; i < upper; i += width) {
            for (int lane = 0; lane < width; lane++) {
                lanes[lane] = CpuDTypeOps.fromBFloat16Bits(data[i + lane]);
            }
            vectorAcc = vectorAcc.add(FloatVector.fromArray(FLOAT_SPECIES, lanes, 0));
        }
        double sum = vectorAcc.reduceLanes(VectorOperators.ADD);
        for (; i < end; i++) sum += CpuDTypeOps.fromBFloat16Bits(data[i]);
        return sum;
    }

    private static double accumulateStridedFixedBaseF32(float[] data, int base, int step, int count, SumAccuracyMode accuracy) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) sum += data[idx];
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double y = data[idx] - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0;
                double c = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double x = data[idx];
                    double t = sum + x;
                    if (Math.abs(sum) >= Math.abs(x)) {
                        c += (sum - t) + x;
                    } else {
                        c += (x - t) + sum;
                    }
                    sum = t;
                }
                yield sum + c;
            }
        };
    }

    private static double accumulateStridedFixedBaseBF16(short[] data, int base, int step, int count, SumAccuracyMode accuracy) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) sum += CpuDTypeOps.fromBFloat16Bits(data[idx]);
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double y = CpuDTypeOps.fromBFloat16Bits(data[idx]) - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0;
                double c = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double x = CpuDTypeOps.fromBFloat16Bits(data[idx]);
                    double t = sum + x;
                    if (Math.abs(sum) >= Math.abs(x)) {
                        c += (sum - t) + x;
                    } else {
                        c += (x - t) + sum;
                    }
                    sum = t;
                }
                yield sum + c;
            }
        };
    }

    private static double mergePartials(double[] partials, SumAccuracyMode accuracy) {
        if (partials.length == 0) return 0.0;
        return switch (accuracy) {
            case FAST -> pairwiseFast(partials, 0, partials.length);
            case KAHAN, NEUMAIER -> accumulateScalar(partials, 0, partials.length, accuracy);
        };
    }

    private static double pairwiseFast(double[] values, int from, int to) {
        int len = to - from;
        if (len <= 32) {
            double sum = 0.0;
            for (int i = from; i < to; i++) sum += values[i];
            return sum;
        }
        int mid = from + (len >>> 1);
        return pairwiseFast(values, from, mid) + pairwiseFast(values, mid, to);
    }

    private static Tensor materializeContiguous(Tensor input) {
        Tensor contiguous = new Tensor(input.getShapeUnsafe(), null, input.getLabel() + "_sum_contiguous_tmp", DataType.FLOAT64);
        double[] src = input.getFloat64Data();
        double[] dst = contiguous.getData();
        int[] shape = input.getShapeUnsafe();
        int[] strides = input.getStridesUnsafe();
        int[] dense = denseStrides(shape);
        int baseOffset = input.getStorageOffsetUnsafe();
        int logical = logicalSize(shape);
        for (int i = 0; i < logical; i++) {
            dst[i] = src[logicalToOffset(i, shape, strides, dense, baseOffset)];
        }
        return contiguous;
    }

    private static Tensor materializeContiguousTyped(Tensor input, CpuKernelContext context, DataType dataType) {
        Tensor contiguous = new Tensor(input.getShapeUnsafe(), null, input.getLabel() + "_sum_contiguous_tmp", dataType);
        TensorRemap.apply(input, contiguous, materializeThreshold(context));
        return contiguous;
    }

    private static ResolvedReductionHints requireHints(CpuKernelContext context) {
        ResolvedReductionHints hints = context.reductionHints();
        if (hints == null) {
            throw new IllegalStateException("Missing ResolvedReductionHints for sum execution");
        }
        return hints;
    }

    private static CpuExecutionMode reductionMode(CpuKernelContext context) {
        return requireHints(context).mode();
    }

    private static SumAccuracyMode reductionAccuracy(CpuKernelContext context) {
        return requireHints(context).accuracyMode();
    }

    private static int reductionChunkSize(CpuKernelContext context) {
        return requireHints(context).chunkSize();
    }

    private static int reductionWorkers(CpuKernelContext context) {
        return requireHints(context).plannedWorkers();
    }

    private static int reductionVectorWidth(CpuKernelContext context) {
        return requireHints(context).vectorWidth();
    }

    private static int materializeThreshold(CpuKernelContext context) {
        return context.planner().contiguousMaterializeThreshold();
    }

    private static int logicalToOffset(int logicalIndex, int[] shape, int[] strides, int[] denseStrides, int baseOffset) {
        int rank = shape.length;
        int rem = logicalIndex;
        int offset = baseOffset;
        for (int d = 0; d < rank; d++) {
            int coord = rem / denseStrides[d];
            rem %= denseStrides[d];
            offset += coord * strides[d];
        }
        return offset;
    }

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static int[] denseStrides(int[] shape) {
        int[] out = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            out[i] = stride;
            stride *= shape[i];
        }
        return out;
    }

    private static int[] denseStridesExcludingDim(int[] shape, int excludedDim) {
        int outRank = shape.length - 1;
        int[] outShape = new int[outRank];
        int idx = 0;
        for (int d = 0; d < shape.length; d++) {
            if (d == excludedDim) continue;
            outShape[idx++] = shape[d];
        }
        return denseStrides(outShape);
    }
}
