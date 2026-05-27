package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.plan.reduction.ResolvedReductionHints;
import backend.cpu.plan.CpuExecutionMode;
import backend.cpu.storage.CpuStorageView;
import tensor.dtype.TensorDTypeOps;
import backend.cpu.execution.CpuThreadPool;
import config.backend.SumAccuracyMode;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;

public final class SumLoops {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private SumLoops() {}

    public static void execute(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        int[] shape = validateShapeAndDimension(input, dimension);
        int logicalSize = logicalSize(shape);
        validateOutputSize(output, shape, logicalSize, dimension);

        if (input.isArray() && output.isArray() && isExactDenseArrayOutput(output)) {
            double[] in = input.requireF64Array();
            double[] out = output.requireF64Array();
            if (dimension == -1) {
                out[output.storageOffset()] = sumAll(input, in, logicalSize, context);
                return;
            }
            sumAxis(input, in, out, logicalSize, dimension, context, output.storageOffset());
            return;
        }
        executeStorageF64(input, output, dimension, context);
    }

    public static void executeF32(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        int[] shape = validateShapeAndDimension(input, dimension);
        int logicalSize = logicalSize(shape);
        validateOutputSize(output, shape, logicalSize, dimension);

        if (input.isArray() && output.isArray() && isExactDenseArrayOutput(output)) {
            float[] in = input.requireF32Array();
            float[] out = output.requireF32Array();
            if (dimension == -1) {
                out[output.storageOffset()] = (float) sumAllF32(input, in, logicalSize, context);
                return;
            }
            sumAxisF32(input, in, out, logicalSize, dimension, context, output.storageOffset());
            return;
        }
        executeStorageF32(input, output, dimension, context);
    }

    public static void executeBF16(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        int[] shape = validateShapeAndDimension(input, dimension);
        int logicalSize = logicalSize(shape);
        validateOutputSize(output, shape, logicalSize, dimension);

        if (input.isArray() && output.isArray() && isExactDenseArrayOutput(output)) {
            short[] in = input.requireBF16Array();
            short[] out = output.requireBF16Array();
            if (dimension == -1) {
                out[output.storageOffset()] = TensorDTypeOps.toBFloat16Bits((float) sumAllBF16(input, in, logicalSize, context));
                return;
            }
            sumAxisBF16(input, in, out, logicalSize, dimension, context, output.storageOffset());
            return;
        }
        executeStorageBF16(input, output, dimension, context);
    }

    public static void executeF32ToBF16(CpuStorageView input, float[] data, CpuStorageView output, int dimension, CpuKernelContext context) {
        int[] shape = validateShapeAndDimension(input, dimension);
        int logicalSize = logicalSize(shape);
        validateOutputSize(output, shape, logicalSize, dimension);
        if (data == null || data.length < logicalSize) {
            throw new IllegalArgumentException("Float continuation input is missing or too small");
        }

        if (output.isArray() && isExactDenseArrayOutput(output)) {
            short[] out = output.requireBF16Array();
            if (dimension == -1) {
                out[output.storageOffset()] = TensorDTypeOps.toBFloat16Bits((float) sumAllContiguousF32(data, logicalSize, context));
                return;
            }
            sumAxisF32ContinuationToBF16(input, data, output, dimension, context);
            return;
        }
        executeStorageF32ContinuationToBF16(input, data, output, dimension, context);
    }

    private static double sumAll(CpuStorageView input, double[] data, int logicalSize, CpuKernelContext context) {
        if (!isDenseContiguous(input)) {
            if (logicalSize >= materializeThreshold(context)) {
                return sumAllContiguous(materializeContiguousF64(input, data), logicalSize, context);
            }
            return sumAllStrided(input, data, logicalSize, context);
        }
        return sumAllContiguous(data, input.storageOffset(), input.storageOffset() + logicalSize, context);
    }

    private static double sumAllContiguous(double[] data, int logicalSize, CpuKernelContext context) {
        return sumAllContiguous(data, 0, logicalSize, context);
    }

    private static double sumAllContiguous(double[] data, int start, int end, CpuKernelContext context) {
        CpuExecutionMode mode = reductionMode(context);
        SumAccuracyMode accuracy = reductionAccuracy(context);
        return switch (mode) {
            case SCALAR -> accumulateScalar(data, start, end, accuracy);
            case VECTOR -> (accuracy == SumAccuracyMode.FAST)
                    ? accumulateVectorFast(data, start, end)
                    : accumulateScalar(data, start, end, accuracy);
            case PARALLEL -> parallelSumContiguous(data, start, end, context, false);
            case PARALLEL_VECTOR -> parallelSumContiguous(data, start, end, context, true);
        };
    }

    private static double parallelSumContiguous(double[] data, int logicalSize, CpuKernelContext context, boolean preferVector) {
        return parallelSumContiguous(data, 0, logicalSize, context, preferVector);
    }

    private static double parallelSumContiguous(double[] data, int startOffset, int endOffset, CpuKernelContext context, boolean preferVector) {
        SumAccuracyMode accuracy = reductionAccuracy(context);
        boolean useVector = preferVector && accuracy == SumAccuracyMode.FAST && reductionVectorWidth(context) > 1;
        int chunkSize = reductionChunkSize(context);
        int logicalSize = endOffset - startOffset;
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        double[] partials = new double[chunks];

        CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
            int start = startOffset + chunk * chunkSize;
            int end = Math.min(start + chunkSize, endOffset);
            partials[chunk] = useVector
                    ? accumulateVectorFast(data, start, end)
                    : accumulateScalar(data, start, end, accuracy);
        });

        return mergePartials(partials, accuracy);
    }

    private static double sumAllStrided(CpuStorageView input, double[] data, int logicalSize, CpuKernelContext context) {
        int[] shape = input.shape();
        int[] strides = input.strides();
        int[] denseStrides = denseStrides(shape);
        int baseOffset = input.storageOffset();
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

    private static double sumAllF32(CpuStorageView input, float[] data, int logicalSize, CpuKernelContext context) {
        if (!isDenseContiguous(input)) {
            if (logicalSize >= materializeThreshold(context)) {
                return sumAllContiguousF32(materializeContiguousF32(input, data), logicalSize, context);
            }
            return sumAllStridedF32(input, data, logicalSize, context);
        }
        return sumAllContiguousF32(data, input.storageOffset(), input.storageOffset() + logicalSize, context);
    }

    private static double sumAllContiguousF32(float[] data, int logicalSize, CpuKernelContext context) {
        return sumAllContiguousF32(data, 0, logicalSize, context);
    }

    private static double sumAllContiguousF32(float[] data, int start, int end, CpuKernelContext context) {
        CpuExecutionMode mode = reductionMode(context);
        SumAccuracyMode accuracy = reductionAccuracy(context);
        return switch (mode) {
            case SCALAR -> accumulateScalarF32(data, start, end, accuracy);
            case VECTOR -> (accuracy == SumAccuracyMode.FAST)
                    ? accumulateVectorFastF32(data, start, end)
                    : accumulateScalarF32(data, start, end, accuracy);
            case PARALLEL -> parallelSumContiguousF32(data, start, end, context, false);
            case PARALLEL_VECTOR -> parallelSumContiguousF32(data, start, end, context, true);
        };
    }

    private static double parallelSumContiguousF32(float[] data, int logicalSize, CpuKernelContext context, boolean preferVector) {
        return parallelSumContiguousF32(data, 0, logicalSize, context, preferVector);
    }

    private static double parallelSumContiguousF32(float[] data, int startOffset, int endOffset, CpuKernelContext context, boolean preferVector) {
        SumAccuracyMode accuracy = reductionAccuracy(context);
        boolean useVector = preferVector && accuracy == SumAccuracyMode.FAST && reductionVectorWidth(context) > 1;
        int chunkSize = reductionChunkSize(context);
        int logicalSize = endOffset - startOffset;
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        double[] partials = new double[chunks];

        CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
            int start = startOffset + chunk * chunkSize;
            int end = Math.min(start + chunkSize, endOffset);
            partials[chunk] = useVector
                    ? accumulateVectorFastF32(data, start, end)
                    : accumulateScalarF32(data, start, end, accuracy);
        });
        return mergePartials(partials, accuracy);
    }

    private static double sumAllStridedF32(CpuStorageView input, float[] data, int logicalSize, CpuKernelContext context) {
        int[] shape = input.shape();
        int[] strides = input.strides();
        int[] denseStrides = denseStrides(shape);
        int baseOffset = input.storageOffset();
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

    private static double sumAllBF16(CpuStorageView input, short[] data, int logicalSize, CpuKernelContext context) {
        if (!isDenseContiguous(input)) {
            if (logicalSize >= materializeThreshold(context)) {
                return sumAllContiguousBF16(materializeContiguousBF16(input, data), logicalSize, context);
            }
            return sumAllStridedBF16(input, data, logicalSize, context);
        }
        return sumAllContiguousBF16(data, input.storageOffset(), input.storageOffset() + logicalSize, context);
    }

    private static double sumAllContiguousBF16(short[] data, int logicalSize, CpuKernelContext context) {
        return sumAllContiguousBF16(data, 0, logicalSize, context);
    }

    private static double sumAllContiguousBF16(short[] data, int start, int end, CpuKernelContext context) {
        CpuExecutionMode mode = reductionMode(context);
        SumAccuracyMode accuracy = reductionAccuracy(context);
        return switch (mode) {
            case SCALAR -> accumulateScalarBF16(data, start, end, accuracy);
            case VECTOR -> accuracy == SumAccuracyMode.FAST && reductionVectorWidth(context) > 1
                    ? accumulateVectorFastBF16(data, start, end)
                    : accumulateScalarBF16(data, start, end, accuracy);
            case PARALLEL, PARALLEL_VECTOR -> parallelSumContiguousBF16(data, start, end, context, mode == CpuExecutionMode.PARALLEL_VECTOR);
        };
    }

    private static double parallelSumContiguousBF16(short[] data, int logicalSize, CpuKernelContext context, boolean preferVector) {
        return parallelSumContiguousBF16(data, 0, logicalSize, context, preferVector);
    }

    private static double parallelSumContiguousBF16(short[] data, int startOffset, int endOffset, CpuKernelContext context, boolean preferVector) {
        SumAccuracyMode accuracy = reductionAccuracy(context);
        boolean useVector = preferVector && accuracy == SumAccuracyMode.FAST && reductionVectorWidth(context) > 1;
        int chunkSize = reductionChunkSize(context);
        int logicalSize = endOffset - startOffset;
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        double[] partials = new double[chunks];
        CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
            int start = startOffset + chunk * chunkSize;
            int end = Math.min(start + chunkSize, endOffset);
            partials[chunk] = useVector
                    ? accumulateVectorFastBF16(data, start, end)
                    : accumulateScalarBF16(data, start, end, accuracy);
        });
        return mergePartials(partials, accuracy);
    }

    private static double sumAllStridedBF16(CpuStorageView input, short[] data, int logicalSize, CpuKernelContext context) {
        int[] shape = input.shape();
        int[] strides = input.strides();
        int[] denseStrides = denseStrides(shape);
        int baseOffset = input.storageOffset();
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

    private static void sumAxis(CpuStorageView input, double[] data, double[] out, int logicalSize, int dimension, CpuKernelContext context, int outBaseOffset) {
        int[] shape = input.shape();
        int reducedDim = shape[dimension];
        int outSize = logicalSize / reducedDim;
        if (input.logicalSize() / reducedDim != outSize) {
            throw new IllegalArgumentException("Output length does not match reduction size");
        }

        if (!isDenseContiguous(input)) {
            if (logicalSize >= materializeThreshold(context)) {
                sumAxisContiguous(materializeContiguousF64(input, data), shape, denseStrides(shape), 0, out, outBaseOffset, dimension, context);
                return;
            }
            sumAxisStrided(input, data, out, dimension, context, outBaseOffset);
            return;
        }
        sumAxisContiguous(data, shape, input.strides(), input.storageOffset(), out, outBaseOffset, dimension, context);
    }

    private static void sumAxisContiguous(double[] data, int[] shape, int[] strides, int inputBaseOffset, double[] out, int outBaseOffset, int dimension, CpuKernelContext context) {
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
                    reduceOutputRange(data, out, start, end, shape, strides, inputBaseOffset, dimension, reducedDim, useVector, reductionAccuracy(context), outBaseOffset);
                });
            }
            case VECTOR -> reduceOutputRange(data, out, 0, outSize, shape, strides, inputBaseOffset, dimension, reducedDim, canVectorizeLastDim, reductionAccuracy(context), outBaseOffset);
            case SCALAR -> reduceOutputRange(data, out, 0, outSize, shape, strides, inputBaseOffset, dimension, reducedDim, false, reductionAccuracy(context), outBaseOffset);
        }
    }

    private static void sumAxisStrided(CpuStorageView input, double[] data, double[] out, int dimension, CpuKernelContext context, int outBaseOffset) {
        int[] shape = input.shape();
        int[] strides = input.strides();
        int reducedDim = shape[dimension];
        int outSize = out.length;
        CpuExecutionMode mode = reductionMode(context);

        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = reductionChunkSize(context);
            int chunks = (outSize + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, reductionWorkers(context), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, outSize);
                reduceOutputRange(data, out, start, end, shape, strides, input.storageOffset(), dimension, reducedDim, false, reductionAccuracy(context), outBaseOffset);
            });
            return;
        }
        reduceOutputRange(data, out, 0, outSize, shape, strides, input.storageOffset(), dimension, reducedDim, false, reductionAccuracy(context), outBaseOffset);
    }

    private static void sumAxisF32(CpuStorageView input, float[] data, float[] out, int logicalSize, int dimension, CpuKernelContext context, int outBaseOffset) {
        int[] shape = input.shape();
        int reducedDim = shape[dimension];
        int outSize = logicalSize / reducedDim;
        if (input.logicalSize() / reducedDim != outSize) {
            throw new IllegalArgumentException("Output length does not match reduction size");
        }
        if (!isDenseContiguous(input)) {
            if (logicalSize >= materializeThreshold(context)) {
                sumAxisContiguousF32(materializeContiguousF32(input, data), shape, denseStrides(shape), 0, out, outBaseOffset, dimension, context);
                return;
            }
            sumAxisStridedF32(data, shape, input.strides(), input.storageOffset(), out, outBaseOffset, dimension, context);
            return;
        }
        sumAxisContiguousF32(data, shape, input.strides(), input.storageOffset(), out, outBaseOffset, dimension, context);
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

    private static void sumAxisBF16(CpuStorageView input, short[] data, short[] out, int logicalSize, int dimension, CpuKernelContext context, int outBaseOffset) {
        int[] shape = input.shape();
        int reducedDim = shape[dimension];
        int outSize = logicalSize / reducedDim;
        if (input.logicalSize() / reducedDim != outSize) {
            throw new IllegalArgumentException("Output length does not match reduction size");
        }
        if (!isDenseContiguous(input)) {
            if (logicalSize >= materializeThreshold(context)) {
                sumAxisContiguousBF16(materializeContiguousBF16(input, data), shape, denseStrides(shape), 0, out, outBaseOffset, dimension, context);
                return;
            }
            sumAxisStridedBF16(data, shape, input.strides(), input.storageOffset(), out, outBaseOffset, dimension, context);
            return;
        }
        sumAxisContiguousBF16(data, shape, input.strides(), input.storageOffset(), out, outBaseOffset, dimension, context);
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

    private static void executeStorageF64(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        double[] inArray = ReductionStorageAccess.f64Array(input);
        MemorySegment inSegment = ReductionStorageAccess.f64Segment(input);
        double[] outArray = ReductionStorageAccess.f64Array(output);
        MemorySegment outSegment = ReductionStorageAccess.f64Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        SumAccuracyMode accuracy = reductionAccuracy(context);

        if (dimension == -1) {
            double sum = accumulateStorageF64(inArray, inSegment, inputShape, inputStrides, input.storageOffset(), 0, input.logicalSize(), accuracy);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF64(outArray, outSegment, outOffset, sum);
            return;
        }

        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < output.logicalSize(); outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            double sum = accumulateStorageFixedBaseF64(inArray, inSegment, inputBase, reducedStride, reducedSize, accuracy);
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF64(outArray, outSegment, outOffset, sum);
        }
    }

    private static void executeStorageF32(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        float[] inArray = ReductionStorageAccess.f32Array(input);
        MemorySegment inSegment = ReductionStorageAccess.f32Segment(input);
        float[] outArray = ReductionStorageAccess.f32Array(output);
        MemorySegment outSegment = ReductionStorageAccess.f32Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        SumAccuracyMode accuracy = reductionAccuracy(context);

        if (dimension == -1) {
            double sum = accumulateStorageF32(inArray, inSegment, inputShape, inputStrides, input.storageOffset(), 0, input.logicalSize(), accuracy);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF32(outArray, outSegment, outOffset, (float) sum);
            return;
        }

        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < output.logicalSize(); outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            double sum = accumulateStorageFixedBaseF32(inArray, inSegment, inputBase, reducedStride, reducedSize, accuracy);
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF32(outArray, outSegment, outOffset, (float) sum);
        }
    }

    private static void executeStorageBF16(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        short[] inArray = ReductionStorageAccess.bf16Array(input);
        MemorySegment inSegment = ReductionStorageAccess.bf16Segment(input);
        short[] outArray = ReductionStorageAccess.bf16Array(output);
        MemorySegment outSegment = ReductionStorageAccess.bf16Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        SumAccuracyMode accuracy = reductionAccuracy(context);

        if (dimension == -1) {
            double sum = accumulateStorageBF16(inArray, inSegment, inputShape, inputStrides, input.storageOffset(), 0, input.logicalSize(), accuracy);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeBF16(outArray, outSegment, outOffset, TensorDTypeOps.toBFloat16Bits((float) sum));
            return;
        }

        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < output.logicalSize(); outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            double sum = accumulateStorageFixedBaseBF16(inArray, inSegment, inputBase, reducedStride, reducedSize, accuracy);
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeBF16(outArray, outSegment, outOffset, TensorDTypeOps.toBFloat16Bits((float) sum));
        }
    }

    private static void sumAxisF32ContinuationToBF16(
            CpuStorageView input,
            float[] data,
            CpuStorageView output,
            int dimension,
            CpuKernelContext context
    ) {
        float[] tmp = new float[output.logicalSize()];
        sumAxisContiguousF32(data, input.shape(), input.strides(), 0, tmp, 0, dimension, context);
        short[] out = output.requireBF16Array();
        for (int i = 0; i < output.logicalSize(); i++) {
            out[output.storageOffset() + i] = TensorDTypeOps.toBFloat16Bits(tmp[i]);
        }
    }

    private static void executeStorageF32ContinuationToBF16(
            CpuStorageView input,
            float[] data,
            CpuStorageView output,
            int dimension,
            CpuKernelContext context
    ) {
        short[] outArray = ReductionStorageAccess.bf16Array(output);
        MemorySegment outSegment = ReductionStorageAccess.bf16Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        SumAccuracyMode accuracy = reductionAccuracy(context);

        if (dimension == -1) {
            double sum = sumAllContiguousF32(data, input.logicalSize(), context);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeBF16(outArray, outSegment, outOffset, TensorDTypeOps.toBFloat16Bits((float) sum));
            return;
        }

        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < output.logicalSize(); outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, 0, outputShape, dimension);
            double sum = accumulateStridedFixedBaseF32(data, inputBase, reducedStride, reducedSize, accuracy);
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeBF16(outArray, outSegment, outOffset, TensorDTypeOps.toBFloat16Bits((float) sum));
        }
    }

    private static double accumulateStorageF64(
            double[] data,
            MemorySegment segment,
            int[] shape,
            int[] strides,
            int baseOffset,
            int startLogical,
            int endLogical,
            SumAccuracyMode accuracy
    ) {
        int[] denseStrides = denseStrides(shape);
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0d;
                for (int i = startLogical; i < endLogical; i++) {
                    int offset = logicalToOffset(i, shape, strides, denseStrides, baseOffset);
                    sum += ReductionStorageAccess.readF64(data, segment, offset);
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0d;
                double c = 0.0d;
                for (int i = startLogical; i < endLogical; i++) {
                    int offset = logicalToOffset(i, shape, strides, denseStrides, baseOffset);
                    double y = ReductionStorageAccess.readF64(data, segment, offset) - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0d;
                double c = 0.0d;
                for (int i = startLogical; i < endLogical; i++) {
                    int offset = logicalToOffset(i, shape, strides, denseStrides, baseOffset);
                    double x = ReductionStorageAccess.readF64(data, segment, offset);
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

    private static double accumulateStorageF32(
            float[] data,
            MemorySegment segment,
            int[] shape,
            int[] strides,
            int baseOffset,
            int startLogical,
            int endLogical,
            SumAccuracyMode accuracy
    ) {
        int[] denseStrides = denseStrides(shape);
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0d;
                for (int i = startLogical; i < endLogical; i++) {
                    int offset = logicalToOffset(i, shape, strides, denseStrides, baseOffset);
                    sum += ReductionStorageAccess.readF32(data, segment, offset);
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0d;
                double c = 0.0d;
                for (int i = startLogical; i < endLogical; i++) {
                    int offset = logicalToOffset(i, shape, strides, denseStrides, baseOffset);
                    double y = ReductionStorageAccess.readF32(data, segment, offset) - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0d;
                double c = 0.0d;
                for (int i = startLogical; i < endLogical; i++) {
                    int offset = logicalToOffset(i, shape, strides, denseStrides, baseOffset);
                    double x = ReductionStorageAccess.readF32(data, segment, offset);
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

    private static double accumulateStorageBF16(
            short[] data,
            MemorySegment segment,
            int[] shape,
            int[] strides,
            int baseOffset,
            int startLogical,
            int endLogical,
            SumAccuracyMode accuracy
    ) {
        int[] denseStrides = denseStrides(shape);
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0d;
                for (int i = startLogical; i < endLogical; i++) {
                    int offset = logicalToOffset(i, shape, strides, denseStrides, baseOffset);
                    sum += TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(data, segment, offset));
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0d;
                double c = 0.0d;
                for (int i = startLogical; i < endLogical; i++) {
                    int offset = logicalToOffset(i, shape, strides, denseStrides, baseOffset);
                    double y = TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(data, segment, offset)) - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0d;
                double c = 0.0d;
                for (int i = startLogical; i < endLogical; i++) {
                    int offset = logicalToOffset(i, shape, strides, denseStrides, baseOffset);
                    double x = TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(data, segment, offset));
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

    private static double accumulateStorageFixedBaseF64(
            double[] data,
            MemorySegment segment,
            int base,
            int step,
            int count,
            SumAccuracyMode accuracy
    ) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0d;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    sum += ReductionStorageAccess.readF64(data, segment, idx);
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0d;
                double c = 0.0d;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double y = ReductionStorageAccess.readF64(data, segment, idx) - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0d;
                double c = 0.0d;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double x = ReductionStorageAccess.readF64(data, segment, idx);
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

    private static double accumulateStorageFixedBaseF32(
            float[] data,
            MemorySegment segment,
            int base,
            int step,
            int count,
            SumAccuracyMode accuracy
    ) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0d;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    sum += ReductionStorageAccess.readF32(data, segment, idx);
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0d;
                double c = 0.0d;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double y = ReductionStorageAccess.readF32(data, segment, idx) - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0d;
                double c = 0.0d;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double x = ReductionStorageAccess.readF32(data, segment, idx);
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

    private static double accumulateStorageFixedBaseBF16(
            short[] data,
            MemorySegment segment,
            int base,
            int step,
            int count,
            SumAccuracyMode accuracy
    ) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0d;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    sum += TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(data, segment, idx));
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0d;
                double c = 0.0d;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double y = TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(data, segment, idx)) - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                yield sum;
            }
            case NEUMAIER -> {
                double sum = 0.0d;
                double c = 0.0d;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double x = TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(data, segment, idx));
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
            out[outBaseOffset + outIndex] = TensorDTypeOps.toBFloat16Bits((float) acc);
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
                    sum += TensorDTypeOps.fromBFloat16Bits(data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)]);
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    double y = TensorDTypeOps.fromBFloat16Bits(data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)]) - c;
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
                    double x = TensorDTypeOps.fromBFloat16Bits(data[logicalToOffset(i, shape, strides, denseStrides, baseOffset)]);
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
                for (int i = start; i < end; i++) sum += TensorDTypeOps.fromBFloat16Bits(data[i]);
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = start; i < end; i++) {
                    double y = TensorDTypeOps.fromBFloat16Bits(data[i]) - c;
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
                    double x = TensorDTypeOps.fromBFloat16Bits(data[i]);
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
                lanes[lane] = TensorDTypeOps.fromBFloat16Bits(data[i + lane]);
            }
            vectorAcc = vectorAcc.add(FloatVector.fromArray(FLOAT_SPECIES, lanes, 0));
        }
        double sum = vectorAcc.reduceLanes(VectorOperators.ADD);
        for (; i < end; i++) sum += TensorDTypeOps.fromBFloat16Bits(data[i]);
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
                for (int i = 0; i < count; i++, idx += step) sum += TensorDTypeOps.fromBFloat16Bits(data[idx]);
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double y = TensorDTypeOps.fromBFloat16Bits(data[idx]) - c;
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
                    double x = TensorDTypeOps.fromBFloat16Bits(data[idx]);
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

    private static double[] materializeContiguousF64(CpuStorageView input, double[] src) {
        int[] shape = input.shape();
        int[] strides = input.strides();
        int[] dense = denseStrides(shape);
        int logical = input.logicalSize();
        double[] dst = new double[logical];
        for (int i = 0; i < logical; i++) {
            dst[i] = src[logicalToOffset(i, shape, strides, dense, input.storageOffset())];
        }
        return dst;
    }

    private static float[] materializeContiguousF32(CpuStorageView input, float[] src) {
        int[] shape = input.shape();
        int[] strides = input.strides();
        int[] dense = denseStrides(shape);
        int logical = input.logicalSize();
        float[] dst = new float[logical];
        for (int i = 0; i < logical; i++) {
            dst[i] = src[logicalToOffset(i, shape, strides, dense, input.storageOffset())];
        }
        return dst;
    }

    private static short[] materializeContiguousBF16(CpuStorageView input, short[] src) {
        int[] shape = input.shape();
        int[] strides = input.strides();
        int[] dense = denseStrides(shape);
        int logical = input.logicalSize();
        short[] dst = new short[logical];
        for (int i = 0; i < logical; i++) {
            dst[i] = src[logicalToOffset(i, shape, strides, dense, input.storageOffset())];
        }
        return dst;
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
        return context.contiguousMaterializeThreshold();
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
            throw new IllegalArgumentException("Output tensor has wrong size for sum reduction");
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
        throw new IllegalArgumentException("Unsupported sum output array type: " + array.getClass().getSimpleName());
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
