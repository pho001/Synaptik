package Backend.kernels.cpu.reduction;

import Backend.kernels.cpu.CpuExecutionConfig;
import Backend.kernels.cpu.CpuExecutionMode;
import Backend.kernels.cpu.CpuDTypeOps;
import Backend.kernels.cpu.CpuThreadPool;
import Config.backend.SumAccuracyMode;
import Tensor.DataType;
import Tensor.Tensor;
import Tensor.TensorRemap;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

final class SumLoops {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private SumLoops() {}

    static void execute(Tensor input, Tensor node, int dimension, CpuExecutionConfig config) {
        int[] shape = input.getShape();
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }

        int logicalSize = logicalSize(shape);
        int expectedOut = (dimension == -1) ? 1 : (logicalSize / shape[dimension]);
        if (node.getData().length != expectedOut) {
            throw new IllegalArgumentException("Output tensor has wrong size for sum reduction");
        }

        if (dimension == -1) {
            node.getData()[0] = sumAll(input, logicalSize, config);
            return;
        }
        sumAxis(input, node.getData(), logicalSize, dimension, config);
    }

    static void executeF32(Tensor input, Tensor node, int dimension, CpuExecutionConfig config) {
        int[] shape = input.getShape();
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
            out[0] = (float) sumAllF32(input, in, logicalSize, config);
            return;
        }
        sumAxisF32(input, in, out, logicalSize, dimension, config);
    }

    static void executeF16(Tensor input, Tensor node, int dimension, CpuExecutionConfig config) {
        int[] shape = input.getShape();
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }

        int logicalSize = logicalSize(shape);
        int expectedOut = (dimension == -1) ? 1 : (logicalSize / shape[dimension]);
        short[] out = node.getFloat16Data();
        if (out == null || out.length != expectedOut) {
            throw new IllegalArgumentException("Output tensor has wrong size for sum reduction");
        }

        short[] in = input.getFloat16Data();
        if (in == null) {
            throw new IllegalStateException("F16 input storage is missing");
        }

        if (dimension == -1) {
            out[0] = CpuDTypeOps.toHalfBits((float) sumAllF16(input, in, logicalSize, config));
            return;
        }
        sumAxisF16(input, in, out, logicalSize, dimension, config);
    }

    private static double sumAll(Tensor input, int logicalSize, CpuExecutionConfig config) {
        if (!input.isContiguous()) {
            if (logicalSize >= config.contiguousMaterializeThreshold()) {
                Tensor contiguous = materializeContiguous(input, config);
                return sumAllContiguous(contiguous.getData(), logicalSize, config);
            }
            return sumAllStrided(input, logicalSize, config);
        }
        return sumAllContiguous(input.getData(), logicalSize, config);
    }

    private static double sumAllContiguous(double[] data, int logicalSize, CpuExecutionConfig config) {
        CpuExecutionMode mode = config.modeForReduction(logicalSize);
        SumAccuracyMode accuracy = config.sumAccuracyMode();
        return switch (mode) {
            case SCALAR -> accumulateScalar(data, 0, logicalSize, accuracy);
            case VECTOR -> (accuracy == SumAccuracyMode.FAST)
                    ? accumulateVectorFast(data, 0, logicalSize)
                    : accumulateScalar(data, 0, logicalSize, accuracy);
            case PARALLEL -> parallelSumContiguous(data, logicalSize, config, false);
            case PARALLEL_VECTOR -> parallelSumContiguous(data, logicalSize, config, true);
        };
    }

    private static double parallelSumContiguous(double[] data, int logicalSize, CpuExecutionConfig config, boolean preferVector) {
        SumAccuracyMode accuracy = config.sumAccuracyMode();
        boolean useVector = preferVector && accuracy == SumAccuracyMode.FAST;
        int vectorWidth = useVector ? SPECIES.length() : 1;
        int chunkSize = config.computeChunkSize(logicalSize, vectorWidth);
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        double[] partials = new double[chunks];

        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, logicalSize);
            partials[chunk] = useVector
                    ? accumulateVectorFast(data, start, end)
                    : accumulateScalar(data, start, end, accuracy);
        });

        return mergePartials(partials, accuracy);
    }

    private static double sumAllStrided(Tensor input, int logicalSize, CpuExecutionConfig config) {
        int[] shape = input.getShape();
        int[] strides = input.getStrides();
        int[] denseStrides = denseStrides(shape);
        double[] data = input.getData();
        SumAccuracyMode accuracy = config.sumAccuracyMode();
        CpuExecutionMode mode = config.modeForReduction(logicalSize);

        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = config.computeChunkSize(logicalSize, 1);
            int chunks = (logicalSize + chunkSize - 1) / chunkSize;
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, logicalSize);
                partials[chunk] = accumulateStridedRange(data, start, end, shape, strides, denseStrides, accuracy);
            });
            return mergePartials(partials, accuracy);
        }

        return accumulateStridedRange(data, 0, logicalSize, shape, strides, denseStrides, accuracy);
    }

    private static double sumAllF32(Tensor input, float[] data, int logicalSize, CpuExecutionConfig config) {
        if (!input.isContiguous()) {
            if (logicalSize >= config.contiguousMaterializeThreshold()) {
                Tensor contiguous = materializeContiguousTyped(input, config, DataType.FLOAT32);
                float[] c = contiguous.getFloat32Data();
                return sumAllContiguousF32(c, logicalSize, config);
            }
            return sumAllStridedF32(input, data, logicalSize, config);
        }
        return sumAllContiguousF32(data, logicalSize, config);
    }

    private static double sumAllContiguousF32(float[] data, int logicalSize, CpuExecutionConfig config) {
        CpuExecutionMode mode = config.modeForReduction(logicalSize);
        SumAccuracyMode accuracy = config.sumAccuracyMode();
        return switch (mode) {
            case SCALAR -> accumulateScalarF32(data, 0, logicalSize, accuracy);
            case VECTOR -> (accuracy == SumAccuracyMode.FAST)
                    ? accumulateVectorFastF32(data, 0, logicalSize)
                    : accumulateScalarF32(data, 0, logicalSize, accuracy);
            case PARALLEL -> parallelSumContiguousF32(data, logicalSize, config, false);
            case PARALLEL_VECTOR -> parallelSumContiguousF32(data, logicalSize, config, true);
        };
    }

    private static double parallelSumContiguousF32(float[] data, int logicalSize, CpuExecutionConfig config, boolean preferVector) {
        SumAccuracyMode accuracy = config.sumAccuracyMode();
        boolean useVector = preferVector && accuracy == SumAccuracyMode.FAST;
        int vectorWidth = useVector ? FLOAT_SPECIES.length() : 1;
        int chunkSize = config.computeChunkSize(logicalSize, vectorWidth);
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        double[] partials = new double[chunks];

        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, logicalSize);
            partials[chunk] = useVector
                    ? accumulateVectorFastF32(data, start, end)
                    : accumulateScalarF32(data, start, end, accuracy);
        });
        return mergePartials(partials, accuracy);
    }

    private static double sumAllStridedF32(Tensor input, float[] data, int logicalSize, CpuExecutionConfig config) {
        int[] shape = input.getShape();
        int[] strides = input.getStrides();
        int[] denseStrides = denseStrides(shape);
        SumAccuracyMode accuracy = config.sumAccuracyMode();
        CpuExecutionMode mode = config.modeForReduction(logicalSize);

        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = config.computeChunkSize(logicalSize, 1);
            int chunks = (logicalSize + chunkSize - 1) / chunkSize;
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, logicalSize);
                partials[chunk] = accumulateStridedRangeF32(data, start, end, shape, strides, denseStrides, accuracy);
            });
            return mergePartials(partials, accuracy);
        }
        return accumulateStridedRangeF32(data, 0, logicalSize, shape, strides, denseStrides, accuracy);
    }

    private static double sumAllF16(Tensor input, short[] data, int logicalSize, CpuExecutionConfig config) {
        if (!input.isContiguous()) {
            if (logicalSize >= config.contiguousMaterializeThreshold()) {
                Tensor contiguous = materializeContiguousTyped(input, config, DataType.FLOAT16);
                short[] c = contiguous.getFloat16Data();
                return sumAllContiguousF16(c, logicalSize, config);
            }
            return sumAllStridedF16(input, data, logicalSize, config);
        }
        return sumAllContiguousF16(data, logicalSize, config);
    }

    private static double sumAllContiguousF16(short[] data, int logicalSize, CpuExecutionConfig config) {
        CpuExecutionMode mode = config.modeForReduction(logicalSize);
        SumAccuracyMode accuracy = config.sumAccuracyMode();
        return switch (mode) {
            case SCALAR, VECTOR -> accumulateScalarF16(data, 0, logicalSize, accuracy);
            case PARALLEL, PARALLEL_VECTOR -> parallelSumContiguousF16(data, logicalSize, config);
        };
    }

    private static double parallelSumContiguousF16(short[] data, int logicalSize, CpuExecutionConfig config) {
        SumAccuracyMode accuracy = config.sumAccuracyMode();
        int chunkSize = config.computeChunkSize(logicalSize, 1);
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        double[] partials = new double[chunks];
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, logicalSize);
            partials[chunk] = accumulateScalarF16(data, start, end, accuracy);
        });
        return mergePartials(partials, accuracy);
    }

    private static double sumAllStridedF16(Tensor input, short[] data, int logicalSize, CpuExecutionConfig config) {
        int[] shape = input.getShape();
        int[] strides = input.getStrides();
        int[] denseStrides = denseStrides(shape);
        SumAccuracyMode accuracy = config.sumAccuracyMode();
        CpuExecutionMode mode = config.modeForReduction(logicalSize);
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = config.computeChunkSize(logicalSize, 1);
            int chunks = (logicalSize + chunkSize - 1) / chunkSize;
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, logicalSize);
                partials[chunk] = accumulateStridedRangeF16(data, start, end, shape, strides, denseStrides, accuracy);
            });
            return mergePartials(partials, accuracy);
        }
        return accumulateStridedRangeF16(data, 0, logicalSize, shape, strides, denseStrides, accuracy);
    }

    private static void sumAxis(Tensor input, double[] out, int logicalSize, int dimension, CpuExecutionConfig config) {
        int[] shape = input.getShape();
        int reducedDim = shape[dimension];
        int outSize = logicalSize / reducedDim;
        if (out.length != outSize) {
            throw new IllegalArgumentException("Output length does not match reduction size");
        }

        if (!input.isContiguous()) {
            if (logicalSize >= config.contiguousMaterializeThreshold()) {
                Tensor contiguous = materializeContiguous(input, config);
                sumAxisContiguous(contiguous, out, dimension, config);
                return;
            }
            sumAxisStrided(input, out, dimension, config);
            return;
        }
        sumAxisContiguous(input, out, dimension, config);
    }

    private static void sumAxisContiguous(Tensor input, double[] out, int dimension, CpuExecutionConfig config) {
        int[] shape = input.getShape();
        int[] strides = input.getStrides();
        int reducedDim = shape[dimension];
        int outSize = out.length;
        double[] data = input.getData();

        boolean canVectorizeLastDim = (dimension == shape.length - 1) && config.sumAccuracyMode() == SumAccuracyMode.FAST;
        CpuExecutionMode mode = config.modeForReduction(logicalSize(shape));

        switch (mode) {
            case PARALLEL, PARALLEL_VECTOR -> {
                boolean useVector = mode == CpuExecutionMode.PARALLEL_VECTOR && canVectorizeLastDim;
                int vectorWidth = useVector ? SPECIES.length() : 1;
                int chunkSize = config.computeChunkSize(outSize, vectorWidth);
                int chunks = (outSize + chunkSize - 1) / chunkSize;
                CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
                    int start = chunk * chunkSize;
                    int end = Math.min(start + chunkSize, outSize);
                    reduceOutputRange(data, out, start, end, shape, strides, dimension, reducedDim, useVector, config.sumAccuracyMode());
                });
            }
            case VECTOR -> reduceOutputRange(data, out, 0, outSize, shape, strides, dimension, reducedDim, canVectorizeLastDim, config.sumAccuracyMode());
            case SCALAR -> reduceOutputRange(data, out, 0, outSize, shape, strides, dimension, reducedDim, false, config.sumAccuracyMode());
        }
    }

    private static void sumAxisStrided(Tensor input, double[] out, int dimension, CpuExecutionConfig config) {
        int[] shape = input.getShape();
        int[] strides = input.getStrides();
        int reducedDim = shape[dimension];
        int outSize = out.length;
        CpuExecutionMode mode = config.modeForReduction(logicalSize(shape));

        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = config.computeChunkSize(outSize, 1);
            int chunks = (outSize + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, outSize);
                reduceOutputRange(input.getData(), out, start, end, shape, strides, dimension, reducedDim, false, config.sumAccuracyMode());
            });
            return;
        }
        reduceOutputRange(input.getData(), out, 0, outSize, shape, strides, dimension, reducedDim, false, config.sumAccuracyMode());
    }

    private static void sumAxisF32(Tensor input, float[] data, float[] out, int logicalSize, int dimension, CpuExecutionConfig config) {
        int[] shape = input.getShape();
        int reducedDim = shape[dimension];
        int outSize = logicalSize / reducedDim;
        if (out.length != outSize) {
            throw new IllegalArgumentException("Output length does not match reduction size");
        }
        if (!input.isContiguous()) {
            if (logicalSize >= config.contiguousMaterializeThreshold()) {
                Tensor contiguous = materializeContiguousTyped(input, config, DataType.FLOAT32);
                sumAxisContiguousF32(contiguous.getFloat32Data(), contiguous.getShape(), contiguous.getStrides(), out, dimension, config);
                return;
            }
            sumAxisStridedF32(data, shape, input.getStrides(), out, dimension, config);
            return;
        }
        sumAxisContiguousF32(data, shape, input.getStrides(), out, dimension, config);
    }

    private static void sumAxisContiguousF32(float[] data, int[] shape, int[] strides, float[] out, int dimension, CpuExecutionConfig config) {
        int reducedDim = shape[dimension];
        int outSize = out.length;
        boolean canVectorizeLastDim = (dimension == shape.length - 1) && config.sumAccuracyMode() == SumAccuracyMode.FAST;
        CpuExecutionMode mode = config.modeForReduction(logicalSize(shape));

        switch (mode) {
            case PARALLEL, PARALLEL_VECTOR -> {
                boolean useVector = mode == CpuExecutionMode.PARALLEL_VECTOR && canVectorizeLastDim;
                int vectorWidth = useVector ? FLOAT_SPECIES.length() : 1;
                int chunkSize = config.computeChunkSize(outSize, vectorWidth);
                int chunks = (outSize + chunkSize - 1) / chunkSize;
                CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
                    int start = chunk * chunkSize;
                    int end = Math.min(start + chunkSize, outSize);
                    reduceOutputRangeF32(data, out, start, end, shape, strides, dimension, reducedDim, useVector, config.sumAccuracyMode());
                });
            }
            case VECTOR -> reduceOutputRangeF32(data, out, 0, outSize, shape, strides, dimension, reducedDim, canVectorizeLastDim, config.sumAccuracyMode());
            case SCALAR -> reduceOutputRangeF32(data, out, 0, outSize, shape, strides, dimension, reducedDim, false, config.sumAccuracyMode());
        }
    }

    private static void sumAxisStridedF32(float[] data, int[] shape, int[] strides, float[] out, int dimension, CpuExecutionConfig config) {
        int reducedDim = shape[dimension];
        int outSize = out.length;
        CpuExecutionMode mode = config.modeForReduction(logicalSize(shape));
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = config.computeChunkSize(outSize, 1);
            int chunks = (outSize + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, outSize);
                reduceOutputRangeF32(data, out, start, end, shape, strides, dimension, reducedDim, false, config.sumAccuracyMode());
            });
            return;
        }
        reduceOutputRangeF32(data, out, 0, outSize, shape, strides, dimension, reducedDim, false, config.sumAccuracyMode());
    }

    private static void sumAxisF16(Tensor input, short[] data, short[] out, int logicalSize, int dimension, CpuExecutionConfig config) {
        int[] shape = input.getShape();
        int reducedDim = shape[dimension];
        int outSize = logicalSize / reducedDim;
        if (out.length != outSize) {
            throw new IllegalArgumentException("Output length does not match reduction size");
        }
        if (!input.isContiguous()) {
            if (logicalSize >= config.contiguousMaterializeThreshold()) {
                Tensor contiguous = materializeContiguousTyped(input, config, DataType.FLOAT16);
                sumAxisContiguousF16(contiguous.getFloat16Data(), contiguous.getShape(), contiguous.getStrides(), out, dimension, config);
                return;
            }
            sumAxisStridedF16(data, shape, input.getStrides(), out, dimension, config);
            return;
        }
        sumAxisContiguousF16(data, shape, input.getStrides(), out, dimension, config);
    }

    private static void sumAxisContiguousF16(short[] data, int[] shape, int[] strides, short[] out, int dimension, CpuExecutionConfig config) {
        int reducedDim = shape[dimension];
        int outSize = out.length;
        CpuExecutionMode mode = config.modeForReduction(logicalSize(shape));
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = config.computeChunkSize(outSize, 1);
            int chunks = (outSize + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, outSize);
                reduceOutputRangeF16(data, out, start, end, shape, strides, dimension, reducedDim, config.sumAccuracyMode());
            });
            return;
        }
        reduceOutputRangeF16(data, out, 0, outSize, shape, strides, dimension, reducedDim, config.sumAccuracyMode());
    }

    private static void sumAxisStridedF16(short[] data, int[] shape, int[] strides, short[] out, int dimension, CpuExecutionConfig config) {
        int reducedDim = shape[dimension];
        int outSize = out.length;
        CpuExecutionMode mode = config.modeForReduction(logicalSize(shape));
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = config.computeChunkSize(outSize, 1);
            int chunks = (outSize + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, outSize);
                reduceOutputRangeF16(data, out, start, end, shape, strides, dimension, reducedDim, config.sumAccuracyMode());
            });
            return;
        }
        reduceOutputRangeF16(data, out, 0, outSize, shape, strides, dimension, reducedDim, config.sumAccuracyMode());
    }

    private static void reduceOutputRange(
            double[] data,
            double[] out,
            int fromOut,
            int toOut,
            int[] shape,
            int[] strides,
            int dimension,
            int reducedDim,
            boolean useVectorLastDim,
            SumAccuracyMode accuracy
    ) {
        int[] outDenseStrides = denseStridesExcludingDim(shape, dimension);
        int axisStride = strides[dimension];
        int rank = shape.length;

        for (int outIndex = fromOut; outIndex < toOut; outIndex++) {
            int rem = outIndex;
            int base = 0;
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
            out[outIndex] = acc;
        }
    }

    private static void reduceOutputRangeF32(
            float[] data,
            float[] out,
            int fromOut,
            int toOut,
            int[] shape,
            int[] strides,
            int dimension,
            int reducedDim,
            boolean useVectorLastDim,
            SumAccuracyMode accuracy
    ) {
        int[] outDenseStrides = denseStridesExcludingDim(shape, dimension);
        int axisStride = strides[dimension];
        int rank = shape.length;

        for (int outIndex = fromOut; outIndex < toOut; outIndex++) {
            int rem = outIndex;
            int base = 0;
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
            out[outIndex] = (float) acc;
        }
    }

    private static void reduceOutputRangeF16(
            short[] data,
            short[] out,
            int fromOut,
            int toOut,
            int[] shape,
            int[] strides,
            int dimension,
            int reducedDim,
            SumAccuracyMode accuracy
    ) {
        int[] outDenseStrides = denseStridesExcludingDim(shape, dimension);
        int axisStride = strides[dimension];
        int rank = shape.length;

        for (int outIndex = fromOut; outIndex < toOut; outIndex++) {
            int rem = outIndex;
            int base = 0;
            int outAxis = 0;
            for (int d = 0; d < rank; d++) {
                if (d == dimension) continue;
                int coord = rem / outDenseStrides[outAxis];
                rem %= outDenseStrides[outAxis];
                base += coord * strides[d];
                outAxis++;
            }
            double acc = accumulateStridedFixedBaseF16(data, base, axisStride, reducedDim, accuracy);
            out[outIndex] = CpuDTypeOps.toHalfBits((float) acc);
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
            SumAccuracyMode accuracy
    ) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    sum += data[logicalToOffset(i, shape, strides, denseStrides)];
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    double y = data[logicalToOffset(i, shape, strides, denseStrides)] - c;
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
                    double x = data[logicalToOffset(i, shape, strides, denseStrides)];
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
            SumAccuracyMode accuracy
    ) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    sum += data[logicalToOffset(i, shape, strides, denseStrides)];
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    double y = data[logicalToOffset(i, shape, strides, denseStrides)] - c;
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
                    double x = data[logicalToOffset(i, shape, strides, denseStrides)];
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

    private static double accumulateStridedRangeF16(
            short[] data,
            int startLogical,
            int endLogical,
            int[] shape,
            int[] strides,
            int[] denseStrides,
            SumAccuracyMode accuracy
    ) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    sum += CpuDTypeOps.fromHalfBits(data[logicalToOffset(i, shape, strides, denseStrides)]);
                }
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = startLogical; i < endLogical; i++) {
                    double y = CpuDTypeOps.fromHalfBits(data[logicalToOffset(i, shape, strides, denseStrides)]) - c;
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
                    double x = CpuDTypeOps.fromHalfBits(data[logicalToOffset(i, shape, strides, denseStrides)]);
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

    private static double accumulateScalarF16(short[] data, int start, int end, SumAccuracyMode accuracy) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                for (int i = start; i < end; i++) sum += CpuDTypeOps.fromHalfBits(data[i]);
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                for (int i = start; i < end; i++) {
                    double y = CpuDTypeOps.fromHalfBits(data[i]) - c;
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
                    double x = CpuDTypeOps.fromHalfBits(data[i]);
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

    private static double accumulateStridedFixedBaseF16(short[] data, int base, int step, int count, SumAccuracyMode accuracy) {
        return switch (accuracy) {
            case FAST -> {
                double sum = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) sum += CpuDTypeOps.fromHalfBits(data[idx]);
                yield sum;
            }
            case KAHAN -> {
                double sum = 0.0;
                double c = 0.0;
                int idx = base;
                for (int i = 0; i < count; i++, idx += step) {
                    double y = CpuDTypeOps.fromHalfBits(data[idx]) - c;
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
                    double x = CpuDTypeOps.fromHalfBits(data[idx]);
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

    private static Tensor materializeContiguous(Tensor input, CpuExecutionConfig config) {
        Tensor contiguous = new Tensor(input.getShape(), null, input.getLabel() + "_sum_contiguous_tmp", DataType.FLOAT64);
        double[] src = input.getData();
        double[] dst = contiguous.getData();
        int[] shape = input.getShape();
        int[] strides = input.getStrides();
        int[] dense = denseStrides(shape);
        int logical = logicalSize(shape);
        for (int i = 0; i < logical; i++) {
            dst[i] = src[logicalToOffset(i, shape, strides, dense)];
        }
        return contiguous;
    }

    private static Tensor materializeContiguousTyped(Tensor input, CpuExecutionConfig config, DataType dataType) {
        Tensor contiguous = new Tensor(input.getShape(), null, input.getLabel() + "_sum_contiguous_tmp", dataType);
        TensorRemap.apply(input, contiguous, config.contiguousMaterializeThreshold());
        return contiguous;
    }

    private static int logicalToOffset(int logicalIndex, int[] shape, int[] strides, int[] denseStrides) {
        int rank = shape.length;
        int rem = logicalIndex;
        int offset = 0;
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
