package backend.kernels.cpu.linalg;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import backend.kernels.cpu.plan.CpuExecutionPlanner;
import config.backend.CpuMatMulMicroKernel;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

final class MatMulJavaBackend {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;
    private static final ThreadLocal<double[]> F64_PACKED_B = ThreadLocal.withInitial(() -> new double[0]);
    private static final ThreadLocal<float[]> F32_PACKED_B = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<double[]> F64_PACKED_A = ThreadLocal.withInitial(() -> new double[0]);
    private static final ThreadLocal<float[]> F32_PACKED_A = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<float[]> BF16_PACKED_A = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<float[]> BF16_PACKED_B = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<float[]> BF16_ACCUM_TILE = ThreadLocal.withInitial(() -> new float[0]);

    @FunctionalInterface
    private interface F64BlockKernel {
        void compute(
                double[] a, double[] b, double[] out,
                int aOffset, int bOffset, int outOffset,
                int iStart, int iEnd,
                int jStart, int jEnd,
                int kStart, int kEnd,
                int n, int k,
                int tn, int tk
        );
    }

    @FunctionalInterface
    private interface F32BlockKernel {
        void compute(
                float[] a, float[] b, float[] out,
                int aOffset, int bOffset, int outOffset,
                int iStart, int iEnd,
                int jStart, int jEnd,
                int kStart, int kEnd,
                int n, int k,
                int tn, int tk
        );
    }

    @FunctionalInterface
    private interface PackedF64BlockKernel {
        void compute(
                double[] a, PackedLinearWeightCache.F64PackedWeights packedB, double[] out,
                int aOffset, int outOffset,
                int iStart, int iEnd,
                int jStart, int jEnd,
                int kStart, int kEnd,
                int n, int k,
                int tn, int tk
        );
    }

    @FunctionalInterface
    private interface PackedF32BlockKernel {
        void compute(
                float[] a, PackedLinearWeightCache.F32PackedWeights packedB, float[] out,
                int aOffset, int outOffset,
                int iStart, int iEnd,
                int jStart, int jEnd,
                int kStart, int kEnd,
                int n, int k,
                int tn, int tk
        );
    }

    private MatMulJavaBackend() {}

    static void runF64(double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape, ResolvedMatMulHints hints) {
        F64BlockKernel kernel = switch (hints.microKernel()) {
            case F64_2X1 -> MatMulJavaBackend::computeBlockF64_2x1;
            case F64_2X2 -> MatMulJavaBackend::computeBlockF64_2x2;
            default -> MatMulJavaBackend::computeBlockF64_4x1;
        };
        runF64Blocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    static void runF64RightTransposed(
            double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F64BlockKernel kernel = switch (hints.microKernel()) {
            case F64_2X1 -> MatMulJavaBackend::computeBlockF64_2x1_RhsTransposed;
            case F64_2X2 -> MatMulJavaBackend::computeBlockF64_2x2_RhsTransposed;
            default -> MatMulJavaBackend::computeBlockF64_4x1_RhsTransposed;
        };
        runF64RightTransposedBlocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    static void runF64LeftTransposed(
            double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F64BlockKernel kernel = switch (hints.microKernel()) {
            case F64_2X1 -> MatMulJavaBackend::computeBlockF64_2x1_LhsTransposed;
            case F64_2X2 -> MatMulJavaBackend::computeBlockF64_2x2_LhsTransposed;
            default -> MatMulJavaBackend::computeBlockF64_4x1_LhsTransposed;
        };
        runF64LeftTransposedBlocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    static void runPackedF64(
            double[] a, int[] aShape, PackedLinearWeightCache.F64PackedWeights packedB,
            double[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        PackedF64BlockKernel kernel = switch (hints.microKernel()) {
            case F64_2X1 -> MatMulJavaBackend::computeBlockPackedF64_2x1;
            case F64_2X2 -> MatMulJavaBackend::computeBlockPackedF64_2x2;
            default -> MatMulJavaBackend::computeBlockPackedF64_4x1;
        };
        runPackedF64Blocks(a, aShape, packedB, out, outShape, hints, kernel);
    }

    static void runF32(float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints) {
        F32BlockKernel kernel = switch (hints.microKernel()) {
            case F32_2X4 -> MatMulJavaBackend::computeBlockF32_2x4;
            case F32_2X8 -> MatMulJavaBackend::computeBlockF32_2x8;
            case F32_4X4 -> MatMulJavaBackend::computeBlockF32_4x4;
            default -> MatMulJavaBackend::computeBlockF32_4x2;
        };
        runF32Blocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    static void runF32RightTransposed(
            float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F32BlockKernel kernel = switch (hints.microKernel()) {
            case F32_2X4 -> MatMulJavaBackend::computeBlockF32_2x4_RhsTransposed;
            case F32_2X8 -> MatMulJavaBackend::computeBlockF32_2x8_RhsTransposed;
            case F32_4X4 -> MatMulJavaBackend::computeBlockF32_4x4_RhsTransposed;
            default -> MatMulJavaBackend::computeBlockF32_4x2_RhsTransposed;
        };
        runF32RightTransposedBlocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    static void runF32LeftTransposed(
            float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F32BlockKernel kernel = switch (hints.microKernel()) {
            case F32_2X4 -> MatMulJavaBackend::computeBlockF32_2x4_LhsTransposed;
            case F32_2X8 -> MatMulJavaBackend::computeBlockF32_2x8_LhsTransposed;
            case F32_4X4 -> MatMulJavaBackend::computeBlockF32_4x4_LhsTransposed;
            default -> MatMulJavaBackend::computeBlockF32_4x2_LhsTransposed;
        };
        runF32LeftTransposedBlocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    static void runPackedF32(
            float[] a, int[] aShape, PackedLinearWeightCache.F32PackedWeights packedB,
            float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        PackedF32BlockKernel kernel = switch (hints.microKernel()) {
            case F32_2X4 -> MatMulJavaBackend::computeBlockPackedF32_2x4;
            case F32_2X8 -> MatMulJavaBackend::computeBlockPackedF32_2x8;
            case F32_4X4 -> MatMulJavaBackend::computeBlockPackedF32_4x4;
            default -> MatMulJavaBackend::computeBlockPackedF32_4x2;
        };
        runPackedF32Blocks(a, aShape, packedB, out, outShape, hints, kernel);
    }

    static void runBF16(short[] a, int[] aShape, short[] b, int[] bShape, short[] out, int[] outShape, ResolvedMatMulHints hints) {
        int batchCount = batchCount(outShape);
        int m = outShape[outShape.length - 2];
        int n = outShape[outShape.length - 1];
        int k = aShape[aShape.length - 1];
        int tm = positiveTile(hints.tileM(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M);
        int tn = positiveTile(hints.tileN(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N);
        int tk = positiveTile(hints.tileK(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K);
        boolean parallel = hints.parallel() && hints.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        int blockCols = (n + tn - 1) / tn;
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        int[] bBatchOffsets = computeBatchOffsets(bShape, outShape);
        if (parallel && batchCount * blockRows * blockCols > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows * blockCols, hints.plannedWorkers(), task -> {
                int batch = task / (blockRows * blockCols);
                int batchTask = task % (blockRows * blockCols);
                int rowBlock = batchTask / blockCols;
                int colBlock = batchTask % blockCols;
                int i0 = rowBlock * tm;
                int i1 = Math.min(i0 + tm, m);
                int j0 = colBlock * tn;
                int j1 = Math.min(j0 + tn, n);
                computeBlockBF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tm, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            computeBlockBF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tm, tn, tk);
        }
    }

    static void runBF16ToFloat(short[] a, int[] aShape, short[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints) {
        int batchCount = batchCount(outShape);
        int m = outShape[outShape.length - 2];
        int n = outShape[outShape.length - 1];
        int k = aShape[aShape.length - 1];
        int tm = positiveTile(hints.tileM(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M);
        int tn = positiveTile(hints.tileN(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N);
        int tk = positiveTile(hints.tileK(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K);
        boolean parallel = hints.parallel() && hints.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        int blockCols = (n + tn - 1) / tn;
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        int[] bBatchOffsets = computeBatchOffsets(bShape, outShape);
        if (parallel && batchCount * blockRows * blockCols > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows * blockCols, hints.plannedWorkers(), task -> {
                int batch = task / (blockRows * blockCols);
                int batchTask = task % (blockRows * blockCols);
                int rowBlock = batchTask / blockCols;
                int colBlock = batchTask % blockCols;
                int i0 = rowBlock * tm;
                int i1 = Math.min(i0 + tm, m);
                int j0 = colBlock * tn;
                int j1 = Math.min(j0 + tn, n);
                computeBlockBF16ToFloat(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tm, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            computeBlockBF16ToFloat(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tm, tn, tk);
        }
    }

    static int batchCount(int[] outShape) {
        int count = 1;
        for (int i = 0; i < outShape.length - 2; i++) {
            count *= outShape[i];
        }
        return count;
    }

    static int[] computeBatchOffsets(int[] inputShape, int[] outShape) {
        int inputBatchRank = inputShape.length - 2;
        int outBatchRank = outShape.length - 2;
        int[] inputDenseStrides = denseStrides(inputShape);
        int[] alignedBatchShape = new int[outBatchRank];
        int[] alignedBatchStrides = new int[outBatchRank];
        int shapeOffset = outBatchRank - inputBatchRank;
        for (int d = 0; d < outBatchRank; d++) {
            if (d < shapeOffset) {
                alignedBatchShape[d] = 1;
                alignedBatchStrides[d] = 0;
                continue;
            }
            int inputDim = inputShape[d - shapeOffset];
            alignedBatchShape[d] = inputDim;
            alignedBatchStrides[d] = inputDim == 1 ? 0 : inputDenseStrides[d - shapeOffset];
        }

        int batchCount = batchCount(outShape);
        int[] offsets = new int[batchCount];
        if (outBatchRank == 0) {
            return offsets;
        }
        int[] outBatchShape = java.util.Arrays.copyOf(outShape, outBatchRank);
        int[] outBatchDenseStrides = denseStrides(outBatchShape);
        for (int batch = 0; batch < batchCount; batch++) {
            int tmp = batch;
            int offset = 0;
            for (int d = 0; d < outBatchRank; d++) {
                int coord = tmp / outBatchDenseStrides[d];
                tmp %= outBatchDenseStrides[d];
                if (alignedBatchShape[d] != 1) {
                    offset += coord * alignedBatchStrides[d];
                }
            }
            offsets[batch] = offset;
        }
        return offsets;
    }

    private static void runF64Blocks(
            double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape,
            ResolvedMatMulHints hints, F64BlockKernel kernel
    ) {
        int batchCount = batchCount(outShape);
        int m = outShape[outShape.length - 2];
        int n = outShape[outShape.length - 1];
        int k = aShape[aShape.length - 1];
        int tm = positiveTile(hints.tileM(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M);
        int tn = positiveTile(hints.tileN(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N);
        int tk = positiveTile(hints.tileK(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K);
        boolean parallel = hints.parallel() && hints.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        int blockCols = (n + tn - 1) / tn;
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        int[] bBatchOffsets = computeBatchOffsets(bShape, outShape);
        if (parallel && batchCount * blockRows * blockCols > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows * blockCols, hints.plannedWorkers(), task -> {
                int batch = task / (blockRows * blockCols);
                int batchTask = task % (blockRows * blockCols);
                int rowBlock = batchTask / blockCols;
                int colBlock = batchTask % blockCols;
                int i0 = rowBlock * tm;
                int i1 = Math.min(i0 + tm, m);
                int j0 = colBlock * tn;
                int j1 = Math.min(j0 + tn, n);
                kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
        }
    }

    private static void runF64RightTransposedBlocks(
            double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape,
            ResolvedMatMulHints hints, F64BlockKernel kernel
    ) {
        int batchCount = batchCount(outShape);
        int m = outShape[outShape.length - 2];
        int n = outShape[outShape.length - 1];
        int k = aShape[aShape.length - 1];
        int tm = positiveTile(hints.tileM(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M);
        int tn = positiveTile(hints.tileN(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N);
        int tk = positiveTile(hints.tileK(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K);
        boolean parallel = hints.parallel() && hints.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        int blockCols = (n + tn - 1) / tn;
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        int[] bBatchOffsets = computeBatchOffsets(bShape, outShape);
        if (parallel && batchCount * blockRows * blockCols > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows * blockCols, hints.plannedWorkers(), task -> {
                int batch = task / (blockRows * blockCols);
                int batchTask = task % (blockRows * blockCols);
                int rowBlock = batchTask / blockCols;
                int colBlock = batchTask % blockCols;
                int i0 = rowBlock * tm;
                int i1 = Math.min(i0 + tm, m);
                int j0 = colBlock * tn;
                int j1 = Math.min(j0 + tn, n);
                kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
        }
    }

    private static void runF64LeftTransposedBlocks(
            double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape,
            ResolvedMatMulHints hints, F64BlockKernel kernel
    ) {
        int batchCount = batchCount(outShape);
        int m = outShape[outShape.length - 2];
        int n = outShape[outShape.length - 1];
        int inner = bShape[bShape.length - 2];
        int sourceK = aShape[aShape.length - 1];
        int tm = positiveTile(hints.tileM(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M);
        int tn = positiveTile(hints.tileN(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N);
        int tk = positiveTile(hints.tileK(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K);
        boolean parallel = hints.parallel() && hints.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        int blockCols = (n + tn - 1) / tn;
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        int[] bBatchOffsets = computeBatchOffsets(bShape, outShape);
        if (parallel && batchCount * blockRows * blockCols > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows * blockCols, hints.plannedWorkers(), task -> {
                int batch = task / (blockRows * blockCols);
                int batchTask = task % (blockRows * blockCols);
                int rowBlock = batchTask / blockCols;
                int colBlock = batchTask % blockCols;
                int i0 = rowBlock * tm;
                int i1 = Math.min(i0 + tm, m);
                int j0 = colBlock * tn;
                int j1 = Math.min(j0 + tn, n);
                kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, inner, n, sourceK, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, inner, n, sourceK, tn, tk);
        }
    }

    private static void runPackedF64Blocks(
            double[] a, int[] aShape, PackedLinearWeightCache.F64PackedWeights packedB, double[] out, int[] outShape,
            ResolvedMatMulHints hints, PackedF64BlockKernel kernel
    ) {
        int batchCount = batchCount(outShape);
        int m = outShape[outShape.length - 2];
        int n = outShape[outShape.length - 1];
        int k = aShape[aShape.length - 1];
        int tm = positiveTile(hints.tileM(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M);
        int tn = positiveTile(hints.tileN(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N);
        int tk = positiveTile(hints.tileK(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K);
        boolean parallel = hints.parallel() && hints.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        int blockCols = (n + tn - 1) / tn;
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        if (parallel && batchCount * blockRows * blockCols > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows * blockCols, hints.plannedWorkers(), task -> {
                int batch = task / (blockRows * blockCols);
                int batchTask = task % (blockRows * blockCols);
                int rowBlock = batchTask / blockCols;
                int colBlock = batchTask % blockCols;
                int i0 = rowBlock * tm;
                int i1 = Math.min(i0 + tm, m);
                int j0 = colBlock * tn;
                int j1 = Math.min(j0 + tn, n);
                kernel.compute(a, packedB, out, aBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            kernel.compute(a, packedB, out, aBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
        }
    }

    private static void runF32Blocks(
            float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape,
            ResolvedMatMulHints hints, F32BlockKernel kernel
    ) {
        int batchCount = batchCount(outShape);
        int m = outShape[outShape.length - 2];
        int n = outShape[outShape.length - 1];
        int k = aShape[aShape.length - 1];
        int tm = positiveTile(hints.tileM(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M);
        int tn = positiveTile(hints.tileN(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N);
        int tk = positiveTile(hints.tileK(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K);
        boolean parallel = hints.parallel() && hints.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        int blockCols = (n + tn - 1) / tn;
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        int[] bBatchOffsets = computeBatchOffsets(bShape, outShape);
        if (parallel && batchCount * blockRows * blockCols > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows * blockCols, hints.plannedWorkers(), task -> {
                int batch = task / (blockRows * blockCols);
                int batchTask = task % (blockRows * blockCols);
                int rowBlock = batchTask / blockCols;
                int colBlock = batchTask % blockCols;
                int i0 = rowBlock * tm;
                int i1 = Math.min(i0 + tm, m);
                int j0 = colBlock * tn;
                int j1 = Math.min(j0 + tn, n);
                kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
        }
    }

    private static void runF32LeftTransposedBlocks(
            float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape,
            ResolvedMatMulHints hints, F32BlockKernel kernel
    ) {
        int batchCount = batchCount(outShape);
        int m = outShape[outShape.length - 2];
        int n = outShape[outShape.length - 1];
        int inner = bShape[bShape.length - 2];
        int sourceK = aShape[aShape.length - 1];
        int tm = positiveTile(hints.tileM(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M);
        int tn = positiveTile(hints.tileN(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N);
        int tk = positiveTile(hints.tileK(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K);
        boolean parallel = hints.parallel() && hints.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        int blockCols = (n + tn - 1) / tn;
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        int[] bBatchOffsets = computeBatchOffsets(bShape, outShape);
        if (parallel && batchCount * blockRows * blockCols > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows * blockCols, hints.plannedWorkers(), task -> {
                int batch = task / (blockRows * blockCols);
                int batchTask = task % (blockRows * blockCols);
                int rowBlock = batchTask / blockCols;
                int colBlock = batchTask % blockCols;
                int i0 = rowBlock * tm;
                int i1 = Math.min(i0 + tm, m);
                int j0 = colBlock * tn;
                int j1 = Math.min(j0 + tn, n);
                kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, inner, n, sourceK, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, inner, n, sourceK, tn, tk);
        }
    }

    private static void runF32RightTransposedBlocks(
            float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape,
            ResolvedMatMulHints hints, F32BlockKernel kernel
    ) {
        int batchCount = batchCount(outShape);
        int m = outShape[outShape.length - 2];
        int n = outShape[outShape.length - 1];
        int k = aShape[aShape.length - 1];
        int tm = positiveTile(hints.tileM(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M);
        int tn = positiveTile(hints.tileN(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N);
        int tk = positiveTile(hints.tileK(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K);
        boolean parallel = hints.parallel() && hints.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        int blockCols = (n + tn - 1) / tn;
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        int[] bBatchOffsets = computeBatchOffsets(bShape, outShape);
        if (parallel && batchCount * blockRows * blockCols > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows * blockCols, hints.plannedWorkers(), task -> {
                int batch = task / (blockRows * blockCols);
                int batchTask = task % (blockRows * blockCols);
                int rowBlock = batchTask / blockCols;
                int colBlock = batchTask % blockCols;
                int i0 = rowBlock * tm;
                int i1 = Math.min(i0 + tm, m);
                int j0 = colBlock * tn;
                int j1 = Math.min(j0 + tn, n);
                kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            kernel.compute(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
        }
    }

    private static void runPackedF32Blocks(
            float[] a, int[] aShape, PackedLinearWeightCache.F32PackedWeights packedB, float[] out, int[] outShape,
            ResolvedMatMulHints hints, PackedF32BlockKernel kernel
    ) {
        int batchCount = batchCount(outShape);
        int m = outShape[outShape.length - 2];
        int n = outShape[outShape.length - 1];
        int k = aShape[aShape.length - 1];
        int tm = positiveTile(hints.tileM(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M);
        int tn = positiveTile(hints.tileN(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N);
        int tk = positiveTile(hints.tileK(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K);
        boolean parallel = hints.parallel() && hints.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        int blockCols = (n + tn - 1) / tn;
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        if (parallel && batchCount * blockRows * blockCols > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows * blockCols, hints.plannedWorkers(), task -> {
                int batch = task / (blockRows * blockCols);
                int batchTask = task % (blockRows * blockCols);
                int rowBlock = batchTask / blockCols;
                int colBlock = batchTask % blockCols;
                int i0 = rowBlock * tm;
                int i1 = Math.min(i0 + tm, m);
                int j0 = colBlock * tn;
                int j1 = Math.min(j0 + tn, n);
                kernel.compute(a, packedB, out, aBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            kernel.compute(a, packedB, out, aBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
        }
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int positiveTile(int value, int fallback) {
        return value > 0 ? value : fallback;
    }

    private static double[] packedPanelF64(double[] b, int bOffset, int kStart, int kEnd, int jStart, int jEnd, int n) {
        int panelWidth = jEnd - jStart;
        int required = (kEnd - kStart) * panelWidth;
        double[] packed = F64_PACKED_B.get();
        if (packed.length < required) {
            packed = new double[required];
            F64_PACKED_B.set(packed);
        }
        int dst = 0;
        for (int p = kStart; p < kEnd; p++) {
            System.arraycopy(b, bOffset + p * n + jStart, packed, dst, panelWidth);
            dst += panelWidth;
        }
        return packed;
    }

    private static double[] packedPanelF64LeftTransposed(double[] a, int aOffset, int kStart, int kEnd, int iStart, int iEnd, int sourceK) {
        int rows = iEnd - iStart;
        int panelDepth = kEnd - kStart;
        int required = rows * panelDepth;
        double[] packed = F64_PACKED_A.get();
        if (packed.length < required) {
            packed = new double[required];
            F64_PACKED_A.set(packed);
        }
        for (int row = 0; row < rows; row++) {
            int srcCol = iStart + row;
            int dstBase = row * panelDepth;
            for (int p = kStart; p < kEnd; p++) {
                packed[dstBase + (p - kStart)] = a[aOffset + p * sourceK + srcCol];
            }
        }
        return packed;
    }

    private static float[] packedPanelF32(float[] b, int bOffset, int kStart, int kEnd, int jStart, int jEnd, int n) {
        int panelWidth = jEnd - jStart;
        int required = (kEnd - kStart) * panelWidth;
        float[] packed = F32_PACKED_B.get();
        if (packed.length < required) {
            packed = new float[required];
            F32_PACKED_B.set(packed);
        }
        int dst = 0;
        for (int p = kStart; p < kEnd; p++) {
            System.arraycopy(b, bOffset + p * n + jStart, packed, dst, panelWidth);
            dst += panelWidth;
        }
        return packed;
    }

    private static float[] packedPanelF32LeftTransposed(float[] a, int aOffset, int kStart, int kEnd, int iStart, int iEnd, int sourceK) {
        int rows = iEnd - iStart;
        int panelDepth = kEnd - kStart;
        int required = rows * panelDepth;
        float[] packed = F32_PACKED_A.get();
        if (packed.length < required) {
            packed = new float[required];
            F32_PACKED_A.set(packed);
        }
        for (int row = 0; row < rows; row++) {
            int srcCol = iStart + row;
            int dstBase = row * panelDepth;
            for (int p = kStart; p < kEnd; p++) {
                packed[dstBase + (p - kStart)] = a[aOffset + p * sourceK + srcCol];
            }
        }
        return packed;
    }

    private static double[] packedPanelF64Transposed(double[] b, int bOffset, int kStart, int kEnd, int jStart, int jEnd, int k) {
        int panelWidth = jEnd - jStart;
        int required = (kEnd - kStart) * panelWidth;
        double[] packed = F64_PACKED_B.get();
        if (packed.length < required) {
            packed = new double[required];
            F64_PACKED_B.set(packed);
        }
        int dst = 0;
        for (int p = kStart; p < kEnd; p++) {
            for (int j = jStart; j < jEnd; j++) {
                packed[dst++] = b[bOffset + j * k + p];
            }
        }
        return packed;
    }

    private static float[] packedPanelF32Transposed(float[] b, int bOffset, int kStart, int kEnd, int jStart, int jEnd, int k) {
        int panelWidth = jEnd - jStart;
        int required = (kEnd - kStart) * panelWidth;
        float[] packed = F32_PACKED_B.get();
        if (packed.length < required) {
            packed = new float[required];
            F32_PACKED_B.set(packed);
        }
        int dst = 0;
        for (int p = kStart; p < kEnd; p++) {
            for (int j = jStart; j < jEnd; j++) {
                packed[dst++] = b[bOffset + j * k + p];
            }
        }
        return packed;
    }

    private static float[] packedPanelBF16(short[] b, int bOffset, int kStart, int kEnd, int jStart, int jEnd, int n) {
        int panelWidth = jEnd - jStart;
        int required = (kEnd - kStart) * panelWidth;
        float[] packed = BF16_PACKED_B.get();
        if (packed.length < required) {
            packed = new float[required];
            BF16_PACKED_B.set(packed);
        }
        int dst = 0;
        for (int p = kStart; p < kEnd; p++) {
            int srcBase = bOffset + p * n + jStart;
            for (int j = 0; j < panelWidth; j++) {
                packed[dst++] = CpuDTypeOps.fromBFloat16Bits(b[srcBase + j]);
            }
        }
        return packed;
    }

    private static float[] packedPanelBF16Left(short[] a, int aOffset, int iStart, int iEnd, int kStart, int kEnd, int sourceK) {
        int rows = iEnd - iStart;
        int panelDepth = kEnd - kStart;
        int required = rows * panelDepth;
        float[] packed = BF16_PACKED_A.get();
        if (packed.length < required) {
            packed = new float[required];
            BF16_PACKED_A.set(packed);
        }
        int dst = 0;
        for (int i = iStart; i < iEnd; i++) {
            int srcBase = aOffset + i * sourceK + kStart;
            for (int p = 0; p < panelDepth; p++) {
                packed[dst++] = CpuDTypeOps.fromBFloat16Bits(a[srcBase + p]);
            }
        }
        return packed;
    }

    private static float[] bf16AccumTile(int required) {
        float[] tile = BF16_ACCUM_TILE.get();
        if (tile.length < required) {
            tile = new float[required];
            BF16_ACCUM_TILE.set(tile);
        }
        return tile;
    }

    private static void computeBlockF64_2x1(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    private static void computeBlockPackedF64_2x1(
            double[] a, PackedLinearWeightCache.F64PackedWeights packedWeights, double[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    private static void computeBlockF64_2x1_RhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = packedPanelF64Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    private static void computeBlockF64_2x1_LhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            int panelDepth = kkEnd - kk;
            int i = iStart;
            for (; i + 1 < iEnd; i += 2) {
                double[] packedA = packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 2, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    double[] packedB = packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeTwoRowsOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
                }
            }
            for (; i < iEnd; i++) {
                double[] packedA = packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 1, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    double[] packedB = packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeSingleRowOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
                }
            }
        }
    }

    private static void computeBlockF64_4x1(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    private static void computeBlockF64_4x1_RhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = packedPanelF64Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    private static void computeBlockF64_4x1_LhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            int panelDepth = kkEnd - kk;
            int i = iStart;
            for (; i + 3 < iEnd; i += 4) {
                double[] packedA = packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 4, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    double[] packedB = packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeFourRowsOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
                }
            }
            for (; i + 1 < iEnd; i += 2) {
                double[] packedA = packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 2, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    double[] packedB = packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeTwoRowsOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
                }
            }
            for (; i < iEnd; i++) {
                double[] packedA = packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 1, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    double[] packedB = packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeSingleRowOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
                }
            }
        }
    }

    private static void computeBlockPackedF64_4x1(
            double[] a, PackedLinearWeightCache.F64PackedWeights packedWeights, double[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    private static void computeBlockF64_2x2(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        int vectorBlockWidth = width * 2;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                double[] packedB = packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockPackedF64_2x2(
            double[] a, PackedLinearWeightCache.F64PackedWeights packedWeights, double[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        int vectorBlockWidth = width * 2;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                double[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF64_2x2_RhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        int vectorBlockWidth = width * 2;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                double[] packedB = packedPanelF64Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF64_2x2_LhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F64.length();
        int vectorBlockWidth = width * 2;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            int panelDepth = kkEnd - kk;
            int i = iStart;
            for (; i + 1 < iEnd; i += 2) {
                double[] packedA = packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 2, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    double[] packedB = packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeTwoRowsTwoColsF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
            for (; i < iEnd; i++) {
                double[] packedA = packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 1, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    double[] packedB = packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeSingleRowTwoColsF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeFourRowsOneColF64(
            double[] a, double[] out, double[] packedB,
            int aOffset, int outOffset,
            int row,
            int colStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int vectorLimit
    ) {
        int aRow0 = aOffset + row * k;
        int aRow1 = aOffset + (row + 1) * k;
        int aRow2 = aOffset + (row + 2) * k;
        int aRow3 = aOffset + (row + 3) * k;
        int outRow0 = outOffset + row * n;
        int outRow1 = outOffset + (row + 1) * n;
        int outRow2 = outOffset + (row + 2) * n;
        int outRow3 = outOffset + (row + 3) * n;
        int j = 0;
        for (; j < vectorLimit; j += F64.length()) {
            int outCol = colStart + j;
            DoubleVector c0 = DoubleVector.fromArray(F64, out, outRow0 + outCol);
            DoubleVector c1 = DoubleVector.fromArray(F64, out, outRow1 + outCol);
            DoubleVector c2 = DoubleVector.fromArray(F64, out, outRow2 + outCol);
            DoubleVector c3 = DoubleVector.fromArray(F64, out, outRow3 + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                DoubleVector bv = DoubleVector.fromArray(F64, packedB, packedRow + j);
                c0 = c0.add(DoubleVector.broadcast(F64, a[aRow0 + p]).mul(bv));
                c1 = c1.add(DoubleVector.broadcast(F64, a[aRow1 + p]).mul(bv));
                c2 = c2.add(DoubleVector.broadcast(F64, a[aRow2 + p]).mul(bv));
                c3 = c3.add(DoubleVector.broadcast(F64, a[aRow3 + p]).mul(bv));
            }
            c0.intoArray(out, outRow0 + outCol);
            c1.intoArray(out, outRow1 + outCol);
            c2.intoArray(out, outRow2 + outCol);
            c3.intoArray(out, outRow3 + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = colStart + j;
            double sum0 = out[outRow0 + outCol];
            double sum1 = out[outRow1 + outCol];
            double sum2 = out[outRow2 + outCol];
            double sum3 = out[outRow3 + outCol];
            for (int p = kStart; p < kEnd; p++) {
                double bv = packedB[(p - kStart) * panelWidth + j];
                sum0 += a[aRow0 + p] * bv;
                sum1 += a[aRow1 + p] * bv;
                sum2 += a[aRow2 + p] * bv;
                sum3 += a[aRow3 + p] * bv;
            }
            out[outRow0 + outCol] = sum0;
            out[outRow1 + outCol] = sum1;
            out[outRow2 + outCol] = sum2;
            out[outRow3 + outCol] = sum3;
        }
    }

    private static void computeTwoRowsOneColF64(
            double[] a, double[] out, double[] packedB,
            int aOffset, int outOffset,
            int row,
            int colStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int vectorLimit
    ) {
        int aRow0 = aOffset + row * k;
        int aRow1 = aOffset + (row + 1) * k;
        int outRow0 = outOffset + row * n;
        int outRow1 = outOffset + (row + 1) * n;
        int j = 0;
        for (; j < vectorLimit; j += F64.length()) {
            int outCol = colStart + j;
            DoubleVector c0 = DoubleVector.fromArray(F64, out, outRow0 + outCol);
            DoubleVector c1 = DoubleVector.fromArray(F64, out, outRow1 + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                DoubleVector bv = DoubleVector.fromArray(F64, packedB, packedRow + j);
                c0 = c0.add(DoubleVector.broadcast(F64, a[aRow0 + p]).mul(bv));
                c1 = c1.add(DoubleVector.broadcast(F64, a[aRow1 + p]).mul(bv));
            }
            c0.intoArray(out, outRow0 + outCol);
            c1.intoArray(out, outRow1 + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = colStart + j;
            double sum0 = out[outRow0 + outCol];
            double sum1 = out[outRow1 + outCol];
            for (int p = kStart; p < kEnd; p++) {
                double bv = packedB[(p - kStart) * panelWidth + j];
                sum0 += a[aRow0 + p] * bv;
                sum1 += a[aRow1 + p] * bv;
            }
            out[outRow0 + outCol] = sum0;
            out[outRow1 + outCol] = sum1;
        }
    }

    private static void computeSingleRowOneColF64(
            double[] a, double[] out, double[] packedB,
            int aOffset, int outOffset,
            int row,
            int colStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int vectorLimit
    ) {
        int aRow = aOffset + row * k;
        int outRow = outOffset + row * n;
        int j = 0;
        for (; j < vectorLimit; j += F64.length()) {
            int outCol = colStart + j;
            DoubleVector acc = DoubleVector.fromArray(F64, out, outRow + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                acc = acc.add(DoubleVector.broadcast(F64, a[aRow + p]).mul(DoubleVector.fromArray(F64, packedB, packedRow + j)));
            }
            acc.intoArray(out, outRow + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = colStart + j;
            double sum = out[outRow + outCol];
            for (int p = kStart; p < kEnd; p++) {
                sum += a[aRow + p] * packedB[(p - kStart) * panelWidth + j];
            }
            out[outRow + outCol] = sum;
        }
    }

    private static void computeTwoRowsTwoColsF64(
            double[] a, double[] out, double[] packedB,
            int aOffset, int outOffset,
            int row,
            int colStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int blockLimit,
            int vectorLimit,
            int width
    ) {
        int aRow0 = aOffset + row * k;
        int aRow1 = aOffset + (row + 1) * k;
        int outRow0 = outOffset + row * n;
        int outRow1 = outOffset + (row + 1) * n;
        int j = 0;
        for (; j < blockLimit; j += width * 2) {
            int outCol = colStart + j;
            DoubleVector c00 = DoubleVector.fromArray(F64, out, outRow0 + outCol);
            DoubleVector c01 = DoubleVector.fromArray(F64, out, outRow0 + outCol + width);
            DoubleVector c10 = DoubleVector.fromArray(F64, out, outRow1 + outCol);
            DoubleVector c11 = DoubleVector.fromArray(F64, out, outRow1 + outCol + width);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                DoubleVector b0 = DoubleVector.fromArray(F64, packedB, packedRow + j);
                DoubleVector b1 = DoubleVector.fromArray(F64, packedB, packedRow + j + width);
                DoubleVector a0 = DoubleVector.broadcast(F64, a[aRow0 + p]);
                DoubleVector a1 = DoubleVector.broadcast(F64, a[aRow1 + p]);
                c00 = c00.add(a0.mul(b0));
                c01 = c01.add(a0.mul(b1));
                c10 = c10.add(a1.mul(b0));
                c11 = c11.add(a1.mul(b1));
            }
            c00.intoArray(out, outRow0 + outCol);
            c01.intoArray(out, outRow0 + outCol + width);
            c10.intoArray(out, outRow1 + outCol);
            c11.intoArray(out, outRow1 + outCol + width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = colStart + j;
            DoubleVector c0 = DoubleVector.fromArray(F64, out, outRow0 + outCol);
            DoubleVector c1 = DoubleVector.fromArray(F64, out, outRow1 + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                DoubleVector bv = DoubleVector.fromArray(F64, packedB, packedRow + j);
                c0 = c0.add(DoubleVector.broadcast(F64, a[aRow0 + p]).mul(bv));
                c1 = c1.add(DoubleVector.broadcast(F64, a[aRow1 + p]).mul(bv));
            }
            c0.intoArray(out, outRow0 + outCol);
            c1.intoArray(out, outRow1 + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = colStart + j;
            double sum0 = out[outRow0 + outCol];
            double sum1 = out[outRow1 + outCol];
            for (int p = kStart; p < kEnd; p++) {
                double bv = packedB[(p - kStart) * panelWidth + j];
                sum0 += a[aRow0 + p] * bv;
                sum1 += a[aRow1 + p] * bv;
            }
            out[outRow0 + outCol] = sum0;
            out[outRow1 + outCol] = sum1;
        }
    }

    private static void computeSingleRowTwoColsF64(
            double[] a, double[] out, double[] packedB,
            int aOffset, int outOffset,
            int row,
            int colStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int blockLimit,
            int vectorLimit,
            int width
    ) {
        int aRow = aOffset + row * k;
        int outRow = outOffset + row * n;
        int j = 0;
        for (; j < blockLimit; j += width * 2) {
            int outCol = colStart + j;
            DoubleVector c0 = DoubleVector.fromArray(F64, out, outRow + outCol);
            DoubleVector c1 = DoubleVector.fromArray(F64, out, outRow + outCol + width);
            for (int p = kStart; p < kEnd; p++) {
                DoubleVector av = DoubleVector.broadcast(F64, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                c0 = c0.add(av.mul(DoubleVector.fromArray(F64, packedB, packedRow + j)));
                c1 = c1.add(av.mul(DoubleVector.fromArray(F64, packedB, packedRow + j + width)));
            }
            c0.intoArray(out, outRow + outCol);
            c1.intoArray(out, outRow + outCol + width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = colStart + j;
            DoubleVector acc = DoubleVector.fromArray(F64, out, outRow + outCol);
            for (int p = kStart; p < kEnd; p++) {
                DoubleVector av = DoubleVector.broadcast(F64, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                acc = acc.add(av.mul(DoubleVector.fromArray(F64, packedB, packedRow + j)));
            }
            acc.intoArray(out, outRow + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = colStart + j;
            double sum = out[outRow + outCol];
            for (int p = kStart; p < kEnd; p++) {
                sum += a[aRow + p] * packedB[(p - kStart) * panelWidth + j];
            }
            out[outRow + outCol] = sum;
        }
    }

    private static void computeBlockF32_2x4(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockPackedF32_2x4(
            float[] a, PackedLinearWeightCache.F32PackedWeights packedWeights, float[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF32_2x4_RhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedPanelF32Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF32_2x4_LhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            int panelDepth = kkEnd - kk;
            int i = iStart;
            for (; i + 1 < iEnd; i += 2) {
                float[] packedA = packedPanelF32LeftTransposed(a, aOffset, kk, kkEnd, i, i + 2, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeTwoRowsFourColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
            for (; i < iEnd; i++) {
                float[] packedA = packedPanelF32LeftTransposed(a, aOffset, kk, kkEnd, i, i + 1, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeSingleRowFourColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF32_2x8(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 8;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockPackedF32_2x8(
            float[] a, PackedLinearWeightCache.F32PackedWeights packedWeights, float[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 8;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF32_2x8_RhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 8;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedPanelF32Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF32_2x8_LhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 8;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            int panelDepth = kkEnd - kk;
            int i = iStart;
            for (; i + 1 < iEnd; i += 2) {
                float[] packedA = packedPanelF32LeftTransposed(a, aOffset, kk, kkEnd, i, i + 2, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeTwoRowsEightColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
            for (; i < iEnd; i++) {
                float[] packedA = packedPanelF32LeftTransposed(a, aOffset, kk, kkEnd, i, i + 1, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeSingleRowEightColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF32_4x2(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 2;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockPackedF32_4x2(
            float[] a, PackedLinearWeightCache.F32PackedWeights packedWeights, float[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 2;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF32_4x2_RhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 2;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedPanelF32Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF32_4x2_LhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 2;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            int panelDepth = kkEnd - kk;
            int i = iStart;
            for (; i + 3 < iEnd; i += 4) {
                float[] packedA = packedPanelF32LeftTransposed(a, aOffset, kk, kkEnd, i, i + 4, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeFourRowsTwoColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
            for (; i < iEnd; i++) {
                float[] packedA = packedPanelF32LeftTransposed(a, aOffset, kk, kkEnd, i, i + 1, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeSingleRowTwoColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF32_4x4(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockPackedF32_4x4(
            float[] a, PackedLinearWeightCache.F32PackedWeights packedWeights, float[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF32_4x4_RhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedPanelF32Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeBlockF32_4x4_LhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            int panelDepth = kkEnd - kk;
            int i = iStart;
            for (; i + 3 < iEnd; i += 4) {
                float[] packedA = packedPanelF32LeftTransposed(a, aOffset, kk, kkEnd, i, i + 4, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeFourRowsFourColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
            for (; i < iEnd; i++) {
                float[] packedA = packedPanelF32LeftTransposed(a, aOffset, kk, kkEnd, i, i + 1, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    float[] packedB = packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeSingleRowFourColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeTwoRowsFourColsF32(
            float[] a, float[] out, float[] packedB,
            int aOffset, int outOffset,
            int row,
            int jStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int blockLimit,
            int vectorLimit,
            int width
    ) {
        int aRow0 = aOffset + row * k;
        int aRow1 = aOffset + (row + 1) * k;
        int outRow0 = outOffset + row * n;
        int outRow1 = outOffset + (row + 1) * n;
        int j = 0;
        for (; j < blockLimit; j += width * 4) {
            int outCol = jStart + j;
            FloatVector c00 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c01 = FloatVector.fromArray(F32, out, outRow0 + outCol + width);
            FloatVector c02 = FloatVector.fromArray(F32, out, outRow0 + outCol + 2 * width);
            FloatVector c03 = FloatVector.fromArray(F32, out, outRow0 + outCol + 3 * width);
            FloatVector c10 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            FloatVector c11 = FloatVector.fromArray(F32, out, outRow1 + outCol + width);
            FloatVector c12 = FloatVector.fromArray(F32, out, outRow1 + outCol + 2 * width);
            FloatVector c13 = FloatVector.fromArray(F32, out, outRow1 + outCol + 3 * width);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector b0 = FloatVector.fromArray(F32, packedB, packedRow + j);
                FloatVector b1 = FloatVector.fromArray(F32, packedB, packedRow + j + width);
                FloatVector b2 = FloatVector.fromArray(F32, packedB, packedRow + j + 2 * width);
                FloatVector b3 = FloatVector.fromArray(F32, packedB, packedRow + j + 3 * width);
                FloatVector a0 = FloatVector.broadcast(F32, a[aRow0 + p]);
                FloatVector a1 = FloatVector.broadcast(F32, a[aRow1 + p]);
                c00 = c00.add(a0.mul(b0));
                c01 = c01.add(a0.mul(b1));
                c02 = c02.add(a0.mul(b2));
                c03 = c03.add(a0.mul(b3));
                c10 = c10.add(a1.mul(b0));
                c11 = c11.add(a1.mul(b1));
                c12 = c12.add(a1.mul(b2));
                c13 = c13.add(a1.mul(b3));
            }
            c00.intoArray(out, outRow0 + outCol);
            c01.intoArray(out, outRow0 + outCol + width);
            c02.intoArray(out, outRow0 + outCol + 2 * width);
            c03.intoArray(out, outRow0 + outCol + 3 * width);
            c10.intoArray(out, outRow1 + outCol);
            c11.intoArray(out, outRow1 + outCol + width);
            c12.intoArray(out, outRow1 + outCol + 2 * width);
            c13.intoArray(out, outRow1 + outCol + 3 * width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = jStart + j;
            FloatVector c0 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c1 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector bv = FloatVector.fromArray(F32, packedB, packedRow + j);
                c0 = c0.add(FloatVector.broadcast(F32, a[aRow0 + p]).mul(bv));
                c1 = c1.add(FloatVector.broadcast(F32, a[aRow1 + p]).mul(bv));
            }
            c0.intoArray(out, outRow0 + outCol);
            c1.intoArray(out, outRow1 + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = jStart + j;
            float sum0 = out[outRow0 + outCol];
            float sum1 = out[outRow1 + outCol];
            for (int p = kStart; p < kEnd; p++) {
                float bv = packedB[(p - kStart) * panelWidth + j];
                sum0 += a[aRow0 + p] * bv;
                sum1 += a[aRow1 + p] * bv;
            }
            out[outRow0 + outCol] = sum0;
            out[outRow1 + outCol] = sum1;
        }
    }

    private static void computeTwoRowsEightColsF32(
            float[] a, float[] out, float[] packedB,
            int aOffset, int outOffset,
            int row,
            int jStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int blockLimit,
            int vectorLimit,
            int width
    ) {
        int aRow0 = aOffset + row * k;
        int aRow1 = aOffset + (row + 1) * k;
        int outRow0 = outOffset + row * n;
        int outRow1 = outOffset + (row + 1) * n;
        int j = 0;
        for (; j < blockLimit; j += width * 8) {
            int outCol = jStart + j;
            FloatVector c00 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c01 = FloatVector.fromArray(F32, out, outRow0 + outCol + width);
            FloatVector c02 = FloatVector.fromArray(F32, out, outRow0 + outCol + 2 * width);
            FloatVector c03 = FloatVector.fromArray(F32, out, outRow0 + outCol + 3 * width);
            FloatVector c04 = FloatVector.fromArray(F32, out, outRow0 + outCol + 4 * width);
            FloatVector c05 = FloatVector.fromArray(F32, out, outRow0 + outCol + 5 * width);
            FloatVector c06 = FloatVector.fromArray(F32, out, outRow0 + outCol + 6 * width);
            FloatVector c07 = FloatVector.fromArray(F32, out, outRow0 + outCol + 7 * width);
            FloatVector c10 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            FloatVector c11 = FloatVector.fromArray(F32, out, outRow1 + outCol + width);
            FloatVector c12 = FloatVector.fromArray(F32, out, outRow1 + outCol + 2 * width);
            FloatVector c13 = FloatVector.fromArray(F32, out, outRow1 + outCol + 3 * width);
            FloatVector c14 = FloatVector.fromArray(F32, out, outRow1 + outCol + 4 * width);
            FloatVector c15 = FloatVector.fromArray(F32, out, outRow1 + outCol + 5 * width);
            FloatVector c16 = FloatVector.fromArray(F32, out, outRow1 + outCol + 6 * width);
            FloatVector c17 = FloatVector.fromArray(F32, out, outRow1 + outCol + 7 * width);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector b0 = FloatVector.fromArray(F32, packedB, packedRow + j);
                FloatVector b1 = FloatVector.fromArray(F32, packedB, packedRow + j + width);
                FloatVector b2 = FloatVector.fromArray(F32, packedB, packedRow + j + 2 * width);
                FloatVector b3 = FloatVector.fromArray(F32, packedB, packedRow + j + 3 * width);
                FloatVector b4 = FloatVector.fromArray(F32, packedB, packedRow + j + 4 * width);
                FloatVector b5 = FloatVector.fromArray(F32, packedB, packedRow + j + 5 * width);
                FloatVector b6 = FloatVector.fromArray(F32, packedB, packedRow + j + 6 * width);
                FloatVector b7 = FloatVector.fromArray(F32, packedB, packedRow + j + 7 * width);
                FloatVector a0 = FloatVector.broadcast(F32, a[aRow0 + p]);
                FloatVector a1 = FloatVector.broadcast(F32, a[aRow1 + p]);
                c00 = c00.add(a0.mul(b0));
                c01 = c01.add(a0.mul(b1));
                c02 = c02.add(a0.mul(b2));
                c03 = c03.add(a0.mul(b3));
                c04 = c04.add(a0.mul(b4));
                c05 = c05.add(a0.mul(b5));
                c06 = c06.add(a0.mul(b6));
                c07 = c07.add(a0.mul(b7));
                c10 = c10.add(a1.mul(b0));
                c11 = c11.add(a1.mul(b1));
                c12 = c12.add(a1.mul(b2));
                c13 = c13.add(a1.mul(b3));
                c14 = c14.add(a1.mul(b4));
                c15 = c15.add(a1.mul(b5));
                c16 = c16.add(a1.mul(b6));
                c17 = c17.add(a1.mul(b7));
            }
            c00.intoArray(out, outRow0 + outCol);
            c01.intoArray(out, outRow0 + outCol + width);
            c02.intoArray(out, outRow0 + outCol + 2 * width);
            c03.intoArray(out, outRow0 + outCol + 3 * width);
            c04.intoArray(out, outRow0 + outCol + 4 * width);
            c05.intoArray(out, outRow0 + outCol + 5 * width);
            c06.intoArray(out, outRow0 + outCol + 6 * width);
            c07.intoArray(out, outRow0 + outCol + 7 * width);
            c10.intoArray(out, outRow1 + outCol);
            c11.intoArray(out, outRow1 + outCol + width);
            c12.intoArray(out, outRow1 + outCol + 2 * width);
            c13.intoArray(out, outRow1 + outCol + 3 * width);
            c14.intoArray(out, outRow1 + outCol + 4 * width);
            c15.intoArray(out, outRow1 + outCol + 5 * width);
            c16.intoArray(out, outRow1 + outCol + 6 * width);
            c17.intoArray(out, outRow1 + outCol + 7 * width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = jStart + j;
            FloatVector c0 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c1 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector bv = FloatVector.fromArray(F32, packedB, packedRow + j);
                c0 = c0.add(FloatVector.broadcast(F32, a[aRow0 + p]).mul(bv));
                c1 = c1.add(FloatVector.broadcast(F32, a[aRow1 + p]).mul(bv));
            }
            c0.intoArray(out, outRow0 + outCol);
            c1.intoArray(out, outRow1 + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = jStart + j;
            float sum0 = out[outRow0 + outCol];
            float sum1 = out[outRow1 + outCol];
            for (int p = kStart; p < kEnd; p++) {
                float bv = packedB[(p - kStart) * panelWidth + j];
                sum0 += a[aRow0 + p] * bv;
                sum1 += a[aRow1 + p] * bv;
            }
            out[outRow0 + outCol] = sum0;
            out[outRow1 + outCol] = sum1;
        }
    }

    private static void computeFourRowsTwoColsF32(
            float[] a, float[] out, float[] packedB,
            int aOffset, int outOffset,
            int row,
            int jStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int blockLimit,
            int vectorLimit,
            int width
    ) {
        int aRow0 = aOffset + row * k;
        int aRow1 = aOffset + (row + 1) * k;
        int aRow2 = aOffset + (row + 2) * k;
        int aRow3 = aOffset + (row + 3) * k;
        int outRow0 = outOffset + row * n;
        int outRow1 = outOffset + (row + 1) * n;
        int outRow2 = outOffset + (row + 2) * n;
        int outRow3 = outOffset + (row + 3) * n;
        int j = 0;
        for (; j < blockLimit; j += width * 2) {
            int outCol = jStart + j;
            FloatVector c00 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c01 = FloatVector.fromArray(F32, out, outRow0 + outCol + width);
            FloatVector c10 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            FloatVector c11 = FloatVector.fromArray(F32, out, outRow1 + outCol + width);
            FloatVector c20 = FloatVector.fromArray(F32, out, outRow2 + outCol);
            FloatVector c21 = FloatVector.fromArray(F32, out, outRow2 + outCol + width);
            FloatVector c30 = FloatVector.fromArray(F32, out, outRow3 + outCol);
            FloatVector c31 = FloatVector.fromArray(F32, out, outRow3 + outCol + width);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector b0 = FloatVector.fromArray(F32, packedB, packedRow + j);
                FloatVector b1 = FloatVector.fromArray(F32, packedB, packedRow + j + width);
                FloatVector a0 = FloatVector.broadcast(F32, a[aRow0 + p]);
                FloatVector a1 = FloatVector.broadcast(F32, a[aRow1 + p]);
                FloatVector a2 = FloatVector.broadcast(F32, a[aRow2 + p]);
                FloatVector a3 = FloatVector.broadcast(F32, a[aRow3 + p]);
                c00 = c00.add(a0.mul(b0));
                c01 = c01.add(a0.mul(b1));
                c10 = c10.add(a1.mul(b0));
                c11 = c11.add(a1.mul(b1));
                c20 = c20.add(a2.mul(b0));
                c21 = c21.add(a2.mul(b1));
                c30 = c30.add(a3.mul(b0));
                c31 = c31.add(a3.mul(b1));
            }
            c00.intoArray(out, outRow0 + outCol);
            c01.intoArray(out, outRow0 + outCol + width);
            c10.intoArray(out, outRow1 + outCol);
            c11.intoArray(out, outRow1 + outCol + width);
            c20.intoArray(out, outRow2 + outCol);
            c21.intoArray(out, outRow2 + outCol + width);
            c30.intoArray(out, outRow3 + outCol);
            c31.intoArray(out, outRow3 + outCol + width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = jStart + j;
            FloatVector c0 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c1 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            FloatVector c2 = FloatVector.fromArray(F32, out, outRow2 + outCol);
            FloatVector c3 = FloatVector.fromArray(F32, out, outRow3 + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector bv = FloatVector.fromArray(F32, packedB, packedRow + j);
                c0 = c0.add(FloatVector.broadcast(F32, a[aRow0 + p]).mul(bv));
                c1 = c1.add(FloatVector.broadcast(F32, a[aRow1 + p]).mul(bv));
                c2 = c2.add(FloatVector.broadcast(F32, a[aRow2 + p]).mul(bv));
                c3 = c3.add(FloatVector.broadcast(F32, a[aRow3 + p]).mul(bv));
            }
            c0.intoArray(out, outRow0 + outCol);
            c1.intoArray(out, outRow1 + outCol);
            c2.intoArray(out, outRow2 + outCol);
            c3.intoArray(out, outRow3 + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = jStart + j;
            float sum0 = out[outRow0 + outCol];
            float sum1 = out[outRow1 + outCol];
            float sum2 = out[outRow2 + outCol];
            float sum3 = out[outRow3 + outCol];
            for (int p = kStart; p < kEnd; p++) {
                float bv = packedB[(p - kStart) * panelWidth + j];
                sum0 += a[aRow0 + p] * bv;
                sum1 += a[aRow1 + p] * bv;
                sum2 += a[aRow2 + p] * bv;
                sum3 += a[aRow3 + p] * bv;
            }
            out[outRow0 + outCol] = sum0;
            out[outRow1 + outCol] = sum1;
            out[outRow2 + outCol] = sum2;
            out[outRow3 + outCol] = sum3;
        }
    }

    private static void computeSingleRowTwoColsF32(
            float[] a, float[] out, float[] packedB,
            int aOffset, int outOffset,
            int row,
            int jStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int blockLimit,
            int vectorLimit,
            int width
    ) {
        int aRow = aOffset + row * k;
        int outRow = outOffset + row * n;
        int j = 0;
        for (; j < blockLimit; j += width * 2) {
            int outCol = jStart + j;
            FloatVector c0 = FloatVector.fromArray(F32, out, outRow + outCol);
            FloatVector c1 = FloatVector.fromArray(F32, out, outRow + outCol + width);
            for (int p = kStart; p < kEnd; p++) {
                FloatVector av = FloatVector.broadcast(F32, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                c0 = c0.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j)));
                c1 = c1.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + width)));
            }
            c0.intoArray(out, outRow + outCol);
            c1.intoArray(out, outRow + outCol + width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = jStart + j;
            FloatVector acc = FloatVector.fromArray(F32, out, outRow + outCol);
            for (int p = kStart; p < kEnd; p++) {
                FloatVector av = FloatVector.broadcast(F32, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                acc = acc.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j)));
            }
            acc.intoArray(out, outRow + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = jStart + j;
            float sum = out[outRow + outCol];
            for (int p = kStart; p < kEnd; p++) {
                sum += a[aRow + p] * packedB[(p - kStart) * panelWidth + j];
            }
            out[outRow + outCol] = sum;
        }
    }

    private static void computeFourRowsFourColsF32(
            float[] a, float[] out, float[] packedB,
            int aOffset, int outOffset,
            int row,
            int jStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int blockLimit,
            int vectorLimit,
            int width
    ) {
        int aRow0 = aOffset + row * k;
        int aRow1 = aOffset + (row + 1) * k;
        int aRow2 = aOffset + (row + 2) * k;
        int aRow3 = aOffset + (row + 3) * k;
        int outRow0 = outOffset + row * n;
        int outRow1 = outOffset + (row + 1) * n;
        int outRow2 = outOffset + (row + 2) * n;
        int outRow3 = outOffset + (row + 3) * n;
        int j = 0;
        for (; j < blockLimit; j += width * 4) {
            int outCol = jStart + j;
            FloatVector c00 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c01 = FloatVector.fromArray(F32, out, outRow0 + outCol + width);
            FloatVector c02 = FloatVector.fromArray(F32, out, outRow0 + outCol + 2 * width);
            FloatVector c03 = FloatVector.fromArray(F32, out, outRow0 + outCol + 3 * width);
            FloatVector c10 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            FloatVector c11 = FloatVector.fromArray(F32, out, outRow1 + outCol + width);
            FloatVector c12 = FloatVector.fromArray(F32, out, outRow1 + outCol + 2 * width);
            FloatVector c13 = FloatVector.fromArray(F32, out, outRow1 + outCol + 3 * width);
            FloatVector c20 = FloatVector.fromArray(F32, out, outRow2 + outCol);
            FloatVector c21 = FloatVector.fromArray(F32, out, outRow2 + outCol + width);
            FloatVector c22 = FloatVector.fromArray(F32, out, outRow2 + outCol + 2 * width);
            FloatVector c23 = FloatVector.fromArray(F32, out, outRow2 + outCol + 3 * width);
            FloatVector c30 = FloatVector.fromArray(F32, out, outRow3 + outCol);
            FloatVector c31 = FloatVector.fromArray(F32, out, outRow3 + outCol + width);
            FloatVector c32 = FloatVector.fromArray(F32, out, outRow3 + outCol + 2 * width);
            FloatVector c33 = FloatVector.fromArray(F32, out, outRow3 + outCol + 3 * width);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector b0 = FloatVector.fromArray(F32, packedB, packedRow + j);
                FloatVector b1 = FloatVector.fromArray(F32, packedB, packedRow + j + width);
                FloatVector b2 = FloatVector.fromArray(F32, packedB, packedRow + j + 2 * width);
                FloatVector b3 = FloatVector.fromArray(F32, packedB, packedRow + j + 3 * width);
                FloatVector a0 = FloatVector.broadcast(F32, a[aRow0 + p]);
                FloatVector a1 = FloatVector.broadcast(F32, a[aRow1 + p]);
                FloatVector a2 = FloatVector.broadcast(F32, a[aRow2 + p]);
                FloatVector a3 = FloatVector.broadcast(F32, a[aRow3 + p]);
                c00 = c00.add(a0.mul(b0));
                c01 = c01.add(a0.mul(b1));
                c02 = c02.add(a0.mul(b2));
                c03 = c03.add(a0.mul(b3));
                c10 = c10.add(a1.mul(b0));
                c11 = c11.add(a1.mul(b1));
                c12 = c12.add(a1.mul(b2));
                c13 = c13.add(a1.mul(b3));
                c20 = c20.add(a2.mul(b0));
                c21 = c21.add(a2.mul(b1));
                c22 = c22.add(a2.mul(b2));
                c23 = c23.add(a2.mul(b3));
                c30 = c30.add(a3.mul(b0));
                c31 = c31.add(a3.mul(b1));
                c32 = c32.add(a3.mul(b2));
                c33 = c33.add(a3.mul(b3));
            }
            c00.intoArray(out, outRow0 + outCol);
            c01.intoArray(out, outRow0 + outCol + width);
            c02.intoArray(out, outRow0 + outCol + 2 * width);
            c03.intoArray(out, outRow0 + outCol + 3 * width);
            c10.intoArray(out, outRow1 + outCol);
            c11.intoArray(out, outRow1 + outCol + width);
            c12.intoArray(out, outRow1 + outCol + 2 * width);
            c13.intoArray(out, outRow1 + outCol + 3 * width);
            c20.intoArray(out, outRow2 + outCol);
            c21.intoArray(out, outRow2 + outCol + width);
            c22.intoArray(out, outRow2 + outCol + 2 * width);
            c23.intoArray(out, outRow2 + outCol + 3 * width);
            c30.intoArray(out, outRow3 + outCol);
            c31.intoArray(out, outRow3 + outCol + width);
            c32.intoArray(out, outRow3 + outCol + 2 * width);
            c33.intoArray(out, outRow3 + outCol + 3 * width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = jStart + j;
            FloatVector c0 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c1 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            FloatVector c2 = FloatVector.fromArray(F32, out, outRow2 + outCol);
            FloatVector c3 = FloatVector.fromArray(F32, out, outRow3 + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector bv = FloatVector.fromArray(F32, packedB, packedRow + j);
                c0 = c0.add(FloatVector.broadcast(F32, a[aRow0 + p]).mul(bv));
                c1 = c1.add(FloatVector.broadcast(F32, a[aRow1 + p]).mul(bv));
                c2 = c2.add(FloatVector.broadcast(F32, a[aRow2 + p]).mul(bv));
                c3 = c3.add(FloatVector.broadcast(F32, a[aRow3 + p]).mul(bv));
            }
            c0.intoArray(out, outRow0 + outCol);
            c1.intoArray(out, outRow1 + outCol);
            c2.intoArray(out, outRow2 + outCol);
            c3.intoArray(out, outRow3 + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = jStart + j;
            float sum0 = out[outRow0 + outCol];
            float sum1 = out[outRow1 + outCol];
            float sum2 = out[outRow2 + outCol];
            float sum3 = out[outRow3 + outCol];
            for (int p = kStart; p < kEnd; p++) {
                float bv = packedB[(p - kStart) * panelWidth + j];
                sum0 += a[aRow0 + p] * bv;
                sum1 += a[aRow1 + p] * bv;
                sum2 += a[aRow2 + p] * bv;
                sum3 += a[aRow3 + p] * bv;
            }
            out[outRow0 + outCol] = sum0;
            out[outRow1 + outCol] = sum1;
            out[outRow2 + outCol] = sum2;
            out[outRow3 + outCol] = sum3;
        }
    }

    private static void computeSingleRowFourColsF32(
            float[] a, float[] out, float[] packedB,
            int aOffset, int outOffset,
            int row,
            int jStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int blockLimit,
            int vectorLimit,
            int width
    ) {
        int aRow = aOffset + row * k;
        int outRow = outOffset + row * n;
        int j = 0;
        for (; j < blockLimit; j += width * 4) {
            int outCol = jStart + j;
            FloatVector c0 = FloatVector.fromArray(F32, out, outRow + outCol);
            FloatVector c1 = FloatVector.fromArray(F32, out, outRow + outCol + width);
            FloatVector c2 = FloatVector.fromArray(F32, out, outRow + outCol + 2 * width);
            FloatVector c3 = FloatVector.fromArray(F32, out, outRow + outCol + 3 * width);
            for (int p = kStart; p < kEnd; p++) {
                FloatVector av = FloatVector.broadcast(F32, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                c0 = c0.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j)));
                c1 = c1.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + width)));
                c2 = c2.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 2 * width)));
                c3 = c3.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 3 * width)));
            }
            c0.intoArray(out, outRow + outCol);
            c1.intoArray(out, outRow + outCol + width);
            c2.intoArray(out, outRow + outCol + 2 * width);
            c3.intoArray(out, outRow + outCol + 3 * width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = jStart + j;
            FloatVector acc = FloatVector.fromArray(F32, out, outRow + outCol);
            for (int p = kStart; p < kEnd; p++) {
                FloatVector av = FloatVector.broadcast(F32, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                acc = acc.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j)));
            }
            acc.intoArray(out, outRow + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = jStart + j;
            float sum = out[outRow + outCol];
            for (int p = kStart; p < kEnd; p++) {
                sum += a[aRow + p] * packedB[(p - kStart) * panelWidth + j];
            }
            out[outRow + outCol] = sum;
        }
    }

    private static void computeSingleRowEightColsF32(
            float[] a, float[] out, float[] packedB,
            int aOffset, int outOffset,
            int row,
            int jStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int blockLimit,
            int vectorLimit,
            int width
    ) {
        int aRow = aOffset + row * k;
        int outRow = outOffset + row * n;
        int j = 0;
        for (; j < blockLimit; j += width * 8) {
            int outCol = jStart + j;
            FloatVector c0 = FloatVector.fromArray(F32, out, outRow + outCol);
            FloatVector c1 = FloatVector.fromArray(F32, out, outRow + outCol + width);
            FloatVector c2 = FloatVector.fromArray(F32, out, outRow + outCol + 2 * width);
            FloatVector c3 = FloatVector.fromArray(F32, out, outRow + outCol + 3 * width);
            FloatVector c4 = FloatVector.fromArray(F32, out, outRow + outCol + 4 * width);
            FloatVector c5 = FloatVector.fromArray(F32, out, outRow + outCol + 5 * width);
            FloatVector c6 = FloatVector.fromArray(F32, out, outRow + outCol + 6 * width);
            FloatVector c7 = FloatVector.fromArray(F32, out, outRow + outCol + 7 * width);
            for (int p = kStart; p < kEnd; p++) {
                FloatVector av = FloatVector.broadcast(F32, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                c0 = c0.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j)));
                c1 = c1.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + width)));
                c2 = c2.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 2 * width)));
                c3 = c3.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 3 * width)));
                c4 = c4.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 4 * width)));
                c5 = c5.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 5 * width)));
                c6 = c6.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 6 * width)));
                c7 = c7.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 7 * width)));
            }
            c0.intoArray(out, outRow + outCol);
            c1.intoArray(out, outRow + outCol + width);
            c2.intoArray(out, outRow + outCol + 2 * width);
            c3.intoArray(out, outRow + outCol + 3 * width);
            c4.intoArray(out, outRow + outCol + 4 * width);
            c5.intoArray(out, outRow + outCol + 5 * width);
            c6.intoArray(out, outRow + outCol + 6 * width);
            c7.intoArray(out, outRow + outCol + 7 * width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = jStart + j;
            FloatVector acc = FloatVector.fromArray(F32, out, outRow + outCol);
            for (int p = kStart; p < kEnd; p++) {
                FloatVector av = FloatVector.broadcast(F32, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                acc = acc.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j)));
            }
            acc.intoArray(out, outRow + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = jStart + j;
            float sum = out[outRow + outCol];
            for (int p = kStart; p < kEnd; p++) {
                sum += a[aRow + p] * packedB[(p - kStart) * panelWidth + j];
            }
            out[outRow + outCol] = sum;
        }
    }

    private static void computeBlockBF16(
            short[] a, short[] b, short[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tm, int tn, int tk
    ) {
        int vectorWidth = F32.length();
        int totalCols = jEnd - jStart;
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            float[] accum = bf16AccumTile(tileRows * totalCols);
            java.util.Arrays.fill(accum, 0, tileRows * totalCols, 0.0f);

            for (int kk = kStart; kk < kEnd; kk += tk) {
                int kkEnd = Math.min(kk + tk, kEnd);
                int panelDepth = kkEnd - kk;
                float[] packedA = packedPanelBF16Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int tileCols = jjEnd - jj;
                    int vectorLimit = tileCols - (tileCols % vectorWidth);
                    int colOffset = jj - jStart;
                    float[] packedB = packedPanelBF16(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    for (int row = 0; row < tileRows; row++) {
                        int aBase = row * panelDepth;
                        int accumBase = row * totalCols + colOffset;
                        for (int p = 0; p < panelDepth; p++) {
                            float av = packedA[aBase + p];
                            int panelBase = p * tileCols;
                            int j = 0;
                            if (vectorLimit > 0) {
                                FloatVector avVector = FloatVector.broadcast(F32, av);
                                for (; j < vectorLimit; j += vectorWidth) {
                                    FloatVector.fromArray(F32, accum, accumBase + j)
                                            .add(FloatVector.fromArray(F32, packedB, panelBase + j).mul(avVector))
                                            .intoArray(accum, accumBase + j);
                                }
                            }
                            for (; j < tileCols; j++) {
                                accum[accumBase + j] += av * packedB[panelBase + j];
                            }
                        }
                    }
                }
            }

            for (int row = 0; row < tileRows; row++) {
                int outRow = outOffset + (ii + row) * n + jStart;
                int accumBase = row * totalCols;
                for (int j = 0; j < totalCols; j++) {
                    out[outRow + j] = CpuDTypeOps.toBFloat16Bits(accum[accumBase + j]);
                }
            }
        }
    }

    private static void computeBlockBF16ToFloat(
            short[] a, short[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tm, int tn, int tk
    ) {
        int vectorWidth = F32.length();
        int totalCols = jEnd - jStart;
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            float[] accum = bf16AccumTile(tileRows * totalCols);
            java.util.Arrays.fill(accum, 0, tileRows * totalCols, 0.0f);

            for (int kk = kStart; kk < kEnd; kk += tk) {
                int kkEnd = Math.min(kk + tk, kEnd);
                int panelDepth = kkEnd - kk;
                float[] packedA = packedPanelBF16Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int tileCols = jjEnd - jj;
                    int vectorLimit = tileCols - (tileCols % vectorWidth);
                    int colOffset = jj - jStart;
                    float[] packedB = packedPanelBF16(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    for (int row = 0; row < tileRows; row++) {
                        int aBase = row * panelDepth;
                        int accumBase = row * totalCols + colOffset;
                        for (int p = 0; p < panelDepth; p++) {
                            float av = packedA[aBase + p];
                            int panelBase = p * tileCols;
                            int j = 0;
                            if (vectorLimit > 0) {
                                FloatVector avVector = FloatVector.broadcast(F32, av);
                                for (; j < vectorLimit; j += vectorWidth) {
                                    FloatVector.fromArray(F32, accum, accumBase + j)
                                            .add(FloatVector.fromArray(F32, packedB, panelBase + j).mul(avVector))
                                            .intoArray(accum, accumBase + j);
                                }
                            }
                            for (; j < tileCols; j++) {
                                accum[accumBase + j] += av * packedB[panelBase + j];
                            }
                        }
                    }
                }
            }

            for (int row = 0; row < tileRows; row++) {
                int outRow = outOffset + (ii + row) * n + jStart;
                int accumBase = row * totalCols;
                System.arraycopy(accum, accumBase, out, outRow, totalCols);
            }
        }
    }
}
