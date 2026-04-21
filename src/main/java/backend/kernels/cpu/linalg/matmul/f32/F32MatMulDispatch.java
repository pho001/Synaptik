package backend.kernels.cpu.linalg.matmul.f32;

import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.linalg.matmul.common.MatMulBatchingSupport;
import backend.kernels.cpu.linalg.matmul.common.PackedLinearWeightCache;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import backend.kernels.cpu.plan.CpuExecutionPlanner;

final class F32MatMulDispatch {
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
    private interface PackedF32BlockKernel {
        void compute(
                float[] a, PackedLinearWeightCache.PackedFloatPanels packedB, float[] out,
                int aOffset, int outOffset,
                int iStart, int iEnd,
                int jStart, int jEnd,
                int kStart, int kEnd,
                int n, int k,
                int tn, int tk
        );
    }

    private F32MatMulDispatch() {
    }

    public static void run(float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints) {
        F32BlockKernel kernel = switch (hints.microKernel()) {
            case F32_2X4 -> F32MatMulKernel4Cols::computeBlockF32_2x4;
            case F32_2X8 -> F32MatMulKernel8Cols::computeBlockF32_2x8;
            case F32_4X4 -> F32MatMulKernel4Cols::computeBlockF32_4x4;
            default -> F32MatMulKernel2Cols::computeBlockF32_4x2;
        };
        runF32Blocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    public static void runRightTransposed(
            float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F32BlockKernel kernel = switch (hints.microKernel()) {
            case F32_2X4 -> F32MatMulKernel4Cols::computeBlockF32_2x4_RhsTransposed;
            case F32_2X8 -> F32MatMulKernel8Cols::computeBlockF32_2x8_RhsTransposed;
            case F32_4X4 -> F32MatMulKernel4Cols::computeBlockF32_4x4_RhsTransposed;
            default -> F32MatMulKernel2Cols::computeBlockF32_4x2_RhsTransposed;
        };
        runF32RightTransposedBlocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    public static void runLeftTransposed(
            float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F32BlockKernel kernel = switch (hints.microKernel()) {
            case F32_2X4 -> F32MatMulKernel4Cols::computeBlockF32_2x4_LhsTransposed;
            case F32_2X8 -> F32MatMulKernel8Cols::computeBlockF32_2x8_LhsTransposed;
            case F32_4X4 -> F32MatMulKernel4Cols::computeBlockF32_4x4_LhsTransposed;
            default -> F32MatMulKernel2Cols::computeBlockF32_4x2_LhsTransposed;
        };
        runF32LeftTransposedBlocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    public static void runPacked(
            float[] a, int[] aShape, PackedLinearWeightCache.F32PackedWeights packedB,
            float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        runPacked(a, aShape, (PackedLinearWeightCache.PackedFloatPanels) packedB, out, outShape, hints);
    }

    public static void runPacked(
            float[] a, int[] aShape, PackedLinearWeightCache.PackedFloatPanels packedB,
            float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        PackedF32BlockKernel kernel = switch (hints.microKernel()) {
            case F32_2X4 -> F32MatMulKernel4Cols::computeBlockPackedF32_2x4;
            case F32_2X8 -> F32MatMulKernel8Cols::computeBlockPackedF32_2x8;
            case F32_4X4 -> F32MatMulKernel4Cols::computeBlockPackedF32_4x4;
            default -> F32MatMulKernel2Cols::computeBlockPackedF32_4x2;
        };
        runPackedF32Blocks(a, aShape, packedB, out, outShape, hints, kernel);
    }
    private static int batchCount(int[] outShape) {
        return MatMulBatchingSupport.batchCount(outShape);
    }

    private static int[] computeBatchOffsets(int[] inputShape, int[] outShape) {
        return MatMulBatchingSupport.computeBatchOffsets(inputShape, outShape);
    }

    private static int positiveTile(int value, int fallback) {
        return MatMulBatchingSupport.positiveTile(value, fallback);
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
            float[] a, int[] aShape, PackedLinearWeightCache.PackedFloatPanels packedB, float[] out, int[] outShape,
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

}
