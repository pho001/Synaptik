package backend.kernels.cpu.linalg.matmul.bf16;

import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.linalg.matmul.common.MatMulBatchingSupport;
import backend.kernels.cpu.linalg.matmul.common.PackedLinearWeightCache;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import backend.kernels.cpu.plan.CpuExecutionPlanner;

final class BF16MatMulDispatch {
    public static void runPacked(
            short[] a, int[] aShape, PackedLinearWeightCache.BF16PackedWeights packedB,
            short[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        BF16AccumKernel kernel = BF16MatMulAccumulatorSelector.select(hints.microKernel());
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
                BF16MatMulBlockKernels.computeBlockPackedBF16(a, packedB, out, aBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tm, tn, tk, kernel);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            BF16MatMulBlockKernels.computeBlockPackedBF16(a, packedB, out, aBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tm, tn, tk, kernel);
        }
    }

    public static void runPackedF32ToBF16(
            float[] a, int[] aShape, PackedLinearWeightCache.PackedFloatPanels packedB,
            short[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        BF16AccumKernel kernel = BF16MatMulAccumulatorSelector.select(hints.microKernel());
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
                BF16MatMulBlockKernels.computeBlockPackedF32ToBF16(a, packedB, out, aBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tm, tn, tk, kernel);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            BF16MatMulBlockKernels.computeBlockPackedF32ToBF16(a, packedB, out, aBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tm, tn, tk, kernel);
        }
    }

    public static void runPackedToFloat(
            short[] a, int[] aShape, PackedLinearWeightCache.BF16PackedWeights packedB,
            float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        BF16AccumKernel kernel = BF16MatMulAccumulatorSelector.select(hints.microKernel());
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
                BF16MatMulBlockKernels.computeBlockPackedBF16ToFloat(a, packedB, out, aBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tm, tn, tk, kernel);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            BF16MatMulBlockKernels.computeBlockPackedBF16ToFloat(a, packedB, out, aBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tm, tn, tk, kernel);
        }
    }

    public static void run(short[] a, int[] aShape, short[] b, int[] bShape, short[] out, int[] outShape, ResolvedMatMulHints hints) {
        BF16AccumKernel kernel = BF16MatMulAccumulatorSelector.select(hints.microKernel());
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
                BF16MatMulBlockKernels.computeBlockBF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tm, tn, tk, kernel);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            BF16MatMulBlockKernels.computeBlockBF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tm, tn, tk, kernel);
        }
    }

    public static void runToFloat(short[] a, int[] aShape, short[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints) {
        BF16AccumKernel kernel = BF16MatMulAccumulatorSelector.select(hints.microKernel());
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
                BF16MatMulBlockKernels.computeBlockBF16ToFloat(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tm, tn, tk, kernel);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            BF16MatMulBlockKernels.computeBlockBF16ToFloat(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tm, tn, tk, kernel);
        }
    }

    public static void runF32ToBF16(float[] a, int[] aShape, float[] b, int[] bShape, short[] out, int[] outShape, ResolvedMatMulHints hints) {
        BF16AccumKernel kernel = BF16MatMulAccumulatorSelector.select(hints.microKernel());
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
                BF16MatMulBlockKernels.computeBlockF32ToBF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tm, tn, tk, kernel);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            BF16MatMulBlockKernels.computeBlockF32ToBF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tm, tn, tk, kernel);
        }
    }

    public static void runF32LeftBF16RightToBF16(
            float[] a, int[] aShape, short[] b, int[] bShape, short[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        BF16AccumKernel kernel = BF16MatMulAccumulatorSelector.select(hints.microKernel());
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
                BF16MatMulBlockKernels.computeBlockF32LeftBF16RightToBF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tm, tn, tk, kernel);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            BF16MatMulBlockKernels.computeBlockF32LeftBF16RightToBF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tm, tn, tk, kernel);
        }
    }

    public static void runF32LeftBF16RightToFloat(
            float[] a, int[] aShape, short[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        BF16AccumKernel kernel = BF16MatMulAccumulatorSelector.select(hints.microKernel());
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
                BF16MatMulBlockKernels.computeBlockF32LeftBF16RightToFloat(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tm, tn, tk, kernel);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            BF16MatMulBlockKernels.computeBlockF32LeftBF16RightToFloat(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tm, tn, tk, kernel);
        }
    }

    private BF16MatMulDispatch() {
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
}
