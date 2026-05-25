package backend.cpu.kernels.linalg.matmul.f64;

import backend.cpu.execution.CpuThreadPool;
import backend.cpu.kernels.linalg.matmul.common.MatMulBatchingSupport;
import backend.cpu.kernels.linalg.matmul.common.PackedLinearWeightCache;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import backend.cpu.prepare.CpuExecutionPlanner;

final class F64MatMulDispatch {
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

    private F64MatMulDispatch() {
    }

    public static void run(double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape, ResolvedMatMulHints hints) {
        F64BlockKernel kernel = switch (hints.microKernel()) {
            case F64_2X1 -> F64MatMulKernel2x1::computeBlockF64_2x1;
            case F64_2X2 -> F64MatMulKernel2x2::computeBlockF64_2x2;
            default -> F64MatMulKernel4x1::computeBlockF64_4x1;
        };
        runF64Blocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    public static void runRightTransposed(
            double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F64BlockKernel kernel = switch (hints.microKernel()) {
            case F64_2X1 -> F64MatMulKernel2x1::computeBlockF64_2x1_RhsTransposed;
            case F64_2X2 -> F64MatMulKernel2x2::computeBlockF64_2x2_RhsTransposed;
            default -> F64MatMulKernel4x1::computeBlockF64_4x1_RhsTransposed;
        };
        runF64RightTransposedBlocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    public static void runLeftTransposed(
            double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F64BlockKernel kernel = switch (hints.microKernel()) {
            case F64_2X1 -> F64MatMulKernel2x1::computeBlockF64_2x1_LhsTransposed;
            case F64_2X2 -> F64MatMulKernel2x2::computeBlockF64_2x2_LhsTransposed;
            default -> F64MatMulKernel4x1::computeBlockF64_4x1_LhsTransposed;
        };
        runF64LeftTransposedBlocks(a, aShape, b, bShape, out, outShape, hints, kernel);
    }

    public static void runPacked(
            double[] a, int[] aShape, PackedLinearWeightCache.F64PackedWeights packedB,
            double[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        PackedF64BlockKernel kernel = switch (hints.microKernel()) {
            case F64_2X1 -> F64MatMulKernel2x1::computeBlockPackedF64_2x1;
            case F64_2X2 -> F64MatMulKernel2x2::computeBlockPackedF64_2x2;
            default -> F64MatMulKernel4x1::computeBlockPackedF64_4x1;
        };
        runPackedF64Blocks(a, aShape, packedB, out, outShape, hints, kernel);
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

}
