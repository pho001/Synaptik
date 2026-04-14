package backend.kernels.cpu.linalg;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionPlanner;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedMatMulHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

final class MatMulJavaBackend {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    private MatMulJavaBackend() {}

    static void runF64(double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape, ResolvedMatMulHints hints) {
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
                computeBlockF64(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            computeBlockF64(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
        }
    }

    static void runF32(float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints) {
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
                computeBlockF32(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            computeBlockF32(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
        }
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
                computeBlockBF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, j0, j1, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            computeBlockBF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
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

    private static void computeBlockF64(
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
                for (int i = iStart; i < iEnd; i++) {
                    int aRow = aOffset + i * k;
                    int oRow = outOffset + i * n;
                    for (int p = kk; p < kkEnd; p++) {
                        double av = a[aRow + p];
                        int bRow = bOffset + p * n;
                        DoubleVector avv = DoubleVector.broadcast(F64, av);
                        int j = jj;
                        int upper = jjEnd - ((jjEnd - j) % width);
                        for (; j < upper; j += width) {
                            DoubleVector cv = DoubleVector.fromArray(F64, out, oRow + j);
                            DoubleVector bv = DoubleVector.fromArray(F64, b, bRow + j);
                            cv.add(avv.mul(bv)).intoArray(out, oRow + j);
                        }
                        for (; j < jjEnd; j++) {
                            out[oRow + j] += av * b[bRow + j];
                        }
                    }
                }
            }
        }
    }

    private static void computeBlockF32(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                for (int i = iStart; i < iEnd; i++) {
                    int aRow = aOffset + i * k;
                    int oRow = outOffset + i * n;
                    for (int p = kk; p < kkEnd; p++) {
                        float av = a[aRow + p];
                        int bRow = bOffset + p * n;
                        FloatVector avv = FloatVector.broadcast(F32, av);
                        int j = jj;
                        int upper = jjEnd - ((jjEnd - j) % width);
                        for (; j < upper; j += width) {
                            FloatVector cv = FloatVector.fromArray(F32, out, oRow + j);
                            FloatVector bv = FloatVector.fromArray(F32, b, bRow + j);
                            cv.add(avv.mul(bv)).intoArray(out, oRow + j);
                        }
                        for (; j < jjEnd; j++) {
                            out[oRow + j] += av * b[bRow + j];
                        }
                    }
                }
            }
        }
    }

    private static void computeBlockBF16(
            short[] a, short[] b, short[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                for (int i = iStart; i < iEnd; i++) {
                    int aRow = aOffset + i * k;
                    int oRow = outOffset + i * n;
                    for (int p = kk; p < kkEnd; p++) {
                        float av = CpuDTypeOps.fromBFloat16Bits(a[aRow + p]);
                        int bRow = bOffset + p * n;
                        for (int j = jj; j < jjEnd; j++) {
                            float cur = CpuDTypeOps.fromBFloat16Bits(out[oRow + j]);
                            float bv = CpuDTypeOps.fromBFloat16Bits(b[bRow + j]);
                            out[oRow + j] = CpuDTypeOps.toBFloat16Bits(cur + av * bv);
                        }
                    }
                }
            }
        }
    }
}
