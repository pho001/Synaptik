package backend.kernels.cpu;

import backend.blas.BlasRuntime;
import backend.blas.OpenBlasFfmBridge;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import operations.Operation;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public class CpuMatMulKernel implements CpuKernel {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;
    private static volatile boolean blasAvailabilityLogged;

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor a = inputs.get(0);
        Tensor b = inputs.get(1);
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];

        double[] ad = a.getFloat64Data();
        double[] bd = b.getFloat64Data();
        double[] out = node.getFloat64Data();
        ResolvedMatMulHints hints = requireHints(context);
        if (as.length == 2 && bs.length == 2 && hints.useBlas() && tryBlasF64(ad, bd, out, m, n, k)) {
            return;
        }
        if (hints.useBatchedBlas() && tryBatchedBlasF64(ad, as, bd, bs, out, node.getShapeUnsafe(), m, n, k)) {
            return;
        }
        Arrays.fill(out, 0.0d);
        runF64(ad, as, bd, bs, out, node.getShapeUnsafe(), hints);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor a = inputs.get(0);
        Tensor b = inputs.get(1);
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];

        float[] ad = a.getFloat32Data();
        float[] bd = b.getFloat32Data();
        float[] out = node.getFloat32Data();
        ResolvedMatMulHints hints = requireHints(context);
        if (as.length == 2 && bs.length == 2 && hints.useBlas() && tryBlasF32(ad, bd, out, m, n, k)) {
            return;
        }
        if (hints.useBatchedBlas() && tryBatchedBlasF32(ad, as, bd, bs, out, node.getShapeUnsafe(), m, n, k)) {
            return;
        }
        Arrays.fill(out, 0.0f);
        runF32(ad, as, bd, bs, out, node.getShapeUnsafe(), hints);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor a = inputs.get(0);
        Tensor b = inputs.get(1);
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];

        short[] ad = a.getFloat16Data();
        short[] bd = b.getFloat16Data();
        short[] out = node.getFloat16Data();
        Arrays.fill(out, (short) 0);
        runF16(ad, as, bd, bs, out, node.getShapeUnsafe(), requireHints(context));
    }

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
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        int[] bBatchOffsets = computeBatchOffsets(bShape, outShape);
        if (parallel && batchCount * blockRows > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows, hints.plannedWorkers(), task -> {
                int batch = task / blockRows;
                int block = task % blockRows;
                int i0 = block * tm;
                int i1 = Math.min(i0 + tm, m);
                computeBlockF64(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, 0, n, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            computeBlockF64(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
        }
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
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        int[] bBatchOffsets = computeBatchOffsets(bShape, outShape);
        if (parallel && batchCount * blockRows > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows, hints.plannedWorkers(), task -> {
                int batch = task / blockRows;
                int block = task % blockRows;
                int i0 = block * tm;
                int i1 = Math.min(i0 + tm, m);
                computeBlockF32(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, 0, n, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            computeBlockF32(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
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

    static void runF16(short[] a, int[] aShape, short[] b, int[] bShape, short[] out, int[] outShape, ResolvedMatMulHints hints) {
        int batchCount = batchCount(outShape);
        int m = outShape[outShape.length - 2];
        int n = outShape[outShape.length - 1];
        int k = aShape[aShape.length - 1];
        int tm = positiveTile(hints.tileM(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M);
        int tn = positiveTile(hints.tileN(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N);
        int tk = positiveTile(hints.tileK(), CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K);
        boolean parallel = hints.parallel() && hints.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        int[] aBatchOffsets = computeBatchOffsets(aShape, outShape);
        int[] bBatchOffsets = computeBatchOffsets(bShape, outShape);
        if (parallel && batchCount * blockRows > 1) {
            CpuThreadPool.runChunks(batchCount * blockRows, hints.plannedWorkers(), task -> {
                int batch = task / blockRows;
                int block = task % blockRows;
                int i0 = block * tm;
                int i1 = Math.min(i0 + tm, m);
                computeBlockF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, i0, i1, 0, n, 0, k, n, k, tn, tk);
            });
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            computeBlockF16(a, b, out, aBatchOffsets[batch], bBatchOffsets[batch], batch * m * n, 0, m, 0, n, 0, k, n, k, tn, tk);
        }
    }

    private static void computeBlockF16(
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
                        float av = CpuDTypeOps.fromHalfBits(a[aRow + p]);
                        int bRow = bOffset + p * n;
                        for (int j = jj; j < jjEnd; j++) {
                            float cur = CpuDTypeOps.fromHalfBits(out[oRow + j]);
                            float bv = CpuDTypeOps.fromHalfBits(b[bRow + j]);
                            out[oRow + j] = CpuDTypeOps.toHalfBits(cur + av * bv);
                        }
                    }
                }
            }
        }
    }

    private static int positiveTile(int value, int fallback) {
        return value > 0 ? value : fallback;
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

    static boolean tryBlasF64(
            double[] ad,
            double[] bd,
            double[] od,
            int m,
            int n,
            int k
    ) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            OpenBlasFfmBridge.dgemmRowMajorNoTrans(
                    m, n, k,
                    1.0d,
                    ad, k,
                    bd, n,
                    0.0d,
                    od, n
            );
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] DGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    static boolean tryBlasF32(
            float[] ad,
            float[] bd,
            float[] od,
            int m,
            int n,
            int k
    ) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            OpenBlasFfmBridge.sgemmRowMajorNoTrans(
                    m, n, k,
                    1.0f,
                    ad, k,
                    bd, n,
                    0.0f,
                    od, n
            );
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] SGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    static boolean tryBatchedBlasF64(
            double[] ad,
            int[] as,
            double[] bd,
            int[] bs,
            double[] od,
            int[] outShape,
            int m,
            int n,
            int k
    ) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            int batchCount = batchCount(outShape);
            int[] aBatchOffsets = computeBatchOffsets(as, outShape);
            int[] bBatchOffsets = computeBatchOffsets(bs, outShape);
            int mn = m * n;
            for (int batch = 0; batch < batchCount; batch++) {
                OpenBlasFfmBridge.dgemmRowMajorNoTransOffsets(
                        m, n, k,
                        1.0d,
                        ad, aBatchOffsets[batch], k,
                        bd, bBatchOffsets[batch], n,
                        0.0d,
                        od, batch * mn, n
                );
            }
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] Batched DGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    static boolean tryBatchedBlasF32(
            float[] ad,
            int[] as,
            float[] bd,
            int[] bs,
            float[] od,
            int[] outShape,
            int m,
            int n,
            int k
    ) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            int batchCount = batchCount(outShape);
            int[] aBatchOffsets = computeBatchOffsets(as, outShape);
            int[] bBatchOffsets = computeBatchOffsets(bs, outShape);
            int mn = m * n;
            for (int batch = 0; batch < batchCount; batch++) {
                OpenBlasFfmBridge.sgemmRowMajorNoTransOffsets(
                        m, n, k,
                        1.0f,
                        ad, aBatchOffsets[batch], k,
                        bd, bBatchOffsets[batch], n,
                        0.0f,
                        od, batch * mn, n
                );
            }
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] Batched SGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    private static void maybeLogBlasUnavailable() {
        if (blasAvailabilityLogged) {
            return;
        }
        synchronized (CpuMatMulKernel.class) {
            if (blasAvailabilityLogged) {
                return;
            }
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] OpenBLAS FFM unavailable, using Java matmul fallback. Reason: "
                        + OpenBlasFfmBridge.unavailableReason());
            }
            blasAvailabilityLogged = true;
        }
    }

    @Override
    public CpuKernelCostClass costClass(Operation op) {
        return CpuKernelCostClass.HIGH;
    }

    private static ResolvedMatMulHints requireHints(CpuKernelContext context) {
        ResolvedMatMulHints hints = context.matMulHints();
        if (hints == null) {
            throw new IllegalStateException("Missing ResolvedMatMulHints for matmul execution");
        }
        return hints;
    }
}
