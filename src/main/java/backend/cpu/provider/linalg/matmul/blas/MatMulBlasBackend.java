package backend.cpu.provider.linalg.matmul.blas;

import backend.cpu.kernels.linalg.matmul.common.MatMulBatchingSupport;
import backend.provider.blas.openblas.OpenBlasArrayGemm;
import backend.provider.blas.openblas.OpenBlasRuntime;
import tensor.dtype.TensorDTypeOps;

import java.util.OptionalInt;

public final class MatMulBlasBackend {
    private static final int NO_THREAD_RESTORE = -1;

    private MatMulBlasBackend() {}

    public static boolean tryBlasF64(
            double[] ad, double[] bd, double[] od, int m, int n, int k,
            boolean debug, int openBlasThreads
    ) {
        int previousThreads = NO_THREAD_RESTORE;
        try {
            previousThreads = applyPreparedThreads(openBlasThreads);
            try {
                OpenBlasArrayGemm.dgemmRowMajorNoTrans(m, n, k, 1.0d, ad, k, bd, n, 0.0d, od, n);
            } finally {
                restoreThreads(previousThreads);
            }
            return true;
        } catch (Throwable t) {
            logFailure(debug, "DGEMM", t);
            return false;
        }
    }

    public static boolean tryBlasF32(
            float[] ad, float[] bd, float[] od, int m, int n, int k,
            boolean debug, int openBlasThreads
    ) {
        int previousThreads = NO_THREAD_RESTORE;
        try {
            previousThreads = applyPreparedThreads(openBlasThreads);
            try {
                OpenBlasArrayGemm.sgemmRowMajorNoTrans(m, n, k, 1.0f, ad, k, bd, n, 0.0f, od, n);
            } finally {
                restoreThreads(previousThreads);
            }
            return true;
        } catch (Throwable t) {
            logFailure(debug, "SGEMM", t);
            return false;
        }
    }

    public static boolean tryBatchedBlasF64(
            double[] ad, int[] as, double[] bd, int[] bs, double[] od, int[] outShape,
            int m, int n, int k, boolean debug, int openBlasThreads
    ) {
        int previousThreads = NO_THREAD_RESTORE;
        try {
            previousThreads = applyPreparedThreads(openBlasThreads);
            try {
                int batchCount = MatMulBatchingSupport.batchCount(outShape);
                int[] aBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(as, outShape);
                int[] bBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(bs, outShape);
                int mn = m * n;
                for (int batch = 0; batch < batchCount; batch++) {
                    OpenBlasArrayGemm.dgemmRowMajorNoTransOffsets(
                            m, n, k, 1.0d, ad, aBatchOffsets[batch], k,
                            bd, bBatchOffsets[batch], n, 0.0d, od, batch * mn, n
                    );
                }
            } finally {
                restoreThreads(previousThreads);
            }
            return true;
        } catch (Throwable t) {
            logFailure(debug, "Batched DGEMM", t);
            return false;
        }
    }

    public static boolean tryBatchedBlasF32(
            float[] ad, int[] as, float[] bd, int[] bs, float[] od, int[] outShape,
            int m, int n, int k, boolean debug, int openBlasThreads
    ) {
        int previousThreads = NO_THREAD_RESTORE;
        try {
            previousThreads = applyPreparedThreads(openBlasThreads);
            try {
                int batchCount = MatMulBatchingSupport.batchCount(outShape);
                int[] aBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(as, outShape);
                int[] bBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(bs, outShape);
                int mn = m * n;
                for (int batch = 0; batch < batchCount; batch++) {
                    OpenBlasArrayGemm.sgemmRowMajorNoTransOffsets(
                            m, n, k, 1.0f, ad, aBatchOffsets[batch], k,
                            bd, bBatchOffsets[batch], n, 0.0f, od, batch * mn, n
                    );
                }
            } finally {
                restoreThreads(previousThreads);
            }
            return true;
        } catch (Throwable t) {
            logFailure(debug, "Batched SGEMM", t);
            return false;
        }
    }

    public static boolean tryBlasBF16(
            short[] ad, short[] bd, short[] od, float[] tmp, int m, int n, int k,
            boolean debug, int openBlasThreads
    ) {
        int previousThreads = NO_THREAD_RESTORE;
        try {
            previousThreads = applyPreparedThreads(openBlasThreads);
            try {
                OpenBlasArrayGemm.bgemmRowMajorNoTrans(
                        m, n, k, TensorDTypeOps.toBFloat16Bits(1.0f), ad, k, bd, n,
                        TensorDTypeOps.toBFloat16Bits(0.0f), od, n
                );
            } finally {
                restoreThreads(previousThreads);
            }
            return true;
        } catch (Throwable t) {
            logFailure(debug, "BGEMM", t);
            return false;
        }
    }

    public static boolean tryBatchedBlasBF16(
            short[] ad, int[] as, short[] bd, int[] bs, short[] od, float[] tmp, int[] outShape,
            int m, int n, int k, boolean debug, int openBlasThreads
    ) {
        int previousThreads = NO_THREAD_RESTORE;
        try {
            previousThreads = applyPreparedThreads(openBlasThreads);
            try {
                int batchCount = MatMulBatchingSupport.batchCount(outShape);
                int[] aBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(as, outShape);
                int[] bBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(bs, outShape);
                int mn = m * n;
                short alpha = TensorDTypeOps.toBFloat16Bits(1.0f);
                short beta = TensorDTypeOps.toBFloat16Bits(0.0f);
                for (int batch = 0; batch < batchCount; batch++) {
                    OpenBlasArrayGemm.bgemmRowMajorNoTransOffsets(
                            m, n, k, alpha, ad, aBatchOffsets[batch], k,
                            bd, bBatchOffsets[batch], n, beta, od, batch * mn, n
                    );
                }
            } finally {
                restoreThreads(previousThreads);
            }
            return true;
        } catch (Throwable t) {
            logFailure(debug, "Batched BGEMM", t);
            return false;
        }
    }

    public static boolean tryBlasBF16ToFloat(
            short[] ad, short[] bd, float[] out, int m, int n, int k,
            boolean debug, int openBlasThreads
    ) {
        if (out == null || out.length < m * n) {
            return false;
        }
        int previousThreads = NO_THREAD_RESTORE;
        try {
            previousThreads = applyPreparedThreads(openBlasThreads);
            try {
                OpenBlasArrayGemm.sbgemmRowMajorNoTrans(m, n, k, 1.0f, ad, k, bd, n, 0.0f, out, n);
            } finally {
                restoreThreads(previousThreads);
            }
            return true;
        } catch (Throwable t) {
            logFailure(debug, "SBGEMM", t);
            return false;
        }
    }

    public static boolean tryBatchedBlasBF16ToFloat(
            short[] ad, int[] as, short[] bd, int[] bs, float[] out, int[] outShape,
            int m, int n, int k, boolean debug, int openBlasThreads
    ) {
        int batchCount = MatMulBatchingSupport.batchCount(outShape);
        int mn = m * n;
        if (out == null || out.length < batchCount * mn) {
            return false;
        }
        int previousThreads = NO_THREAD_RESTORE;
        try {
            previousThreads = applyPreparedThreads(openBlasThreads);
            try {
                int[] aBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(as, outShape);
                int[] bBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(bs, outShape);
                for (int batch = 0; batch < batchCount; batch++) {
                    OpenBlasArrayGemm.sbgemmRowMajorNoTransOffsets(
                            m, n, k, 1.0f, ad, aBatchOffsets[batch], k,
                            bd, bBatchOffsets[batch], n, 0.0f, out, batch * mn, n
                    );
                }
            } finally {
                restoreThreads(previousThreads);
            }
            return true;
        } catch (Throwable t) {
            logFailure(debug, "Batched SBGEMM", t);
            return false;
        }
    }

    private static int applyPreparedThreads(int requestedThreads) {
        if (requestedThreads <= 0) {
            return NO_THREAD_RESTORE;
        }
        OptionalInt previousThreads = OpenBlasRuntime.getNumThreads();
        if (previousThreads.isEmpty()) {
            throw new IllegalStateException("OpenBLAS thread override requires openblas_get_num_threads.");
        }
        int previous = previousThreads.getAsInt();
        if (previous == requestedThreads) {
            return NO_THREAD_RESTORE;
        }
        if (!OpenBlasRuntime.setNumThreads(requestedThreads)) {
            throw new IllegalStateException("OpenBLAS thread override requires openblas_set_num_threads.");
        }
        return previous;
    }

    private static void restoreThreads(int previousThreads) {
        if (previousThreads > 0) {
            OpenBlasRuntime.setNumThreads(previousThreads);
        }
    }

    private static void logFailure(boolean debug, String operation, Throwable failure) {
        if (debug) {
            System.err.println("[BLAS] " + operation + " failed, fallback to Java kernel: " + failure.getMessage());
        }
    }
}
