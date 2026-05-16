package backend.cpu.kernels.linalg.matmul.blas;

import backend.blas.BlasRuntime;
import backend.blas.OpenBlasFfmBridge;
import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.linalg.matmul.common.MatMulBatchingSupport;

public final class MatMulBlasBackend {
    private static volatile boolean blasAvailabilityLogged;

    private MatMulBlasBackend() {}

    public static boolean tryBlasF64(double[] ad, double[] bd, double[] od, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            OpenBlasFfmBridge.dgemmRowMajorNoTrans(m, n, k, 1.0d, ad, k, bd, n, 0.0d, od, n);
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] DGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    public static boolean tryBlasF32(float[] ad, float[] bd, float[] od, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            OpenBlasFfmBridge.sgemmRowMajorNoTrans(m, n, k, 1.0f, ad, k, bd, n, 0.0f, od, n);
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] SGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    public static boolean tryBatchedBlasF64(double[] ad, int[] as, double[] bd, int[] bs, double[] od, int[] outShape, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            int batchCount = MatMulBatchingSupport.batchCount(outShape);
            int[] aBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(as, outShape);
            int[] bBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(bs, outShape);
            int mn = m * n;
            for (int batch = 0; batch < batchCount; batch++) {
                OpenBlasFfmBridge.dgemmRowMajorNoTransOffsets(m, n, k, 1.0d, ad, aBatchOffsets[batch], k, bd, bBatchOffsets[batch], n, 0.0d, od, batch * mn, n);
            }
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] Batched DGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    public static boolean tryBatchedBlasF32(float[] ad, int[] as, float[] bd, int[] bs, float[] od, int[] outShape, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            int batchCount = MatMulBatchingSupport.batchCount(outShape);
            int[] aBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(as, outShape);
            int[] bBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(bs, outShape);
            int mn = m * n;
            for (int batch = 0; batch < batchCount; batch++) {
                OpenBlasFfmBridge.sgemmRowMajorNoTransOffsets(m, n, k, 1.0f, ad, aBatchOffsets[batch], k, bd, bBatchOffsets[batch], n, 0.0f, od, batch * mn, n);
            }
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] Batched SGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    public static boolean tryBlasBF16(short[] ad, short[] bd, short[] od, float[] tmp, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isBFloat16OutputGemmAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            OpenBlasFfmBridge.bgemmRowMajorNoTrans(
                    m,
                    n,
                    k,
                    CpuDTypeOps.toBFloat16Bits(1.0f),
                    ad,
                    k,
                    bd,
                    n,
                    CpuDTypeOps.toBFloat16Bits(0.0f),
                    od,
                    n
            );
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] BGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    public static boolean tryBatchedBlasBF16(short[] ad, int[] as, short[] bd, int[] bs, short[] od, float[] tmp, int[] outShape, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isBFloat16OutputGemmAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            int batchCount = MatMulBatchingSupport.batchCount(outShape);
            int[] aBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(as, outShape);
            int[] bBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(bs, outShape);
            int mn = m * n;
            short alpha = CpuDTypeOps.toBFloat16Bits(1.0f);
            short beta = CpuDTypeOps.toBFloat16Bits(0.0f);
            for (int batch = 0; batch < batchCount; batch++) {
                OpenBlasFfmBridge.bgemmRowMajorNoTransOffsets(
                        m,
                        n,
                        k,
                        alpha,
                        ad,
                        aBatchOffsets[batch],
                        k,
                        bd,
                        bBatchOffsets[batch],
                        n,
                        beta,
                        od,
                        batch * mn,
                        n
                );
            }
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] Batched BGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    public static boolean tryBlasBF16ToFloat(short[] ad, short[] bd, float[] out, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isBFloat16ToFloatGemmAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            if (out == null || out.length < m * n) {
                return false;
            }
            OpenBlasFfmBridge.sbgemmRowMajorNoTrans(m, n, k, 1.0f, ad, k, bd, n, 0.0f, out, n);
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] SBGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    public static boolean tryBatchedBlasBF16ToFloat(short[] ad, int[] as, short[] bd, int[] bs, float[] out, int[] outShape, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isBFloat16ToFloatGemmAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            int batchCount = MatMulBatchingSupport.batchCount(outShape);
            int mn = m * n;
            if (out == null || out.length < batchCount * mn) {
                return false;
            }
            int[] aBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(as, outShape);
            int[] bBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(bs, outShape);
            for (int batch = 0; batch < batchCount; batch++) {
                OpenBlasFfmBridge.sbgemmRowMajorNoTransOffsets(m, n, k, 1.0f, ad, aBatchOffsets[batch], k, bd, bBatchOffsets[batch], n, 0.0f, out, batch * mn, n);
            }
            return true;
        } catch (Throwable t) {
            if (BlasRuntime.debug()) {
                System.err.println("[BLAS] Batched SBGEMM failed, fallback to Java kernel: " + t.getMessage());
            }
            return false;
        }
    }

    private static void maybeLogBlasUnavailable() {
        if (blasAvailabilityLogged) {
            return;
        }
        synchronized (MatMulBlasBackend.class) {
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
}
