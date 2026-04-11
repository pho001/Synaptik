package backend.kernels.cpu.linalg;

import backend.blas.BlasRuntime;
import backend.blas.OpenBlasFfmBridge;
import backend.kernels.cpu.CpuDTypeOps;

final class MatMulBlasBackend {
    private static volatile boolean blasAvailabilityLogged;

    private MatMulBlasBackend() {}

    static boolean tryBlasF64(double[] ad, double[] bd, double[] od, int m, int n, int k) {
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

    static boolean tryBlasF32(float[] ad, float[] bd, float[] od, int m, int n, int k) {
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

    static boolean tryBatchedBlasF64(double[] ad, int[] as, double[] bd, int[] bs, double[] od, int[] outShape, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            int batchCount = MatMulJavaBackend.batchCount(outShape);
            int[] aBatchOffsets = MatMulJavaBackend.computeBatchOffsets(as, outShape);
            int[] bBatchOffsets = MatMulJavaBackend.computeBatchOffsets(bs, outShape);
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

    static boolean tryBatchedBlasF32(float[] ad, int[] as, float[] bd, int[] bs, float[] od, int[] outShape, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            int batchCount = MatMulJavaBackend.batchCount(outShape);
            int[] aBatchOffsets = MatMulJavaBackend.computeBatchOffsets(as, outShape);
            int[] bBatchOffsets = MatMulJavaBackend.computeBatchOffsets(bs, outShape);
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

    static boolean tryBlasBF16(short[] ad, short[] bd, short[] od, float[] tmp, int m, int n, int k) {
        if (!tryBlasBF16ToFloat(ad, bd, tmp, m, n, k)) {
            return false;
        }
        materializeBFloat16(tmp, od, m * n);
        return true;
    }

    static boolean tryBatchedBlasBF16(short[] ad, int[] as, short[] bd, int[] bs, short[] od, float[] tmp, int[] outShape, int m, int n, int k) {
        if (!tryBatchedBlasBF16ToFloat(ad, as, bd, bs, tmp, outShape, m, n, k)) {
            return false;
        }
        materializeBFloat16(tmp, od, od.length);
        return true;
    }

    static boolean tryBlasBF16ToFloat(short[] ad, short[] bd, float[] out, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isAvailable()) {
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

    static boolean tryBatchedBlasBF16ToFloat(short[] ad, int[] as, short[] bd, int[] bs, float[] out, int[] outShape, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            maybeLogBlasUnavailable();
            return false;
        }
        try {
            int batchCount = MatMulJavaBackend.batchCount(outShape);
            int mn = m * n;
            if (out == null || out.length < batchCount * mn) {
                return false;
            }
            int[] aBatchOffsets = MatMulJavaBackend.computeBatchOffsets(as, outShape);
            int[] bBatchOffsets = MatMulJavaBackend.computeBatchOffsets(bs, outShape);
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

    static void materializeBFloat16(float[] src, short[] dst, int length) {
        int limit = Math.min(length, Math.min(src.length, dst.length));
        for (int i = 0; i < limit; i++) {
            dst[i] = CpuDTypeOps.toBFloat16Bits(src[i]);
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
