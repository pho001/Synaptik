package backend.cpu.kernels.linalg.matmul.f64;

import backend.cpu.kernels.linalg.matmul.common.PackedLinearWeightCache;

final class F64MatMulKernel4x1 {
    private F64MatMulKernel4x1() {
    }

    static void computeBlockF64_4x1(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64MatMulKernelSupport.F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = F64MatMulPacking.packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    F64MatMulKernelSupport.computeFourRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i + 1 < iEnd; i += 2) {
                    F64MatMulKernelSupport.computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    F64MatMulKernelSupport.computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    static void computeBlockF64_4x1_RhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64MatMulKernelSupport.F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = F64MatMulPacking.packedPanelF64Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    F64MatMulKernelSupport.computeFourRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i + 1 < iEnd; i += 2) {
                    F64MatMulKernelSupport.computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    F64MatMulKernelSupport.computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    static void computeBlockF64_4x1_LhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F64MatMulKernelSupport.F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            int panelDepth = kkEnd - kk;
            int i = iStart;
            for (; i + 3 < iEnd; i += 4) {
                double[] packedA = F64MatMulPacking.packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 4, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    double[] packedB = F64MatMulPacking.packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    F64MatMulKernelSupport.computeFourRowsOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
                }
            }
            for (; i + 1 < iEnd; i += 2) {
                double[] packedA = F64MatMulPacking.packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 2, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    double[] packedB = F64MatMulPacking.packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    F64MatMulKernelSupport.computeTwoRowsOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
                }
            }
            for (; i < iEnd; i++) {
                double[] packedA = F64MatMulPacking.packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 1, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    double[] packedB = F64MatMulPacking.packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    F64MatMulKernelSupport.computeSingleRowOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
                }
            }
        }
    }

    static void computeBlockPackedF64_4x1(
            double[] a, PackedLinearWeightCache.F64PackedWeights packedWeights, double[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64MatMulKernelSupport.F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    F64MatMulKernelSupport.computeFourRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i + 1 < iEnd; i += 2) {
                    F64MatMulKernelSupport.computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    F64MatMulKernelSupport.computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }
}
