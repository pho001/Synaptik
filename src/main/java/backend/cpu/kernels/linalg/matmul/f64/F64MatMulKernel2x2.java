package backend.cpu.kernels.linalg.matmul.f64;

import backend.cpu.kernels.linalg.matmul.common.PackedLinearWeightCache;

final class F64MatMulKernel2x2 {
    private F64MatMulKernel2x2() {
    }

    static void computeBlockF64_2x2(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64MatMulKernelSupport.F64.length();
        int vectorBlockWidth = width * 2;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                double[] packedB = F64MatMulPacking.packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    F64MatMulKernelSupport.computeTwoRowsTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    F64MatMulKernelSupport.computeSingleRowTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockPackedF64_2x2(
            double[] a, PackedLinearWeightCache.F64PackedWeights packedWeights, double[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64MatMulKernelSupport.F64.length();
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
                    F64MatMulKernelSupport.computeTwoRowsTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    F64MatMulKernelSupport.computeSingleRowTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockF64_2x2_RhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64MatMulKernelSupport.F64.length();
        int vectorBlockWidth = width * 2;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                double[] packedB = F64MatMulPacking.packedPanelF64Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    F64MatMulKernelSupport.computeTwoRowsTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    F64MatMulKernelSupport.computeSingleRowTwoColsF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockF64_2x2_LhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F64MatMulKernelSupport.F64.length();
        int vectorBlockWidth = width * 2;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            int panelDepth = kkEnd - kk;
            int i = iStart;
            for (; i + 1 < iEnd; i += 2) {
                double[] packedA = F64MatMulPacking.packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 2, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    double[] packedB = F64MatMulPacking.packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    F64MatMulKernelSupport.computeTwoRowsTwoColsF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
            for (; i < iEnd; i++) {
                double[] packedA = F64MatMulPacking.packedPanelF64LeftTransposed(a, aOffset, kk, kkEnd, i, i + 1, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    double[] packedB = F64MatMulPacking.packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    F64MatMulKernelSupport.computeSingleRowTwoColsF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }
}
