package backend.cpu.kernels.linalg.matmul.f32;

import backend.cpu.kernels.linalg.matmul.common.PackedLinearWeightCache;

final class F32MatMulKernel4x4 {
    private F32MatMulKernel4x4() {
    }

    static void computeBlockF32_4x4(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32MatMulKernel4ColsSupport.F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = F32MatMulPacking.packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    F32MatMulKernel4ColsSupport.computeFourRowsFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    F32MatMulKernel4ColsSupport.computeSingleRowFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockPackedF32_4x4(
            float[] a, PackedLinearWeightCache.PackedFloatPanels packedWeights, float[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32MatMulKernel4ColsSupport.F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    F32MatMulKernel4ColsSupport.computeFourRowsFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    F32MatMulKernel4ColsSupport.computeSingleRowFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockF32_4x4_RhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32MatMulKernel4ColsSupport.F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = F32MatMulPacking.packedPanelF32Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    F32MatMulKernel4ColsSupport.computeFourRowsFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    F32MatMulKernel4ColsSupport.computeSingleRowFourColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockF32_4x4_LhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F32MatMulKernel4ColsSupport.F32.length();
        int vectorBlockWidth = width * 4;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            int panelDepth = kkEnd - kk;
            int i = iStart;
            for (; i + 3 < iEnd; i += 4) {
                float[] packedA = F32MatMulPacking.packedPanelF32LeftTransposed(a, aOffset, kk, kkEnd, i, i + 4, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    float[] packedB = F32MatMulPacking.packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    F32MatMulKernel4ColsSupport.computeFourRowsFourColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
            for (; i < iEnd; i++) {
                float[] packedA = F32MatMulPacking.packedPanelF32LeftTransposed(a, aOffset, kk, kkEnd, i, i + 1, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    float[] packedB = F32MatMulPacking.packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    F32MatMulKernel4ColsSupport.computeSingleRowFourColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }
}
