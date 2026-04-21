package backend.kernels.cpu.linalg.matmul.bf16;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.linalg.matmul.common.PackedLinearWeightCache;

final class BF16MatMulBlockKernels {
    private BF16MatMulBlockKernels() {
    }

    static void computeBlockBF16(
            short[] a, short[] b, short[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tm, int tn, int tk,
            BF16MatMulAccumulators.BF16AccumKernel kernel
    ) {
        int totalCols = jEnd - jStart;
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * totalCols);
            java.util.Arrays.fill(accum, 0, tileRows * totalCols, 0.0f);

            for (int kk = kStart; kk < kEnd; kk += tk) {
                int kkEnd = Math.min(kk + tk, kEnd);
                int panelDepth = kkEnd - kk;
                float[] packedA = BF16MatMulPacking.packedPanelBF16Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int tileCols = jjEnd - jj;
                    int colOffset = jj - jStart;
                    float[] packedB = BF16MatMulPacking.packedPanelBF16(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    kernel.compute(packedA, accum, packedB, tileRows, colOffset, panelDepth, totalCols, tileCols);
                }
            }

            for (int row = 0; row < tileRows; row++) {
                int outRow = outOffset + (ii + row) * n + jStart;
                int accumBase = row * totalCols;
                for (int j = 0; j < totalCols; j++) {
                    out[outRow + j] = CpuDTypeOps.toBFloat16Bits(accum[accumBase + j]);
                }
            }
        }
    }

    static void computeBlockPackedBF16(
            short[] a, PackedLinearWeightCache.PackedFloatPanels packedWeights, short[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tm, int tn, int tk,
            BF16MatMulAccumulators.BF16AccumKernel kernel
    ) {
        int totalCols = jEnd - jStart;
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * totalCols);
            java.util.Arrays.fill(accum, 0, tileRows * totalCols, 0.0f);

            for (int kk = kStart; kk < kEnd; kk += tk) {
                int kkEnd = Math.min(kk + tk, kEnd);
                int panelDepth = kkEnd - kk;
                float[] packedA = BF16MatMulPacking.packedPanelBF16Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int tileCols = jjEnd - jj;
                    int colOffset = jj - jStart;
                    float[] packedB = packedWeights.panel(kk, jj);
                    kernel.compute(packedA, accum, packedB, tileRows, colOffset, panelDepth, totalCols, tileCols);
                }
            }

            for (int row = 0; row < tileRows; row++) {
                int outRow = outOffset + (ii + row) * n + jStart;
                int accumBase = row * totalCols;
                for (int j = 0; j < totalCols; j++) {
                    out[outRow + j] = CpuDTypeOps.toBFloat16Bits(accum[accumBase + j]);
                }
            }
        }
    }

    static void computeBlockBF16ToFloat(
            short[] a, short[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tm, int tn, int tk,
            BF16MatMulAccumulators.BF16AccumKernel kernel
    ) {
        int totalCols = jEnd - jStart;
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * totalCols);
            java.util.Arrays.fill(accum, 0, tileRows * totalCols, 0.0f);

            for (int kk = kStart; kk < kEnd; kk += tk) {
                int kkEnd = Math.min(kk + tk, kEnd);
                int panelDepth = kkEnd - kk;
                float[] packedA = BF16MatMulPacking.packedPanelBF16Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int tileCols = jjEnd - jj;
                    int colOffset = jj - jStart;
                    float[] packedB = BF16MatMulPacking.packedPanelBF16(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    kernel.compute(packedA, accum, packedB, tileRows, colOffset, panelDepth, totalCols, tileCols);
                }
            }

            for (int row = 0; row < tileRows; row++) {
                int outRow = outOffset + (ii + row) * n + jStart;
                int accumBase = row * totalCols;
                System.arraycopy(accum, accumBase, out, outRow, totalCols);
            }
        }
    }

    static void computeBlockPackedBF16ToFloat(
            short[] a, PackedLinearWeightCache.PackedFloatPanels packedWeights, float[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tm, int tn, int tk,
            BF16MatMulAccumulators.BF16AccumKernel kernel
    ) {
        int totalCols = jEnd - jStart;
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * totalCols);
            java.util.Arrays.fill(accum, 0, tileRows * totalCols, 0.0f);

            for (int kk = kStart; kk < kEnd; kk += tk) {
                int kkEnd = Math.min(kk + tk, kEnd);
                int panelDepth = kkEnd - kk;
                float[] packedA = BF16MatMulPacking.packedPanelBF16Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int tileCols = jjEnd - jj;
                    int colOffset = jj - jStart;
                    float[] packedB = packedWeights.panel(kk, jj);
                    kernel.compute(packedA, accum, packedB, tileRows, colOffset, panelDepth, totalCols, tileCols);
                }
            }

            for (int row = 0; row < tileRows; row++) {
                int outRow = outOffset + (ii + row) * n + jStart;
                int accumBase = row * totalCols;
                System.arraycopy(accum, accumBase, out, outRow, totalCols);
            }
        }
    }

    static void computeBlockF32ToBF16(
            float[] a, float[] b, short[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tm, int tn, int tk,
            BF16MatMulAccumulators.BF16AccumKernel kernel
    ) {
        int totalCols = jEnd - jStart;
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * totalCols);
            java.util.Arrays.fill(accum, 0, tileRows * totalCols, 0.0f);

            for (int kk = kStart; kk < kEnd; kk += tk) {
                int kkEnd = Math.min(kk + tk, kEnd);
                int panelDepth = kkEnd - kk;
                float[] packedA = BF16MatMulPacking.packedPanelF32Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int tileCols = jjEnd - jj;
                    int colOffset = jj - jStart;
                    float[] packedB = BF16MatMulPacking.packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    kernel.compute(packedA, accum, packedB, tileRows, colOffset, panelDepth, totalCols, tileCols);
                }
            }

            for (int row = 0; row < tileRows; row++) {
                int outRow = outOffset + (ii + row) * n + jStart;
                int accumBase = row * totalCols;
                for (int j = 0; j < totalCols; j++) {
                    out[outRow + j] = CpuDTypeOps.toBFloat16Bits(accum[accumBase + j]);
                }
            }
        }
    }

    static void computeBlockF32LeftBF16RightToBF16(
            float[] a, short[] b, short[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tm, int tn, int tk,
            BF16MatMulAccumulators.BF16AccumKernel kernel
    ) {
        int totalCols = jEnd - jStart;
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * totalCols);
            java.util.Arrays.fill(accum, 0, tileRows * totalCols, 0.0f);

            for (int kk = kStart; kk < kEnd; kk += tk) {
                int kkEnd = Math.min(kk + tk, kEnd);
                int panelDepth = kkEnd - kk;
                float[] packedA = BF16MatMulPacking.packedPanelF32Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int tileCols = jjEnd - jj;
                    int colOffset = jj - jStart;
                    float[] packedB = BF16MatMulPacking.packedPanelBF16(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    kernel.compute(packedA, accum, packedB, tileRows, colOffset, panelDepth, totalCols, tileCols);
                }
            }

            for (int row = 0; row < tileRows; row++) {
                int outRow = outOffset + (ii + row) * n + jStart;
                int accumBase = row * totalCols;
                for (int j = 0; j < totalCols; j++) {
                    out[outRow + j] = CpuDTypeOps.toBFloat16Bits(accum[accumBase + j]);
                }
            }
        }
    }

    static void computeBlockF32LeftBF16RightToFloat(
            float[] a, short[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tm, int tn, int tk,
            BF16MatMulAccumulators.BF16AccumKernel kernel
    ) {
        int totalCols = jEnd - jStart;
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * totalCols);
            java.util.Arrays.fill(accum, 0, tileRows * totalCols, 0.0f);

            for (int kk = kStart; kk < kEnd; kk += tk) {
                int kkEnd = Math.min(kk + tk, kEnd);
                int panelDepth = kkEnd - kk;
                float[] packedA = BF16MatMulPacking.packedPanelF32Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int tileCols = jjEnd - jj;
                    int colOffset = jj - jStart;
                    float[] packedB = BF16MatMulPacking.packedPanelBF16(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    kernel.compute(packedA, accum, packedB, tileRows, colOffset, panelDepth, totalCols, tileCols);
                }
            }

            for (int row = 0; row < tileRows; row++) {
                int outRow = outOffset + (ii + row) * n + jStart;
                int accumBase = row * totalCols;
                System.arraycopy(accum, accumBase, out, outRow, totalCols);
            }
        }
    }

    static void computeBlockPackedF32ToBF16(
            float[] a, PackedLinearWeightCache.PackedFloatPanels packedWeights, short[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tm, int tn, int tk,
            BF16MatMulAccumulators.BF16AccumKernel kernel
    ) {
        int totalCols = jEnd - jStart;
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * totalCols);
            java.util.Arrays.fill(accum, 0, tileRows * totalCols, 0.0f);

            for (int kk = kStart; kk < kEnd; kk += tk) {
                int kkEnd = Math.min(kk + tk, kEnd);
                int panelDepth = kkEnd - kk;
                float[] packedA = BF16MatMulPacking.packedPanelF32Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int tileCols = jjEnd - jj;
                    int colOffset = jj - jStart;
                    float[] packedB = packedWeights.panel(kk, jj);
                    kernel.compute(packedA, accum, packedB, tileRows, colOffset, panelDepth, totalCols, tileCols);
                }
            }

            for (int row = 0; row < tileRows; row++) {
                int outRow = outOffset + (ii + row) * n + jStart;
                int accumBase = row * totalCols;
                for (int j = 0; j < totalCols; j++) {
                    out[outRow + j] = CpuDTypeOps.toBFloat16Bits(accum[accumBase + j]);
                }
            }
        }
    }
}
