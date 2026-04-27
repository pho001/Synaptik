package backend.cpu.kernels.linalg.matmul.bf16;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.linalg.matmul.common.PackedLinearWeightCache;

final class BF16MatMulBlockKernels {
    private BF16MatMulBlockKernels() {
    }

    private static void storeTileAsBF16(float[] accum, short[] out, int outRowBase, int tileRows, int tileCols, int n) {
        for (int row = 0; row < tileRows; row++) {
            int outRow = outRowBase + row * n;
            int accumBase = row * tileCols;
            for (int j = 0; j < tileCols; j++) {
                out[outRow + j] = CpuDTypeOps.toBFloat16Bits(accum[accumBase + j]);
            }
        }
    }

    private static void storeTileAsFloat(float[] accum, float[] out, int outRowBase, int tileRows, int tileCols, int n) {
        for (int row = 0; row < tileRows; row++) {
            System.arraycopy(accum, row * tileCols, out, outRowBase + row * n, tileCols);
        }
    }

    static void computeBlockBF16(
            short[] a, short[] b, short[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tm, int tn, int tk,
            BF16AccumKernel kernel
    ) {
        for (int jj = jStart; jj < jEnd; jj += tn) {
            int jjEnd = Math.min(jj + tn, jEnd);
            int tileCols = jjEnd - jj;
            float[] packedBStrip = BF16MatMulPacking.packedPanelBF16(b, bOffset, kStart, kEnd, jj, jjEnd, n);
            for (int ii = iStart; ii < iEnd; ii += tm) {
                int iiEnd = Math.min(ii + tm, iEnd);
                int tileRows = iiEnd - ii;
                float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * tileCols);
                java.util.Arrays.fill(accum, 0, tileRows * tileCols, 0.0f);
                for (int kk = kStart; kk < kEnd; kk += tk) {
                    int kkEnd = Math.min(kk + tk, kEnd);
                    int panelDepth = kkEnd - kk;
                    float[] packedA = BF16MatMulPacking.packedPanelBF16Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                    int packedBOffset = (kk - kStart) * tileCols;
                    kernel.compute(packedA, accum, packedBStrip, packedBOffset, tileRows, 0, panelDepth, tileCols, tileCols);
                }
                storeTileAsBF16(accum, out, outOffset + ii * n + jj, tileRows, tileCols, n);
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
            BF16AccumKernel kernel
    ) {
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int tileCols = jjEnd - jj;
                float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * tileCols);
                java.util.Arrays.fill(accum, 0, tileRows * tileCols, 0.0f);
                for (int kk = kStart; kk < kEnd; kk += tk) {
                    int kkEnd = Math.min(kk + tk, kEnd);
                    int panelDepth = kkEnd - kk;
                    float[] packedA = BF16MatMulPacking.packedPanelBF16Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                    float[] packedB = packedWeights.panel(kk, jj);
                    kernel.compute(packedA, accum, packedB, 0, tileRows, 0, panelDepth, tileCols, tileCols);
                }
                storeTileAsBF16(accum, out, outOffset + ii * n + jj, tileRows, tileCols, n);
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
            BF16AccumKernel kernel
    ) {
        for (int jj = jStart; jj < jEnd; jj += tn) {
            int jjEnd = Math.min(jj + tn, jEnd);
            int tileCols = jjEnd - jj;
            float[] packedBStrip = BF16MatMulPacking.packedPanelBF16(b, bOffset, kStart, kEnd, jj, jjEnd, n);
            for (int ii = iStart; ii < iEnd; ii += tm) {
                int iiEnd = Math.min(ii + tm, iEnd);
                int tileRows = iiEnd - ii;
                float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * tileCols);
                java.util.Arrays.fill(accum, 0, tileRows * tileCols, 0.0f);
                for (int kk = kStart; kk < kEnd; kk += tk) {
                    int kkEnd = Math.min(kk + tk, kEnd);
                    int panelDepth = kkEnd - kk;
                    float[] packedA = BF16MatMulPacking.packedPanelBF16Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                    int packedBOffset = (kk - kStart) * tileCols;
                    kernel.compute(packedA, accum, packedBStrip, packedBOffset, tileRows, 0, panelDepth, tileCols, tileCols);
                }
                storeTileAsFloat(accum, out, outOffset + ii * n + jj, tileRows, tileCols, n);
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
            BF16AccumKernel kernel
    ) {
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int tileCols = jjEnd - jj;
                float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * tileCols);
                java.util.Arrays.fill(accum, 0, tileRows * tileCols, 0.0f);
                for (int kk = kStart; kk < kEnd; kk += tk) {
                    int kkEnd = Math.min(kk + tk, kEnd);
                    int panelDepth = kkEnd - kk;
                    float[] packedA = BF16MatMulPacking.packedPanelBF16Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                    float[] packedB = packedWeights.panel(kk, jj);
                    kernel.compute(packedA, accum, packedB, 0, tileRows, 0, panelDepth, tileCols, tileCols);
                }
                storeTileAsFloat(accum, out, outOffset + ii * n + jj, tileRows, tileCols, n);
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
            BF16AccumKernel kernel
    ) {
        for (int jj = jStart; jj < jEnd; jj += tn) {
            int jjEnd = Math.min(jj + tn, jEnd);
            int tileCols = jjEnd - jj;
            float[] packedBStrip = BF16MatMulPacking.packedPanelF32(b, bOffset, kStart, kEnd, jj, jjEnd, n);
            for (int ii = iStart; ii < iEnd; ii += tm) {
                int iiEnd = Math.min(ii + tm, iEnd);
                int tileRows = iiEnd - ii;
                float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * tileCols);
                java.util.Arrays.fill(accum, 0, tileRows * tileCols, 0.0f);
                for (int kk = kStart; kk < kEnd; kk += tk) {
                    int kkEnd = Math.min(kk + tk, kEnd);
                    int panelDepth = kkEnd - kk;
                    float[] packedA = BF16MatMulPacking.packedPanelF32Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                    int packedBOffset = (kk - kStart) * tileCols;
                    kernel.compute(packedA, accum, packedBStrip, packedBOffset, tileRows, 0, panelDepth, tileCols, tileCols);
                }
                storeTileAsBF16(accum, out, outOffset + ii * n + jj, tileRows, tileCols, n);
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
            BF16AccumKernel kernel
    ) {
        for (int jj = jStart; jj < jEnd; jj += tn) {
            int jjEnd = Math.min(jj + tn, jEnd);
            int tileCols = jjEnd - jj;
            float[] packedBStrip = BF16MatMulPacking.packedPanelBF16(b, bOffset, kStart, kEnd, jj, jjEnd, n);
            for (int ii = iStart; ii < iEnd; ii += tm) {
                int iiEnd = Math.min(ii + tm, iEnd);
                int tileRows = iiEnd - ii;
                float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * tileCols);
                java.util.Arrays.fill(accum, 0, tileRows * tileCols, 0.0f);
                for (int kk = kStart; kk < kEnd; kk += tk) {
                    int kkEnd = Math.min(kk + tk, kEnd);
                    int panelDepth = kkEnd - kk;
                    float[] packedA = BF16MatMulPacking.packedPanelF32Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                    int packedBOffset = (kk - kStart) * tileCols;
                    kernel.compute(packedA, accum, packedBStrip, packedBOffset, tileRows, 0, panelDepth, tileCols, tileCols);
                }
                storeTileAsBF16(accum, out, outOffset + ii * n + jj, tileRows, tileCols, n);
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
            BF16AccumKernel kernel
    ) {
        for (int jj = jStart; jj < jEnd; jj += tn) {
            int jjEnd = Math.min(jj + tn, jEnd);
            int tileCols = jjEnd - jj;
            float[] packedBStrip = BF16MatMulPacking.packedPanelBF16(b, bOffset, kStart, kEnd, jj, jjEnd, n);
            for (int ii = iStart; ii < iEnd; ii += tm) {
                int iiEnd = Math.min(ii + tm, iEnd);
                int tileRows = iiEnd - ii;
                float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * tileCols);
                java.util.Arrays.fill(accum, 0, tileRows * tileCols, 0.0f);
                for (int kk = kStart; kk < kEnd; kk += tk) {
                    int kkEnd = Math.min(kk + tk, kEnd);
                    int panelDepth = kkEnd - kk;
                    float[] packedA = BF16MatMulPacking.packedPanelF32Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                    int packedBOffset = (kk - kStart) * tileCols;
                    kernel.compute(packedA, accum, packedBStrip, packedBOffset, tileRows, 0, panelDepth, tileCols, tileCols);
                }
                storeTileAsFloat(accum, out, outOffset + ii * n + jj, tileRows, tileCols, n);
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
            BF16AccumKernel kernel
    ) {
        for (int ii = iStart; ii < iEnd; ii += tm) {
            int iiEnd = Math.min(ii + tm, iEnd);
            int tileRows = iiEnd - ii;
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int tileCols = jjEnd - jj;
                float[] accum = BF16MatMulPacking.bf16AccumTile(tileRows * tileCols);
                java.util.Arrays.fill(accum, 0, tileRows * tileCols, 0.0f);
                for (int kk = kStart; kk < kEnd; kk += tk) {
                    int kkEnd = Math.min(kk + tk, kEnd);
                    int panelDepth = kkEnd - kk;
                    float[] packedA = BF16MatMulPacking.packedPanelF32Left(a, aOffset, ii, iiEnd, kk, kkEnd, k);
                    float[] packedB = packedWeights.panel(kk, jj);
                    kernel.compute(packedA, accum, packedB, 0, tileRows, 0, panelDepth, tileCols, tileCols);
                }
                storeTileAsBF16(accum, out, outOffset + ii * n + jj, tileRows, tileCols, n);
            }
        }
    }
}
