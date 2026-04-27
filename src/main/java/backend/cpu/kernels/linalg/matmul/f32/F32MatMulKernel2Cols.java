package backend.cpu.kernels.linalg.matmul.f32;

import backend.cpu.kernels.linalg.matmul.common.PackedLinearWeightCache;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

final class F32MatMulKernel2Cols {
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    private F32MatMulKernel2Cols() {
    }

    static void computeBlockF32_4x2(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 2;
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
                    computeFourRowsTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockPackedF32_4x2(
            float[] a, PackedLinearWeightCache.PackedFloatPanels packedWeights, float[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 2;
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
                    computeFourRowsTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockF32_4x2_RhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 2;
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
                    computeFourRowsTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowTwoColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockF32_4x2_LhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 2;
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
                    computeFourRowsTwoColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
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
                    computeSingleRowTwoColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeFourRowsTwoColsF32(
            float[] a, float[] out, float[] packedB,
            int aOffset, int outOffset,
            int row,
            int jStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int blockLimit,
            int vectorLimit,
            int width
    ) {
        int aRow0 = aOffset + row * k;
        int aRow1 = aOffset + (row + 1) * k;
        int aRow2 = aOffset + (row + 2) * k;
        int aRow3 = aOffset + (row + 3) * k;
        int outRow0 = outOffset + row * n;
        int outRow1 = outOffset + (row + 1) * n;
        int outRow2 = outOffset + (row + 2) * n;
        int outRow3 = outOffset + (row + 3) * n;
        int j = 0;
        for (; j < blockLimit; j += width * 2) {
            int outCol = jStart + j;
            FloatVector c00 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c01 = FloatVector.fromArray(F32, out, outRow0 + outCol + width);
            FloatVector c10 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            FloatVector c11 = FloatVector.fromArray(F32, out, outRow1 + outCol + width);
            FloatVector c20 = FloatVector.fromArray(F32, out, outRow2 + outCol);
            FloatVector c21 = FloatVector.fromArray(F32, out, outRow2 + outCol + width);
            FloatVector c30 = FloatVector.fromArray(F32, out, outRow3 + outCol);
            FloatVector c31 = FloatVector.fromArray(F32, out, outRow3 + outCol + width);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector b0 = FloatVector.fromArray(F32, packedB, packedRow + j);
                FloatVector b1 = FloatVector.fromArray(F32, packedB, packedRow + j + width);
                FloatVector a0 = FloatVector.broadcast(F32, a[aRow0 + p]);
                FloatVector a1 = FloatVector.broadcast(F32, a[aRow1 + p]);
                FloatVector a2 = FloatVector.broadcast(F32, a[aRow2 + p]);
                FloatVector a3 = FloatVector.broadcast(F32, a[aRow3 + p]);
                c00 = c00.add(a0.mul(b0));
                c01 = c01.add(a0.mul(b1));
                c10 = c10.add(a1.mul(b0));
                c11 = c11.add(a1.mul(b1));
                c20 = c20.add(a2.mul(b0));
                c21 = c21.add(a2.mul(b1));
                c30 = c30.add(a3.mul(b0));
                c31 = c31.add(a3.mul(b1));
            }
            c00.intoArray(out, outRow0 + outCol);
            c01.intoArray(out, outRow0 + outCol + width);
            c10.intoArray(out, outRow1 + outCol);
            c11.intoArray(out, outRow1 + outCol + width);
            c20.intoArray(out, outRow2 + outCol);
            c21.intoArray(out, outRow2 + outCol + width);
            c30.intoArray(out, outRow3 + outCol);
            c31.intoArray(out, outRow3 + outCol + width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = jStart + j;
            FloatVector c0 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c1 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            FloatVector c2 = FloatVector.fromArray(F32, out, outRow2 + outCol);
            FloatVector c3 = FloatVector.fromArray(F32, out, outRow3 + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector bv = FloatVector.fromArray(F32, packedB, packedRow + j);
                c0 = c0.add(FloatVector.broadcast(F32, a[aRow0 + p]).mul(bv));
                c1 = c1.add(FloatVector.broadcast(F32, a[aRow1 + p]).mul(bv));
                c2 = c2.add(FloatVector.broadcast(F32, a[aRow2 + p]).mul(bv));
                c3 = c3.add(FloatVector.broadcast(F32, a[aRow3 + p]).mul(bv));
            }
            c0.intoArray(out, outRow0 + outCol);
            c1.intoArray(out, outRow1 + outCol);
            c2.intoArray(out, outRow2 + outCol);
            c3.intoArray(out, outRow3 + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = jStart + j;
            float sum0 = out[outRow0 + outCol];
            float sum1 = out[outRow1 + outCol];
            float sum2 = out[outRow2 + outCol];
            float sum3 = out[outRow3 + outCol];
            for (int p = kStart; p < kEnd; p++) {
                float bv = packedB[(p - kStart) * panelWidth + j];
                sum0 += a[aRow0 + p] * bv;
                sum1 += a[aRow1 + p] * bv;
                sum2 += a[aRow2 + p] * bv;
                sum3 += a[aRow3 + p] * bv;
            }
            out[outRow0 + outCol] = sum0;
            out[outRow1 + outCol] = sum1;
            out[outRow2 + outCol] = sum2;
            out[outRow3 + outCol] = sum3;
        }
    }

    private static void computeSingleRowTwoColsF32(
            float[] a, float[] out, float[] packedB,
            int aOffset, int outOffset,
            int row,
            int jStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int blockLimit,
            int vectorLimit,
            int width
    ) {
        int aRow = aOffset + row * k;
        int outRow = outOffset + row * n;
        int j = 0;
        for (; j < blockLimit; j += width * 2) {
            int outCol = jStart + j;
            FloatVector c0 = FloatVector.fromArray(F32, out, outRow + outCol);
            FloatVector c1 = FloatVector.fromArray(F32, out, outRow + outCol + width);
            for (int p = kStart; p < kEnd; p++) {
                FloatVector av = FloatVector.broadcast(F32, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                c0 = c0.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j)));
                c1 = c1.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + width)));
            }
            c0.intoArray(out, outRow + outCol);
            c1.intoArray(out, outRow + outCol + width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = jStart + j;
            FloatVector acc = FloatVector.fromArray(F32, out, outRow + outCol);
            for (int p = kStart; p < kEnd; p++) {
                FloatVector av = FloatVector.broadcast(F32, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                acc = acc.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j)));
            }
            acc.intoArray(out, outRow + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = jStart + j;
            float sum = out[outRow + outCol];
            for (int p = kStart; p < kEnd; p++) {
                sum += a[aRow + p] * packedB[(p - kStart) * panelWidth + j];
            }
            out[outRow + outCol] = sum;
        }
    }

}
