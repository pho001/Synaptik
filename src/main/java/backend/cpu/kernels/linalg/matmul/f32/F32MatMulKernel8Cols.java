package backend.cpu.kernels.linalg.matmul.f32;

import backend.cpu.kernels.linalg.matmul.common.PackedLinearWeightCache;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

final class F32MatMulKernel8Cols {
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    private F32MatMulKernel8Cols() {
    }

    static void computeBlockF32_2x8(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 8;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = F32MatMulPacking.packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockPackedF32_2x8(
            float[] a, PackedLinearWeightCache.PackedFloatPanels packedWeights, float[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 8;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockF32_2x8_RhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 8;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                float[] packedB = F32MatMulPacking.packedPanelF32Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowEightColsF32(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    static void computeBlockF32_2x8_LhsTransposed(
            float[] a, float[] b, float[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F32.length();
        int vectorBlockWidth = width * 8;
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            int panelDepth = kkEnd - kk;
            int i = iStart;
            for (; i + 1 < iEnd; i += 2) {
                float[] packedA = F32MatMulPacking.packedPanelF32LeftTransposed(a, aOffset, kk, kkEnd, i, i + 2, sourceK);
                int tileOutOffset = outOffset + i * n;
                for (int jj = jStart; jj < jEnd; jj += tn) {
                    int jjEnd = Math.min(jj + tn, jEnd);
                    int panelWidth = jjEnd - jj;
                    int vectorLimit = panelWidth - (panelWidth % width);
                    int blockLimit = panelWidth - (panelWidth % vectorBlockWidth);
                    float[] packedB = F32MatMulPacking.packedPanelF32(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeTwoRowsEightColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
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
                    computeSingleRowEightColsF32(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, blockLimit, vectorLimit, width);
                }
            }
        }
    }

    private static void computeTwoRowsEightColsF32(
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
        int outRow0 = outOffset + row * n;
        int outRow1 = outOffset + (row + 1) * n;
        int j = 0;
        for (; j < blockLimit; j += width * 8) {
            int outCol = jStart + j;
            FloatVector c00 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c01 = FloatVector.fromArray(F32, out, outRow0 + outCol + width);
            FloatVector c02 = FloatVector.fromArray(F32, out, outRow0 + outCol + 2 * width);
            FloatVector c03 = FloatVector.fromArray(F32, out, outRow0 + outCol + 3 * width);
            FloatVector c04 = FloatVector.fromArray(F32, out, outRow0 + outCol + 4 * width);
            FloatVector c05 = FloatVector.fromArray(F32, out, outRow0 + outCol + 5 * width);
            FloatVector c06 = FloatVector.fromArray(F32, out, outRow0 + outCol + 6 * width);
            FloatVector c07 = FloatVector.fromArray(F32, out, outRow0 + outCol + 7 * width);
            FloatVector c10 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            FloatVector c11 = FloatVector.fromArray(F32, out, outRow1 + outCol + width);
            FloatVector c12 = FloatVector.fromArray(F32, out, outRow1 + outCol + 2 * width);
            FloatVector c13 = FloatVector.fromArray(F32, out, outRow1 + outCol + 3 * width);
            FloatVector c14 = FloatVector.fromArray(F32, out, outRow1 + outCol + 4 * width);
            FloatVector c15 = FloatVector.fromArray(F32, out, outRow1 + outCol + 5 * width);
            FloatVector c16 = FloatVector.fromArray(F32, out, outRow1 + outCol + 6 * width);
            FloatVector c17 = FloatVector.fromArray(F32, out, outRow1 + outCol + 7 * width);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector b0 = FloatVector.fromArray(F32, packedB, packedRow + j);
                FloatVector b1 = FloatVector.fromArray(F32, packedB, packedRow + j + width);
                FloatVector b2 = FloatVector.fromArray(F32, packedB, packedRow + j + 2 * width);
                FloatVector b3 = FloatVector.fromArray(F32, packedB, packedRow + j + 3 * width);
                FloatVector b4 = FloatVector.fromArray(F32, packedB, packedRow + j + 4 * width);
                FloatVector b5 = FloatVector.fromArray(F32, packedB, packedRow + j + 5 * width);
                FloatVector b6 = FloatVector.fromArray(F32, packedB, packedRow + j + 6 * width);
                FloatVector b7 = FloatVector.fromArray(F32, packedB, packedRow + j + 7 * width);
                FloatVector a0 = FloatVector.broadcast(F32, a[aRow0 + p]);
                FloatVector a1 = FloatVector.broadcast(F32, a[aRow1 + p]);
                c00 = c00.add(a0.mul(b0));
                c01 = c01.add(a0.mul(b1));
                c02 = c02.add(a0.mul(b2));
                c03 = c03.add(a0.mul(b3));
                c04 = c04.add(a0.mul(b4));
                c05 = c05.add(a0.mul(b5));
                c06 = c06.add(a0.mul(b6));
                c07 = c07.add(a0.mul(b7));
                c10 = c10.add(a1.mul(b0));
                c11 = c11.add(a1.mul(b1));
                c12 = c12.add(a1.mul(b2));
                c13 = c13.add(a1.mul(b3));
                c14 = c14.add(a1.mul(b4));
                c15 = c15.add(a1.mul(b5));
                c16 = c16.add(a1.mul(b6));
                c17 = c17.add(a1.mul(b7));
            }
            c00.intoArray(out, outRow0 + outCol);
            c01.intoArray(out, outRow0 + outCol + width);
            c02.intoArray(out, outRow0 + outCol + 2 * width);
            c03.intoArray(out, outRow0 + outCol + 3 * width);
            c04.intoArray(out, outRow0 + outCol + 4 * width);
            c05.intoArray(out, outRow0 + outCol + 5 * width);
            c06.intoArray(out, outRow0 + outCol + 6 * width);
            c07.intoArray(out, outRow0 + outCol + 7 * width);
            c10.intoArray(out, outRow1 + outCol);
            c11.intoArray(out, outRow1 + outCol + width);
            c12.intoArray(out, outRow1 + outCol + 2 * width);
            c13.intoArray(out, outRow1 + outCol + 3 * width);
            c14.intoArray(out, outRow1 + outCol + 4 * width);
            c15.intoArray(out, outRow1 + outCol + 5 * width);
            c16.intoArray(out, outRow1 + outCol + 6 * width);
            c17.intoArray(out, outRow1 + outCol + 7 * width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = jStart + j;
            FloatVector c0 = FloatVector.fromArray(F32, out, outRow0 + outCol);
            FloatVector c1 = FloatVector.fromArray(F32, out, outRow1 + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                FloatVector bv = FloatVector.fromArray(F32, packedB, packedRow + j);
                c0 = c0.add(FloatVector.broadcast(F32, a[aRow0 + p]).mul(bv));
                c1 = c1.add(FloatVector.broadcast(F32, a[aRow1 + p]).mul(bv));
            }
            c0.intoArray(out, outRow0 + outCol);
            c1.intoArray(out, outRow1 + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = jStart + j;
            float sum0 = out[outRow0 + outCol];
            float sum1 = out[outRow1 + outCol];
            for (int p = kStart; p < kEnd; p++) {
                float bv = packedB[(p - kStart) * panelWidth + j];
                sum0 += a[aRow0 + p] * bv;
                sum1 += a[aRow1 + p] * bv;
            }
            out[outRow0 + outCol] = sum0;
            out[outRow1 + outCol] = sum1;
        }
    }

    private static void computeSingleRowEightColsF32(
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
        for (; j < blockLimit; j += width * 8) {
            int outCol = jStart + j;
            FloatVector c0 = FloatVector.fromArray(F32, out, outRow + outCol);
            FloatVector c1 = FloatVector.fromArray(F32, out, outRow + outCol + width);
            FloatVector c2 = FloatVector.fromArray(F32, out, outRow + outCol + 2 * width);
            FloatVector c3 = FloatVector.fromArray(F32, out, outRow + outCol + 3 * width);
            FloatVector c4 = FloatVector.fromArray(F32, out, outRow + outCol + 4 * width);
            FloatVector c5 = FloatVector.fromArray(F32, out, outRow + outCol + 5 * width);
            FloatVector c6 = FloatVector.fromArray(F32, out, outRow + outCol + 6 * width);
            FloatVector c7 = FloatVector.fromArray(F32, out, outRow + outCol + 7 * width);
            for (int p = kStart; p < kEnd; p++) {
                FloatVector av = FloatVector.broadcast(F32, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                c0 = c0.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j)));
                c1 = c1.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + width)));
                c2 = c2.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 2 * width)));
                c3 = c3.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 3 * width)));
                c4 = c4.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 4 * width)));
                c5 = c5.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 5 * width)));
                c6 = c6.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 6 * width)));
                c7 = c7.add(av.mul(FloatVector.fromArray(F32, packedB, packedRow + j + 7 * width)));
            }
            c0.intoArray(out, outRow + outCol);
            c1.intoArray(out, outRow + outCol + width);
            c2.intoArray(out, outRow + outCol + 2 * width);
            c3.intoArray(out, outRow + outCol + 3 * width);
            c4.intoArray(out, outRow + outCol + 4 * width);
            c5.intoArray(out, outRow + outCol + 5 * width);
            c6.intoArray(out, outRow + outCol + 6 * width);
            c7.intoArray(out, outRow + outCol + 7 * width);
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
