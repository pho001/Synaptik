package backend.kernels.cpu.linalg.matmul.f64;

import backend.kernels.cpu.linalg.matmul.common.PackedLinearWeightCache;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

final class F64MatMulKernel1Col {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;

    private F64MatMulKernel1Col() {
    }

    static void computeBlockF64_2x1(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = F64MatMulPacking.packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    static void computeBlockPackedF64_2x1(
            double[] a, PackedLinearWeightCache.F64PackedWeights packedWeights, double[] out,
            int aOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    static void computeBlockF64_2x1_RhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = F64MatMulPacking.packedPanelF64Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    static void computeBlockF64_2x1_LhsTransposed(
            double[] a, double[] b, double[] out,
            int aOffset, int bOffset, int outOffset,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int sourceK,
            int tn, int tk
    ) {
        int width = F64.length();
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
                    double[] packedB = F64MatMulPacking.packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                    computeTwoRowsOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
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
                    computeSingleRowOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
                }
            }
        }
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
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = F64MatMulPacking.packedPanelF64(b, bOffset, kk, kkEnd, jj, jjEnd, n);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
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
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = F64MatMulPacking.packedPanelF64Transposed(b, bOffset, kk, kkEnd, jj, jjEnd, k);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
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
        int width = F64.length();
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
                    computeFourRowsOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
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
                    computeTwoRowsOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
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
                    computeSingleRowOneColF64(packedA, out, packedB, 0, tileOutOffset, 0, jj, 0, panelDepth, n, panelDepth, panelWidth, vectorLimit);
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
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                int panelWidth = jjEnd - jj;
                int vectorLimit = panelWidth - (panelWidth % width);
                double[] packedB = packedWeights.panel(kk, jj);
                int i = iStart;
                for (; i + 3 < iEnd; i += 4) {
                    computeFourRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i + 1 < iEnd; i += 2) {
                    computeTwoRowsOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
                for (; i < iEnd; i++) {
                    computeSingleRowOneColF64(a, out, packedB, aOffset, outOffset, i, jj, kk, kkEnd, n, k, panelWidth, vectorLimit);
                }
            }
        }
    }

    private static void computeFourRowsOneColF64(
            double[] a, double[] out, double[] packedB,
            int aOffset, int outOffset,
            int row,
            int colStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int vectorLimit
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
        for (; j < vectorLimit; j += F64.length()) {
            int outCol = colStart + j;
            DoubleVector c0 = DoubleVector.fromArray(F64, out, outRow0 + outCol);
            DoubleVector c1 = DoubleVector.fromArray(F64, out, outRow1 + outCol);
            DoubleVector c2 = DoubleVector.fromArray(F64, out, outRow2 + outCol);
            DoubleVector c3 = DoubleVector.fromArray(F64, out, outRow3 + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                DoubleVector bv = DoubleVector.fromArray(F64, packedB, packedRow + j);
                c0 = c0.add(DoubleVector.broadcast(F64, a[aRow0 + p]).mul(bv));
                c1 = c1.add(DoubleVector.broadcast(F64, a[aRow1 + p]).mul(bv));
                c2 = c2.add(DoubleVector.broadcast(F64, a[aRow2 + p]).mul(bv));
                c3 = c3.add(DoubleVector.broadcast(F64, a[aRow3 + p]).mul(bv));
            }
            c0.intoArray(out, outRow0 + outCol);
            c1.intoArray(out, outRow1 + outCol);
            c2.intoArray(out, outRow2 + outCol);
            c3.intoArray(out, outRow3 + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = colStart + j;
            double sum0 = out[outRow0 + outCol];
            double sum1 = out[outRow1 + outCol];
            double sum2 = out[outRow2 + outCol];
            double sum3 = out[outRow3 + outCol];
            for (int p = kStart; p < kEnd; p++) {
                double bv = packedB[(p - kStart) * panelWidth + j];
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

    private static void computeTwoRowsOneColF64(
            double[] a, double[] out, double[] packedB,
            int aOffset, int outOffset,
            int row,
            int colStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int vectorLimit
    ) {
        int aRow0 = aOffset + row * k;
        int aRow1 = aOffset + (row + 1) * k;
        int outRow0 = outOffset + row * n;
        int outRow1 = outOffset + (row + 1) * n;
        int j = 0;
        for (; j < vectorLimit; j += F64.length()) {
            int outCol = colStart + j;
            DoubleVector c0 = DoubleVector.fromArray(F64, out, outRow0 + outCol);
            DoubleVector c1 = DoubleVector.fromArray(F64, out, outRow1 + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                DoubleVector bv = DoubleVector.fromArray(F64, packedB, packedRow + j);
                c0 = c0.add(DoubleVector.broadcast(F64, a[aRow0 + p]).mul(bv));
                c1 = c1.add(DoubleVector.broadcast(F64, a[aRow1 + p]).mul(bv));
            }
            c0.intoArray(out, outRow0 + outCol);
            c1.intoArray(out, outRow1 + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = colStart + j;
            double sum0 = out[outRow0 + outCol];
            double sum1 = out[outRow1 + outCol];
            for (int p = kStart; p < kEnd; p++) {
                double bv = packedB[(p - kStart) * panelWidth + j];
                sum0 += a[aRow0 + p] * bv;
                sum1 += a[aRow1 + p] * bv;
            }
            out[outRow0 + outCol] = sum0;
            out[outRow1 + outCol] = sum1;
        }
    }

    private static void computeSingleRowOneColF64(
            double[] a, double[] out, double[] packedB,
            int aOffset, int outOffset,
            int row,
            int colStart,
            int kStart, int kEnd,
            int n, int k,
            int panelWidth,
            int vectorLimit
    ) {
        int aRow = aOffset + row * k;
        int outRow = outOffset + row * n;
        int j = 0;
        for (; j < vectorLimit; j += F64.length()) {
            int outCol = colStart + j;
            DoubleVector acc = DoubleVector.fromArray(F64, out, outRow + outCol);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                acc = acc.add(DoubleVector.broadcast(F64, a[aRow + p]).mul(DoubleVector.fromArray(F64, packedB, packedRow + j)));
            }
            acc.intoArray(out, outRow + outCol);
        }
        for (; j < panelWidth; j++) {
            int outCol = colStart + j;
            double sum = out[outRow + outCol];
            for (int p = kStart; p < kEnd; p++) {
                sum += a[aRow + p] * packedB[(p - kStart) * panelWidth + j];
            }
            out[outRow + outCol] = sum;
        }
    }
}
