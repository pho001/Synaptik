package backend.cpu.kernels.linalg.matmul.f64;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

final class F64MatMulKernelSupport {
    static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;

    private F64MatMulKernelSupport() {
    }

    static void computeFourRowsOneColF64(
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

    static void computeTwoRowsOneColF64(
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

    static void computeSingleRowOneColF64(
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

    static void computeTwoRowsTwoColsF64(
            double[] a, double[] out, double[] packedB,
            int aOffset, int outOffset,
            int row,
            int colStart,
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
        for (; j < blockLimit; j += width * 2) {
            int outCol = colStart + j;
            DoubleVector c00 = DoubleVector.fromArray(F64, out, outRow0 + outCol);
            DoubleVector c01 = DoubleVector.fromArray(F64, out, outRow0 + outCol + width);
            DoubleVector c10 = DoubleVector.fromArray(F64, out, outRow1 + outCol);
            DoubleVector c11 = DoubleVector.fromArray(F64, out, outRow1 + outCol + width);
            for (int p = kStart; p < kEnd; p++) {
                int packedRow = (p - kStart) * panelWidth;
                DoubleVector b0 = DoubleVector.fromArray(F64, packedB, packedRow + j);
                DoubleVector b1 = DoubleVector.fromArray(F64, packedB, packedRow + j + width);
                DoubleVector a0 = DoubleVector.broadcast(F64, a[aRow0 + p]);
                DoubleVector a1 = DoubleVector.broadcast(F64, a[aRow1 + p]);
                c00 = c00.add(a0.mul(b0));
                c01 = c01.add(a0.mul(b1));
                c10 = c10.add(a1.mul(b0));
                c11 = c11.add(a1.mul(b1));
            }
            c00.intoArray(out, outRow0 + outCol);
            c01.intoArray(out, outRow0 + outCol + width);
            c10.intoArray(out, outRow1 + outCol);
            c11.intoArray(out, outRow1 + outCol + width);
        }
        for (; j < vectorLimit; j += width) {
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

    static void computeSingleRowTwoColsF64(
            double[] a, double[] out, double[] packedB,
            int aOffset, int outOffset,
            int row,
            int colStart,
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
            int outCol = colStart + j;
            DoubleVector c0 = DoubleVector.fromArray(F64, out, outRow + outCol);
            DoubleVector c1 = DoubleVector.fromArray(F64, out, outRow + outCol + width);
            for (int p = kStart; p < kEnd; p++) {
                DoubleVector av = DoubleVector.broadcast(F64, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                c0 = c0.add(av.mul(DoubleVector.fromArray(F64, packedB, packedRow + j)));
                c1 = c1.add(av.mul(DoubleVector.fromArray(F64, packedB, packedRow + j + width)));
            }
            c0.intoArray(out, outRow + outCol);
            c1.intoArray(out, outRow + outCol + width);
        }
        for (; j < vectorLimit; j += width) {
            int outCol = colStart + j;
            DoubleVector acc = DoubleVector.fromArray(F64, out, outRow + outCol);
            for (int p = kStart; p < kEnd; p++) {
                DoubleVector av = DoubleVector.broadcast(F64, a[aRow + p]);
                int packedRow = (p - kStart) * panelWidth;
                acc = acc.add(av.mul(DoubleVector.fromArray(F64, packedB, packedRow + j)));
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
