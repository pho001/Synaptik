package backend.cpu.kernels.linalg.matmul.f64;

final class F64MatMulPacking {
    private static final ThreadLocal<double[]> F64_PACKED_B = ThreadLocal.withInitial(() -> new double[0]);
    private static final ThreadLocal<double[]> F64_PACKED_A = ThreadLocal.withInitial(() -> new double[0]);

    private F64MatMulPacking() {
    }

    static double[] packedPanelF64(double[] b, int bOffset, int kStart, int kEnd, int jStart, int jEnd, int n) {
        int panelWidth = jEnd - jStart;
        int required = (kEnd - kStart) * panelWidth;
        double[] packed = F64_PACKED_B.get();
        if (packed.length < required) {
            packed = new double[required];
            F64_PACKED_B.set(packed);
        }
        int dst = 0;
        for (int p = kStart; p < kEnd; p++) {
            System.arraycopy(b, bOffset + p * n + jStart, packed, dst, panelWidth);
            dst += panelWidth;
        }
        return packed;
    }

    static double[] packedPanelF64LeftTransposed(double[] a, int aOffset, int kStart, int kEnd, int iStart, int iEnd, int sourceK) {
        int rows = iEnd - iStart;
        int panelDepth = kEnd - kStart;
        int required = rows * panelDepth;
        double[] packed = F64_PACKED_A.get();
        if (packed.length < required) {
            packed = new double[required];
            F64_PACKED_A.set(packed);
        }
        for (int row = 0; row < rows; row++) {
            int srcCol = iStart + row;
            int dstBase = row * panelDepth;
            for (int p = kStart; p < kEnd; p++) {
                packed[dstBase + (p - kStart)] = a[aOffset + p * sourceK + srcCol];
            }
        }
        return packed;
    }
    static double[] packedPanelF64Transposed(double[] b, int bOffset, int kStart, int kEnd, int jStart, int jEnd, int k) {
        int panelWidth = jEnd - jStart;
        int required = (kEnd - kStart) * panelWidth;
        double[] packed = F64_PACKED_B.get();
        if (packed.length < required) {
            packed = new double[required];
            F64_PACKED_B.set(packed);
        }
        int dst = 0;
        for (int p = kStart; p < kEnd; p++) {
            for (int j = jStart; j < jEnd; j++) {
                packed[dst++] = b[bOffset + j * k + p];
            }
        }
        return packed;
    }

}
