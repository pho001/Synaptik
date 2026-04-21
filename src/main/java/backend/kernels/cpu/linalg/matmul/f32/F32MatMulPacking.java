package backend.kernels.cpu.linalg.matmul.f32;

final class F32MatMulPacking {
    private static final ThreadLocal<float[]> F32_PACKED_B = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<float[]> F32_PACKED_A = ThreadLocal.withInitial(() -> new float[0]);

    private F32MatMulPacking() {
    }

    static float[] packedPanelF32(float[] b, int bOffset, int kStart, int kEnd, int jStart, int jEnd, int n) {
        int panelWidth = jEnd - jStart;
        int required = (kEnd - kStart) * panelWidth;
        float[] packed = F32_PACKED_B.get();
        if (packed.length < required) {
            packed = new float[required];
            F32_PACKED_B.set(packed);
        }
        int dst = 0;
        for (int p = kStart; p < kEnd; p++) {
            System.arraycopy(b, bOffset + p * n + jStart, packed, dst, panelWidth);
            dst += panelWidth;
        }
        return packed;
    }

    static float[] packedPanelF32LeftTransposed(float[] a, int aOffset, int kStart, int kEnd, int iStart, int iEnd, int sourceK) {
        int rows = iEnd - iStart;
        int panelDepth = kEnd - kStart;
        int required = rows * panelDepth;
        float[] packed = F32_PACKED_A.get();
        if (packed.length < required) {
            packed = new float[required];
            F32_PACKED_A.set(packed);
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

    static float[] packedPanelF32Transposed(float[] b, int bOffset, int kStart, int kEnd, int jStart, int jEnd, int k) {
        int panelWidth = jEnd - jStart;
        int required = (kEnd - kStart) * panelWidth;
        float[] packed = F32_PACKED_B.get();
        if (packed.length < required) {
            packed = new float[required];
            F32_PACKED_B.set(packed);
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
