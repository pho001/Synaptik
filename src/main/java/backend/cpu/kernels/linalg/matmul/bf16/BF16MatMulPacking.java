package backend.cpu.kernels.linalg.matmul.bf16;

import backend.cpu.kernels.CpuDTypeOps;

final class BF16MatMulPacking {
    private static final ThreadLocal<float[]> F32_PACKED_B = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<float[]> F32_PACKED_A = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<float[]> BF16_PACKED_A = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<float[]> BF16_PACKED_B = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<float[]> BF16_ACCUM_TILE = ThreadLocal.withInitial(() -> new float[0]);

    private BF16MatMulPacking() {
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
    static float[] packedPanelBF16(short[] b, int bOffset, int kStart, int kEnd, int jStart, int jEnd, int n) {
        int panelWidth = jEnd - jStart;
        int required = (kEnd - kStart) * panelWidth;
        float[] packed = BF16_PACKED_B.get();
        if (packed.length < required) {
            packed = new float[required];
            BF16_PACKED_B.set(packed);
        }
        int dst = 0;
        for (int p = kStart; p < kEnd; p++) {
            int srcBase = bOffset + p * n + jStart;
            for (int j = 0; j < panelWidth; j++) {
                packed[dst++] = CpuDTypeOps.fromBFloat16Bits(b[srcBase + j]);
            }
        }
        return packed;
    }

    static float[] packedPanelBF16Left(short[] a, int aOffset, int iStart, int iEnd, int kStart, int kEnd, int sourceK) {
        int rows = iEnd - iStart;
        int panelDepth = kEnd - kStart;
        int required = rows * panelDepth;
        float[] packed = BF16_PACKED_A.get();
        if (packed.length < required) {
            packed = new float[required];
            BF16_PACKED_A.set(packed);
        }
        int dst = 0;
        for (int i = iStart; i < iEnd; i++) {
            int srcBase = aOffset + i * sourceK + kStart;
            for (int p = 0; p < panelDepth; p++) {
                packed[dst++] = CpuDTypeOps.fromBFloat16Bits(a[srcBase + p]);
            }
        }
        return packed;
    }

    static float[] packedPanelF32Left(float[] a, int aOffset, int iStart, int iEnd, int kStart, int kEnd, int sourceK) {
        int rows = iEnd - iStart;
        int panelDepth = kEnd - kStart;
        int required = rows * panelDepth;
        float[] packed = F32_PACKED_A.get();
        if (packed.length < required) {
            packed = new float[required];
            F32_PACKED_A.set(packed);
        }
        int dst = 0;
        for (int i = iStart; i < iEnd; i++) {
            int srcBase = aOffset + i * sourceK + kStart;
            System.arraycopy(a, srcBase, packed, dst, panelDepth);
            dst += panelDepth;
        }
        return packed;
    }

    static float[] bf16AccumTile(int required) {
        float[] tile = BF16_ACCUM_TILE.get();
        if (tile.length < required) {
            tile = new float[required];
            BF16_ACCUM_TILE.set(tile);
        }
        return tile;
    }

}
