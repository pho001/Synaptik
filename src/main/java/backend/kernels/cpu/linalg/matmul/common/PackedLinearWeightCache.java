package backend.kernels.cpu.linalg.matmul.common;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import tensor.Tensor;
import tensor.TensorStorage;

public final class PackedLinearWeightCache {
    public interface PackedFloatPanels {
        float[] panel(int kStart, int jStart);
    }

    private volatile F32PackedWeights cachedF32;
    private volatile F64PackedWeights cachedF64;
    private volatile BF16PackedWeights cachedBF16;

    public F32PackedWeights requireF32(Tensor weight, ResolvedMatMulHints hints) {
        if (weight == null || hints == null || !weight.isContiguous() || weight.getShapeUnsafe().length != 2) {
            return null;
        }
        float[] data = weight.getFloat32Data();
        if (data == null) {
            return null;
        }
        TensorStorage storage = weight.getStorage();
        long version = weight.storageVersion();
        int[] shape = weight.getShapeUnsafe();
        int baseOffset = weight.getStorageOffsetUnsafe();
        F32PackedWeights cached = cachedF32;
        if (cached != null && cached.matches(storage, version, baseOffset, shape[0], shape[1], hints.tileN(), hints.tileK())) {
            return cached;
        }
        synchronized (this) {
            cached = cachedF32;
            if (cached != null && cached.matches(storage, version, baseOffset, shape[0], shape[1], hints.tileN(), hints.tileK())) {
                return cached;
            }
            cached = F32PackedWeights.build(storage, version, data, baseOffset, shape[0], shape[1], hints.tileN(), hints.tileK());
            cachedF32 = cached;
            return cached;
        }
    }

    public F64PackedWeights requireF64(Tensor weight, ResolvedMatMulHints hints) {
        if (weight == null || hints == null || !weight.isContiguous() || weight.getShapeUnsafe().length != 2) {
            return null;
        }
        double[] data = weight.getFloat64Data();
        if (data == null) {
            return null;
        }
        TensorStorage storage = weight.getStorage();
        long version = weight.storageVersion();
        int[] shape = weight.getShapeUnsafe();
        int baseOffset = weight.getStorageOffsetUnsafe();
        F64PackedWeights cached = cachedF64;
        if (cached != null && cached.matches(storage, version, baseOffset, shape[0], shape[1], hints.tileN(), hints.tileK())) {
            return cached;
        }
        synchronized (this) {
            cached = cachedF64;
            if (cached != null && cached.matches(storage, version, baseOffset, shape[0], shape[1], hints.tileN(), hints.tileK())) {
                return cached;
            }
            cached = F64PackedWeights.build(storage, version, data, baseOffset, shape[0], shape[1], hints.tileN(), hints.tileK());
            cachedF64 = cached;
            return cached;
        }
    }

    public BF16PackedWeights requireBF16(Tensor weight, ResolvedMatMulHints hints) {
        if (weight == null || hints == null || !weight.isContiguous() || weight.getShapeUnsafe().length != 2) {
            return null;
        }
        short[] data = weight.getBFloat16Data();
        if (data == null) {
            return null;
        }
        TensorStorage storage = weight.getStorage();
        long version = weight.storageVersion();
        int[] shape = weight.getShapeUnsafe();
        int baseOffset = weight.getStorageOffsetUnsafe();
        BF16PackedWeights cached = cachedBF16;
        if (cached != null && cached.matches(storage, version, baseOffset, shape[0], shape[1], hints.tileN(), hints.tileK())) {
            return cached;
        }
        synchronized (this) {
            cached = cachedBF16;
            if (cached != null && cached.matches(storage, version, baseOffset, shape[0], shape[1], hints.tileN(), hints.tileK())) {
                return cached;
            }
            cached = BF16PackedWeights.build(storage, version, data, baseOffset, shape[0], shape[1], hints.tileN(), hints.tileK());
            cachedBF16 = cached;
            return cached;
        }
    }

    public static final class F32PackedWeights implements PackedFloatPanels {
        private final TensorStorage storage;
        private final long version;
        private final int baseOffset;
        private final int k;
        private final int n;
        private final int tileN;
        private final int tileK;
        private final int blockCols;
        private final float[][] panels;

        private F32PackedWeights(
                TensorStorage storage,
                long version,
                int baseOffset,
                int k,
                int n,
                int tileN,
                int tileK,
                int blockCols,
                float[][] panels
        ) {
            this.storage = storage;
            this.version = version;
            this.baseOffset = baseOffset;
            this.k = k;
            this.n = n;
            this.tileN = tileN;
            this.tileK = tileK;
            this.blockCols = blockCols;
            this.panels = panels;
        }

        static F32PackedWeights build(
                TensorStorage storage,
                long version,
                float[] data,
                int baseOffset,
                int k,
                int n,
                int tileN,
                int tileK
        ) {
            int blockCols = (n + tileN - 1) / tileN;
            int blockRows = (k + tileK - 1) / tileK;
            float[][] panels = new float[blockRows * blockCols][];
            int panelIndex = 0;
            for (int kk = 0; kk < k; kk += tileK) {
                int kkEnd = Math.min(kk + tileK, k);
                for (int jj = 0; jj < n; jj += tileN) {
                    int jjEnd = Math.min(jj + tileN, n);
                    int panelWidth = jjEnd - jj;
                    float[] packed = new float[(kkEnd - kk) * panelWidth];
                    int dst = 0;
                    for (int p = kk; p < kkEnd; p++) {
                        System.arraycopy(data, baseOffset + p * n + jj, packed, dst, panelWidth);
                        dst += panelWidth;
                    }
                    panels[panelIndex++] = packed;
                }
            }
            return new F32PackedWeights(storage, version, baseOffset, k, n, tileN, tileK, blockCols, panels);
        }

        boolean matches(TensorStorage storage, long version, int baseOffset, int k, int n, int tileN, int tileK) {
            return this.storage == storage
                    && this.version == version
                    && this.baseOffset == baseOffset
                    && this.k == k
                    && this.n == n
                    && this.tileN == tileN
                    && this.tileK == tileK;
        }

        @Override
        public float[] panel(int kStart, int jStart) {
            return panels[(kStart / tileK) * blockCols + (jStart / tileN)];
        }
    }

    public static final class F64PackedWeights {
        private final TensorStorage storage;
        private final long version;
        private final int baseOffset;
        private final int k;
        private final int n;
        private final int tileN;
        private final int tileK;
        private final int blockCols;
        private final double[][] panels;

        private F64PackedWeights(
                TensorStorage storage,
                long version,
                int baseOffset,
                int k,
                int n,
                int tileN,
                int tileK,
                int blockCols,
                double[][] panels
        ) {
            this.storage = storage;
            this.version = version;
            this.baseOffset = baseOffset;
            this.k = k;
            this.n = n;
            this.tileN = tileN;
            this.tileK = tileK;
            this.blockCols = blockCols;
            this.panels = panels;
        }

        static F64PackedWeights build(
                TensorStorage storage,
                long version,
                double[] data,
                int baseOffset,
                int k,
                int n,
                int tileN,
                int tileK
        ) {
            int blockCols = (n + tileN - 1) / tileN;
            int blockRows = (k + tileK - 1) / tileK;
            double[][] panels = new double[blockRows * blockCols][];
            int panelIndex = 0;
            for (int kk = 0; kk < k; kk += tileK) {
                int kkEnd = Math.min(kk + tileK, k);
                for (int jj = 0; jj < n; jj += tileN) {
                    int jjEnd = Math.min(jj + tileN, n);
                    int panelWidth = jjEnd - jj;
                    double[] packed = new double[(kkEnd - kk) * panelWidth];
                    int dst = 0;
                    for (int p = kk; p < kkEnd; p++) {
                        System.arraycopy(data, baseOffset + p * n + jj, packed, dst, panelWidth);
                        dst += panelWidth;
                    }
                    panels[panelIndex++] = packed;
                }
            }
            return new F64PackedWeights(storage, version, baseOffset, k, n, tileN, tileK, blockCols, panels);
        }

        boolean matches(TensorStorage storage, long version, int baseOffset, int k, int n, int tileN, int tileK) {
            return this.storage == storage
                    && this.version == version
                    && this.baseOffset == baseOffset
                    && this.k == k
                    && this.n == n
                    && this.tileN == tileN
                    && this.tileK == tileK;
        }

        public double[] panel(int kStart, int jStart) {
            return panels[(kStart / tileK) * blockCols + (jStart / tileN)];
        }
    }

    public static final class BF16PackedWeights implements PackedFloatPanels {
        private final TensorStorage storage;
        private final long version;
        private final int baseOffset;
        private final int k;
        private final int n;
        private final int tileN;
        private final int tileK;
        private final int blockCols;
        private final float[][] panels;

        private BF16PackedWeights(
                TensorStorage storage,
                long version,
                int baseOffset,
                int k,
                int n,
                int tileN,
                int tileK,
                int blockCols,
                float[][] panels
        ) {
            this.storage = storage;
            this.version = version;
            this.baseOffset = baseOffset;
            this.k = k;
            this.n = n;
            this.tileN = tileN;
            this.tileK = tileK;
            this.blockCols = blockCols;
            this.panels = panels;
        }

        static BF16PackedWeights build(
                TensorStorage storage,
                long version,
                short[] data,
                int baseOffset,
                int k,
                int n,
                int tileN,
                int tileK
        ) {
            int blockCols = (n + tileN - 1) / tileN;
            int blockRows = (k + tileK - 1) / tileK;
            float[][] panels = new float[blockRows * blockCols][];
            int panelIndex = 0;
            for (int kk = 0; kk < k; kk += tileK) {
                int kkEnd = Math.min(kk + tileK, k);
                for (int jj = 0; jj < n; jj += tileN) {
                    int jjEnd = Math.min(jj + tileN, n);
                    int panelWidth = jjEnd - jj;
                    float[] packed = new float[(kkEnd - kk) * panelWidth];
                    int dst = 0;
                    for (int p = kk; p < kkEnd; p++) {
                        int src = baseOffset + p * n + jj;
                        for (int col = 0; col < panelWidth; col++) {
                            packed[dst++] = CpuDTypeOps.fromBFloat16Bits(data[src + col]);
                        }
                    }
                    panels[panelIndex++] = packed;
                }
            }
            return new BF16PackedWeights(storage, version, baseOffset, k, n, tileN, tileK, blockCols, panels);
        }

        boolean matches(TensorStorage storage, long version, int baseOffset, int k, int n, int tileN, int tileK) {
            return this.storage == storage
                    && this.version == version
                    && this.baseOffset == baseOffset
                    && this.k == k
                    && this.n == n
                    && this.tileN == tileN
                    && this.tileK == tileK;
        }

        @Override
        public float[] panel(int kStart, int jStart) {
            return panels[(kStart / tileK) * blockCols + (jStart / tileN)];
        }
    }
}
