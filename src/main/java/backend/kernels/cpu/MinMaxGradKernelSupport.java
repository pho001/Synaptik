package backend.kernels.cpu;

import backend.kernels.cpu.CpuDTypeOps;
import tensor.BroadcastPlan;

final class MinMaxGradKernelSupport {
    private MinMaxGradKernelSupport() {
    }

    static void runF64(double[] a, double[] b, double[] outGrad, double[] out, BroadcastPlan plan, boolean forFirstInput, boolean isMax) {
        int[] outShape = plan.outShape();
        int[] aEff = plan.aEffStrides();
        int[] bEff = plan.bEffStrides();
        int rank = outShape.length;
        int outSize = out.length;

        int[] coords = new int[rank];
        int aIdx = 0;
        int bIdx = 0;
        for (int i = 0; i < outSize; i++) {
            out[i] = gradValue(a[aIdx], b[bIdx], outGrad[i], forFirstInput, isMax);
            advance(coords, outShape, aEff, bEff, rank, i, outSize, IndexPair.of(aIdx, bIdx), pair -> {
            });
            if (i + 1 < outSize) {
                int[] next = nextIndices(coords, outShape, aEff, bEff, rank, aIdx, bIdx);
                aIdx = next[0];
                bIdx = next[1];
            }
        }
    }

    static void runF32(float[] a, float[] b, float[] outGrad, float[] out, BroadcastPlan plan, boolean forFirstInput, boolean isMax) {
        int[] outShape = plan.outShape();
        int[] aEff = plan.aEffStrides();
        int[] bEff = plan.bEffStrides();
        int rank = outShape.length;
        int outSize = out.length;

        int[] coords = new int[rank];
        int aIdx = 0;
        int bIdx = 0;
        for (int i = 0; i < outSize; i++) {
            out[i] = (float) gradValue(a[aIdx], b[bIdx], outGrad[i], forFirstInput, isMax);
            if (i + 1 < outSize) {
                int[] next = nextIndices(coords, outShape, aEff, bEff, rank, aIdx, bIdx);
                aIdx = next[0];
                bIdx = next[1];
            }
        }
    }

    static void runF16(short[] a, short[] b, short[] outGrad, short[] out, BroadcastPlan plan, boolean forFirstInput, boolean isMax) {
        int[] outShape = plan.outShape();
        int[] aEff = plan.aEffStrides();
        int[] bEff = plan.bEffStrides();
        int rank = outShape.length;
        int outSize = out.length;

        int[] coords = new int[rank];
        int aIdx = 0;
        int bIdx = 0;
        for (int i = 0; i < outSize; i++) {
            float av = CpuDTypeOps.fromHalfBits(a[aIdx]);
            float bv = CpuDTypeOps.fromHalfBits(b[bIdx]);
            float gv = CpuDTypeOps.fromHalfBits(outGrad[i]);
            out[i] = CpuDTypeOps.toHalfBits((float) gradValue(av, bv, gv, forFirstInput, isMax));
            if (i + 1 < outSize) {
                int[] next = nextIndices(coords, outShape, aEff, bEff, rank, aIdx, bIdx);
                aIdx = next[0];
                bIdx = next[1];
            }
        }
    }

    private static double gradValue(double av, double bv, double gv, boolean forFirstInput, boolean isMax) {
        if (av == bv) {
            return 0.5d * gv;
        }
        boolean firstWins = isMax ? (av > bv) : (av < bv);
        return forFirstInput == firstWins ? gv : 0.0d;
    }

    private static int[] nextIndices(int[] coords, int[] outShape, int[] aEff, int[] bEff, int rank, int aIdx, int bIdx) {
        int nextA = aIdx;
        int nextB = bIdx;
        for (int d = rank - 1; d >= 0; d--) {
            coords[d]++;
            nextA += aEff[d];
            nextB += bEff[d];
            if (coords[d] < outShape[d]) {
                break;
            }
            coords[d] = 0;
            nextA -= outShape[d] * aEff[d];
            nextB -= outShape[d] * bEff[d];
        }
        return new int[]{nextA, nextB};
    }

    private record IndexPair(int a, int b) {
        static IndexPair of(int a, int b) {
            return new IndexPair(a, b);
        }
    }

    private static void advance(int[] coords, int[] outShape, int[] aEff, int[] bEff, int rank, int i, int outSize, IndexPair current, java.util.function.Consumer<IndexPair> sink) {
        // no-op helper retained only to keep iteration logic local and explicit
    }
}
