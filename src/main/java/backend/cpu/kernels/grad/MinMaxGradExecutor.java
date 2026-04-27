package backend.cpu.kernels.grad;

import backend.cpu.kernels.*;

import backend.cpu.kernels.CpuDTypeOps;
import tensor.BroadcastPlan;

final class MinMaxGradExecutor {
    private MinMaxGradExecutor() {
    }

    static void runF64(double[] a, int aBaseOffset, double[] b, int bBaseOffset, double[] outGrad, int outGradBaseOffset, double[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput, boolean isMax) {
        int[] outShape = plan.outShape();
        int[] aEff = plan.aEffStrides();
        int[] bEff = plan.bEffStrides();
        int rank = outShape.length;
        int outSize = logicalSize(outShape);

        int[] coords = new int[rank];
        int aIdx = aBaseOffset;
        int bIdx = bBaseOffset;
        for (int i = 0; i < outSize; i++) {
            out[outBaseOffset + i] = gradValue(a[aIdx], b[bIdx], outGrad[outGradBaseOffset + i], forFirstInput, isMax);
            if (i + 1 < outSize) {
                int[] next = nextIndices(coords, outShape, aEff, bEff, rank, aIdx, bIdx);
                aIdx = next[0];
                bIdx = next[1];
            }
        }
    }

    static void runF32(float[] a, int aBaseOffset, float[] b, int bBaseOffset, float[] outGrad, int outGradBaseOffset, float[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput, boolean isMax) {
        int[] outShape = plan.outShape();
        int[] aEff = plan.aEffStrides();
        int[] bEff = plan.bEffStrides();
        int rank = outShape.length;
        int outSize = logicalSize(outShape);

        int[] coords = new int[rank];
        int aIdx = aBaseOffset;
        int bIdx = bBaseOffset;
        for (int i = 0; i < outSize; i++) {
            out[outBaseOffset + i] = (float) gradValue(a[aIdx], b[bIdx], outGrad[outGradBaseOffset + i], forFirstInput, isMax);
            if (i + 1 < outSize) {
                int[] next = nextIndices(coords, outShape, aEff, bEff, rank, aIdx, bIdx);
                aIdx = next[0];
                bIdx = next[1];
            }
        }
    }

    static void runBF16(short[] a, int aBaseOffset, short[] b, int bBaseOffset, short[] outGrad, int outGradBaseOffset, short[] out, int outBaseOffset, BroadcastPlan plan, boolean forFirstInput, boolean isMax) {
        int[] outShape = plan.outShape();
        int[] aEff = plan.aEffStrides();
        int[] bEff = plan.bEffStrides();
        int rank = outShape.length;
        int outSize = logicalSize(outShape);

        int[] coords = new int[rank];
        int aIdx = aBaseOffset;
        int bIdx = bBaseOffset;
        for (int i = 0; i < outSize; i++) {
            float av = CpuDTypeOps.fromBFloat16Bits(a[aIdx]);
            float bv = CpuDTypeOps.fromBFloat16Bits(b[bIdx]);
            float gv = CpuDTypeOps.fromBFloat16Bits(outGrad[outGradBaseOffset + i]);
            out[outBaseOffset + i] = CpuDTypeOps.toBFloat16Bits((float) gradValue(av, bv, gv, forFirstInput, isMax));
            if (i + 1 < outSize) {
                int[] next = nextIndices(coords, outShape, aEff, bEff, rank, aIdx, bIdx);
                aIdx = next[0];
                bIdx = next[1];
            }
        }
    }

    private static double gradValue(double av, double bv, double gv, boolean forFirstInput, boolean isMax) {
        if (av == bv) {
            return forFirstInput ? 0.0d : gv;
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

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }
}
