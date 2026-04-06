package backend.kernels.cpu;

import operations.Operation;

final class CompareKernelSupport {
    private CompareKernelSupport() {}

    static void runF64(Operation.OpType type, double[] a, double[] b, byte[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (plan != null && !plan.isNoBroadcast()) {
            scalarBroadcastF64(type, a, b, out, plan, 0, out.length);
            return;
        }
        scalarDirectF64(type, a, b, out, 0, out.length);
    }

    static void runF32(Operation.OpType type, float[] a, float[] b, byte[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (plan != null && !plan.isNoBroadcast()) {
            scalarBroadcastF32(type, a, b, out, plan, 0, out.length);
            return;
        }
        scalarDirectF32(type, a, b, out, 0, out.length);
    }

    static void runBF16(Operation.OpType type, short[] a, short[] b, byte[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (plan != null && !plan.isNoBroadcast()) {
            scalarBroadcastF16(type, a, b, out, plan, 0, out.length);
            return;
        }
        scalarDirectF16(type, a, b, out, 0, out.length);
    }

    private static void scalarDirectF64(Operation.OpType type, double[] a, double[] b, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = compare(type, a[i], b[i]);
        }
    }

    private static void scalarDirectF32(Operation.OpType type, float[] a, float[] b, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = compare(type, a[i], b[i]);
        }
    }

    private static void scalarDirectF16(Operation.OpType type, short[] a, short[] b, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = compare(type, CpuDTypeOps.fromBFloat16Bits(a[i]), CpuDTypeOps.fromBFloat16Bits(b[i]));
        }
    }

    private static void scalarBroadcastF64(Operation.OpType type, double[] a, double[] b, byte[] out, ResolvedBroadcastPlan plan, int start, int end) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] aEff = plan.aEffStrides();
        int[] bEff = plan.bEffStrides();
        int[] aResets = plan.aResets();
        int[] bResets = plan.bResets();
        int rank = outStrides.length;
        int[] coords = initCoords(start, outStrides, rank);
        int aIdx = 0;
        int bIdx = 0;
        for (int d = 0; d < rank; d++) {
            aIdx += coords[d] * aEff[d];
            bIdx += coords[d] * bEff[d];
        }
        for (int i = start; i < end; i++) {
            out[i] = compare(type, a[aIdx], b[bIdx]);
            advance(coords, outShape, aEff, bEff, aResets, bResets, rank, IndexPair.of(aIdx, bIdx), pair -> {
            });
            if (i + 1 < end) {
                int[] next = nextIndices(coords, outShape, aEff, bEff, aResets, bResets, rank, aIdx, bIdx);
                aIdx = next[0];
                bIdx = next[1];
            }
        }
    }

    private static void scalarBroadcastF32(Operation.OpType type, float[] a, float[] b, byte[] out, ResolvedBroadcastPlan plan, int start, int end) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] aEff = plan.aEffStrides();
        int[] bEff = plan.bEffStrides();
        int[] aResets = plan.aResets();
        int[] bResets = plan.bResets();
        int rank = outStrides.length;
        int[] coords = initCoords(start, outStrides, rank);
        int aIdx = 0;
        int bIdx = 0;
        for (int d = 0; d < rank; d++) {
            aIdx += coords[d] * aEff[d];
            bIdx += coords[d] * bEff[d];
        }
        for (int i = start; i < end; i++) {
            out[i] = compare(type, a[aIdx], b[bIdx]);
            if (i + 1 < end) {
                int[] next = nextIndices(coords, outShape, aEff, bEff, aResets, bResets, rank, aIdx, bIdx);
                aIdx = next[0];
                bIdx = next[1];
            }
        }
    }

    private static void scalarBroadcastF16(Operation.OpType type, short[] a, short[] b, byte[] out, ResolvedBroadcastPlan plan, int start, int end) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] aEff = plan.aEffStrides();
        int[] bEff = plan.bEffStrides();
        int[] aResets = plan.aResets();
        int[] bResets = plan.bResets();
        int rank = outStrides.length;
        int[] coords = initCoords(start, outStrides, rank);
        int aIdx = 0;
        int bIdx = 0;
        for (int d = 0; d < rank; d++) {
            aIdx += coords[d] * aEff[d];
            bIdx += coords[d] * bEff[d];
        }
        for (int i = start; i < end; i++) {
            out[i] = compare(type, CpuDTypeOps.fromBFloat16Bits(a[aIdx]), CpuDTypeOps.fromBFloat16Bits(b[bIdx]));
            if (i + 1 < end) {
                int[] next = nextIndices(coords, outShape, aEff, bEff, aResets, bResets, rank, aIdx, bIdx);
                aIdx = next[0];
                bIdx = next[1];
            }
        }
    }

    private static byte compare(Operation.OpType type, double left, double right) {
        boolean value = switch (type) {
            case GT -> left > right;
            case GE -> left >= right;
            case LT -> left < right;
            case LE -> left <= right;
            case EQ -> left == right;
            case NE -> left != right;
            default -> throw new IllegalArgumentException("Unsupported compare op: " + type);
        };
        return value ? (byte) 1 : (byte) 0;
    }

    private static int[] initCoords(int start, int[] outStrides, int rank) {
        int[] coords = new int[rank];
        int temp = start;
        for (int d = 0; d < rank; d++) {
            coords[d] = temp / outStrides[d];
            temp %= outStrides[d];
        }
        return coords;
    }

    private static int[] nextIndices(int[] coords, int[] outShape, int[] aEff, int[] bEff, int[] aResets, int[] bResets, int rank, int aIdx, int bIdx) {
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
            nextA -= aResets[d];
            nextB -= bResets[d];
        }
        return new int[]{nextA, nextB};
    }

    private record IndexPair(int a, int b) {
        static IndexPair of(int a, int b) { return new IndexPair(a, b); }
    }

    private static void advance(int[] coords, int[] outShape, int[] aEff, int[] bEff, int[] aResets, int[] bResets, int rank, IndexPair current, java.util.function.Consumer<IndexPair> sink) {
        // intentionally empty; retained to mirror other broadcast helpers
    }
}
