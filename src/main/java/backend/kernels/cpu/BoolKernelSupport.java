package backend.kernels.cpu;

import operations.Operation;

final class BoolKernelSupport {
    private BoolKernelSupport() {}

    static void runBinary(Operation.OpType type, byte[] a, byte[] b, byte[] out, ResolvedBroadcastPlan plan) {
        if (plan != null && !plan.isNoBroadcast()) {
            scalarBroadcast(type, a, b, out, plan);
            return;
        }
        for (int i = 0; i < out.length; i++) {
            out[i] = applyBinary(type, a[i], b[i]);
        }
    }

    static void runUnary(Operation.OpType type, byte[] in, byte[] out) {
        for (int i = 0; i < out.length; i++) {
            out[i] = applyUnary(type, in[i]);
        }
    }

    private static void scalarBroadcast(Operation.OpType type, byte[] a, byte[] b, byte[] out, ResolvedBroadcastPlan plan) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] aEff = plan.aEffStrides();
        int[] bEff = plan.bEffStrides();
        int[] aResets = plan.aResets();
        int[] bResets = plan.bResets();
        int rank = outStrides.length;
        int[] coords = new int[rank];
        int aIdx = 0;
        int bIdx = 0;
        for (int i = 0; i < out.length; i++) {
            out[i] = applyBinary(type, a[aIdx], b[bIdx]);
            for (int d = rank - 1; d >= 0; d--) {
                coords[d]++;
                aIdx += aEff[d];
                bIdx += bEff[d];
                if (coords[d] < outShape[d]) {
                    break;
                }
                coords[d] = 0;
                aIdx -= aResets[d];
                bIdx -= bResets[d];
            }
        }
    }

    private static byte applyBinary(Operation.OpType type, byte left, byte right) {
        boolean l = left != 0;
        boolean r = right != 0;
        boolean out = switch (type) {
            case LOGICAL_AND -> l && r;
            case LOGICAL_OR -> l || r;
            default -> throw new IllegalArgumentException("Unsupported logical binary op: " + type);
        };
        return out ? (byte) 1 : (byte) 0;
    }

    private static byte applyUnary(Operation.OpType type, byte value) {
        boolean in = value != 0;
        boolean out = switch (type) {
            case LOGICAL_NOT -> !in;
            default -> throw new IllegalArgumentException("Unsupported logical unary op: " + type);
        };
        return out ? (byte) 1 : (byte) 0;
    }
}
