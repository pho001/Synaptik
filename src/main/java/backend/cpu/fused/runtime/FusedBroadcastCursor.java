package backend.cpu.fused.runtime;

/**
 * Runtime cursor used by generated fused kernels for broadcasted strided inputs.
 */
public final class FusedBroadcastCursor {
    private final int[] coords;
    private final int[] outShape;
    private final int[] effStrides;
    private final int baseOffset;
    private final boolean scalarBroadcast;
    private int idx;
    private int[] gatherIdx;

    private FusedBroadcastCursor(int[] coords, int[] outShape, int[] effStrides, int baseOffset, int idx, boolean scalarBroadcast) {
        this.coords = coords;
        this.outShape = outShape;
        this.effStrides = effStrides;
        this.baseOffset = baseOffset;
        this.idx = idx;
        this.scalarBroadcast = scalarBroadcast;
    }

    public static FusedBroadcastCursor atStart(int start, int[] outShape, int[] outDenseStrides, int[] effStrides, int baseOffset) {
        int rank = outDenseStrides.length;
        int[] coords = new int[rank];
        int tmp = start;
        for (int d = 0; d < rank; d++) {
            coords[d] = tmp / outDenseStrides[d];
            tmp %= outDenseStrides[d];
        }

        int idx = baseOffset;
        for (int d = 0; d < rank; d++) {
            idx += coords[d] * effStrides[d];
        }
        return new FusedBroadcastCursor(coords, outShape, effStrides, baseOffset, idx, isScalarBroadcast(effStrides));
    }

    public int idx() {
        return idx;
    }

    public boolean isScalarBroadcast() {
        return scalarBroadcast;
    }

    public boolean staysWithinInnermostDimension(int width) {
        if (width <= 0 || coords.length == 0) {
            return false;
        }
        int last = coords.length - 1;
        return coords[last] + width <= outShape[last];
    }

    public int innermostLaneStride() {
        if (effStrides.length == 0) {
            return 0;
        }
        return effStrides[effStrides.length - 1];
    }

    public void advance(int steps) {
        for (int i = 0; i < steps; i++) {
            step();
        }
    }

    public void step() {
        if (scalarBroadcast) {
            return;
        }
        int rank = coords.length;
        for (int d = rank - 1; d >= 0; d--) {
            coords[d]++;
            idx += effStrides[d];
            if (coords[d] < outShape[d]) {
                return;
            }
            coords[d] = 0;
            idx -= outShape[d] * effStrides[d];
        }
        idx = baseOffset;
    }

    public int[] nextIndices(int width) {
        if (gatherIdx == null || gatherIdx.length < width) {
            gatherIdx = new int[width];
        }
        for (int lane = 0; lane < width; lane++) {
            gatherIdx[lane] = idx;
            step();
        }
        return gatherIdx;
    }

    private static boolean isScalarBroadcast(int[] effStrides) {
        if (effStrides == null || effStrides.length == 0) {
            return false;
        }
        for (int stride : effStrides) {
            if (stride != 0) {
                return false;
            }
        }
        return true;
    }
}
