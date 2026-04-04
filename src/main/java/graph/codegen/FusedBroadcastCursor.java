package graph.codegen;

public final class FusedBroadcastCursor {
    private final int[] coords;
    private final int[] outShape;
    private final int[] effStrides;
    private final int baseOffset;
    private int idx;
    private int[] gatherIdx;

    private FusedBroadcastCursor(int[] coords, int[] outShape, int[] effStrides, int baseOffset, int idx) {
        this.coords = coords;
        this.outShape = outShape;
        this.effStrides = effStrides;
        this.baseOffset = baseOffset;
        this.idx = idx;
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
        return new FusedBroadcastCursor(coords, outShape, effStrides, baseOffset, idx);
    }

    public int idx() {
        return idx;
    }

    public void step() {
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
}
