package Graph.codegen;

public final class FusedBroadcastCursor {
    private final int[] coords;
    private final int[] outShape;
    private final int[] effStrides;
    private int idx;
    private int[] gatherIdx;

    private FusedBroadcastCursor(int[] coords, int[] outShape, int[] effStrides, int idx) {
        this.coords = coords;
        this.outShape = outShape;
        this.effStrides = effStrides;
        this.idx = idx;
    }

    public static FusedBroadcastCursor atStart(int start, int[] outShape, int[] outStrides, int[] effStrides) {
        int rank = outStrides.length;
        int[] coords = new int[rank];
        int tmp = start;
        for (int d = 0; d < rank; d++) {
            coords[d] = tmp / outStrides[d];
            tmp %= outStrides[d];
        }

        int idx = 0;
        for (int d = 0; d < rank; d++) {
            idx += coords[d] * effStrides[d];
        }
        return new FusedBroadcastCursor(coords, outShape, effStrides, idx);
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
