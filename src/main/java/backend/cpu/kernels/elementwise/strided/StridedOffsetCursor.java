package backend.cpu.kernels.elementwise.strided;

final class StridedOffsetCursor {
    private final int[] shape;
    private final int[] coords;
    private final int[][] strides;
    private final int[] offsets;

    StridedOffsetCursor(int[] shape, int[][] strides, int[] baseOffsets) {
        this.shape = shape != null ? shape : new int[0];
        this.coords = new int[this.shape.length];
        this.strides = strides;
        this.offsets = baseOffsets.clone();
    }

    int offset(int slot) {
        return offsets[slot];
    }

    void step() {
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            int nextCoord = coords[dim] + 1;
            if (nextCoord < shape[dim]) {
                coords[dim] = nextCoord;
                for (int slot = 0; slot < offsets.length; slot++) {
                    int[] slotStrides = strides[slot];
                    if (slotStrides != null) {
                        offsets[slot] += slotStrides[dim];
                    }
                }
                return;
            }

            int currentCoord = coords[dim];
            if (currentCoord != 0) {
                for (int slot = 0; slot < offsets.length; slot++) {
                    int[] slotStrides = strides[slot];
                    if (slotStrides != null) {
                        offsets[slot] -= currentCoord * slotStrides[dim];
                    }
                }
            }
            coords[dim] = 0;
        }
    }
}
