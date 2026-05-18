package backend.cpu.nativecpu.layout;

final class NativeSegmentOffsetCursor {
    private final int[] shape;
    private final int[] coords;
    private final long[][] strides;
    private final long[] offsets;

    NativeSegmentOffsetCursor(int[] shape, long[][] strides, long[] baseOffsets) {
        this.shape = shape == null ? new int[0] : shape.clone();
        this.coords = new int[this.shape.length];
        this.strides = strides == null ? new long[0][] : strides.clone();
        this.offsets = baseOffsets == null ? new long[0] : baseOffsets.clone();
    }

    long offset(int slot) {
        return offsets[slot];
    }

    void step() {
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            int nextCoord = coords[dim] + 1;
            if (nextCoord < shape[dim]) {
                coords[dim] = nextCoord;
                for (int slot = 0; slot < offsets.length; slot++) {
                    long[] slotStrides = strides[slot];
                    if (slotStrides != null) {
                        offsets[slot] += slotStrides[dim];
                    }
                }
                return;
            }

            int currentCoord = coords[dim];
            if (currentCoord != 0) {
                for (int slot = 0; slot < offsets.length; slot++) {
                    long[] slotStrides = strides[slot];
                    if (slotStrides != null) {
                        offsets[slot] -= (long) currentCoord * slotStrides[dim];
                    }
                }
            }
            coords[dim] = 0;
        }
    }
}
