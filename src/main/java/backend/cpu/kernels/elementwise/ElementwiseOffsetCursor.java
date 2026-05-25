package backend.cpu.kernels.elementwise;

import tensor.TensorMetadata;

public final class ElementwiseOffsetCursor {
    private final int[] shape;
    private final int[] coords;
    private final int[][] strides;
    private final int[] offsets;

    public ElementwiseOffsetCursor(int[] shape, int[][] strides, int[] baseOffsets) {
        this(shape, strides, baseOffsets, 0);
    }

    public ElementwiseOffsetCursor(int[] shape, int[][] strides, int[] baseOffsets, int start) {
        this.shape = shape == null ? new int[0] : shape.clone();
        this.coords = coordinates(this.shape, Math.max(0, start));
        this.strides = strides;
        this.offsets = initialOffsets(strides, baseOffsets, coords);
    }

    public int offset(int slot) {
        return offsets[slot];
    }

    public void step() {
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

    private static int[] coordinates(int[] shape, int start) {
        if (shape.length == 0) {
            return new int[0];
        }
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        int[] out = new int[shape.length];
        int remaining = start;
        for (int dim = 0; dim < shape.length; dim++) {
            out[dim] = denseStrides[dim] == 0 ? 0 : remaining / denseStrides[dim];
            remaining = denseStrides[dim] == 0 ? 0 : remaining % denseStrides[dim];
        }
        return out;
    }

    private static int[] initialOffsets(int[][] strides, int[] baseOffsets, int[] coords) {
        int[] out = baseOffsets.clone();
        for (int slot = 0; slot < out.length; slot++) {
            int[] slotStrides = strides[slot];
            if (slotStrides == null) {
                continue;
            }
            for (int dim = 0; dim < coords.length; dim++) {
                out[slot] += coords[dim] * slotStrides[dim];
            }
        }
        return out;
    }
}
