package tensor.layout;

import java.util.Arrays;

public final class TensorLayoutTransform {
    private TensorLayoutTransform() {}

    public static int[] inferReshape(int[] oldShape, int[] requestedShape) {
        if (requestedShape == null || requestedShape.length == 0) {
            throw new IllegalArgumentException("Requested shape cannot be null/empty.");
        }
        int oldSize = size(oldShape);
        int[] out = requestedShape.clone();
        int minusOneIndex = -1;
        long knownProduct = 1L;
        for (int i = 0; i < out.length; i++) {
            int dim = out[i];
            if (dim == -1) {
                if (minusOneIndex != -1) {
                    throw new IllegalArgumentException("Only one -1 is allowed in reshape shape.");
                }
                minusOneIndex = i;
                continue;
            }
            if (dim <= 0) {
                throw new IllegalArgumentException("Reshape dimensions must be positive or -1.");
            }
            knownProduct *= dim;
        }
        if (minusOneIndex != -1) {
            if (knownProduct == 0 || oldSize % knownProduct != 0) {
                throw new IllegalArgumentException("Cannot infer reshape dimension for size=" + oldSize
                        + " and shape=" + Arrays.toString(requestedShape));
            }
            out[minusOneIndex] = (int) (oldSize / knownProduct);
        }
        if (size(out) != oldSize) {
            throw new IllegalArgumentException("Reshape size mismatch. oldSize=" + oldSize
                    + ", newShape=" + Arrays.toString(out));
        }
        return out;
    }

    public static int[] inferExpandShape(int[] oldShape, int[] requestedShape) {
        if (requestedShape == null || requestedShape.length == 0) {
            throw new IllegalArgumentException("Requested expand shape cannot be null/empty.");
        }
        int[] out = requestedShape.clone();
        for (int dim : out) {
            if (dim <= 0) {
                throw new IllegalArgumentException("Expand dimensions must be positive.");
            }
        }

        int oldRank = oldShape.length;
        int newRank = out.length;
        if (newRank < oldRank) {
            throw new IllegalArgumentException("Expanded rank cannot be smaller than source rank.");
        }

        int offset = newRank - oldRank;
        for (int d = 0; d < newRank; d++) {
            int srcDimIndex = d - offset;
            if (srcDimIndex < 0) {
                continue;
            }
            int srcDim = oldShape[srcDimIndex];
            int dstDim = out[d];
            if (srcDim != dstDim && srcDim != 1) {
                throw new IllegalArgumentException(
                        "Cannot expand non-singleton dimension " + srcDim + " to " + dstDim
                );
            }
        }
        return out;
    }

    public static int[] normalizeAxes(int rank, int[] axes) {
        if (axes == null || axes.length != rank) {
            throw new IllegalArgumentException("Axes length must equal tensor rank.");
        }
        boolean[] seen = new boolean[rank];
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int axis = axes[i];
            if (axis < 0) axis += rank;
            if (axis < 0 || axis >= rank) {
                throw new IllegalArgumentException("Axis out of range for rank " + rank + ": " + axes[i]);
            }
            if (seen[axis]) {
                throw new IllegalArgumentException("Duplicate axis in permutation: " + Arrays.toString(axes));
            }
            seen[axis] = true;
            out[i] = axis;
        }
        return out;
    }

    public static int[] inverseAxes(int[] axes) {
        int[] inv = new int[axes.length];
        for (int i = 0; i < axes.length; i++) {
            inv[axes[i]] = i;
        }
        return inv;
    }

    public static int normalizeInsertAxis(int axis, int rank) {
        int out = axis;
        if (out < 0) out += (rank + 1);
        if (out < 0 || out > rank) {
            throw new IllegalArgumentException("Axis out of range for expandDims: axis=" + axis + ", rank=" + rank);
        }
        return out;
    }

    public static int normalizeAxis(int axis, int rank) {
        int out = axis;
        if (out < 0) out += rank;
        if (out < 0 || out >= rank) {
            throw new IllegalArgumentException("Axis out of range: axis=" + axis + ", rank=" + rank);
        }
        return out;
    }

    private static int size(int[] shape) {
        return TensorShape.checkedFlatSize(shape);
    }
}
