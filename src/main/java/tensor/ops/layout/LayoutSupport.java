package tensor.ops.layout;

import tensor.Tensor;
import tensor.TensorInternalAccess;

final class LayoutSupport {
    private LayoutSupport() {
    }

    static int[] buildExpandedStrides(int[] sourceShape, int[] sourceStrides, int[] targetShape) {
        int targetRank = targetShape.length;
        int sourceRank = sourceShape.length;
        int rankOffset = targetRank - sourceRank;
        int[] outStrides = new int[targetRank];

        for (int d = 0; d < targetRank; d++) {
            int sourceDim = d - rankOffset;
            if (sourceDim < 0) {
                outStrides[d] = 0;
                continue;
            }
            outStrides[d] = sourceShape[sourceDim] == 1 && targetShape[d] != 1
                    ? 0
                    : sourceStrides[sourceDim];
        }
        return outStrides;
    }

    static int insertedAxisStride(int[] shape, int[] strides, int axis) {
        if (axis >= shape.length) {
            return 1;
        }
        return strides[axis] * shape[axis];
    }

    static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            TensorInternalAccess.setGradient(input, gradientDelta);
        } else {
            TensorInternalAccess.setGradient(input, input.getGradient().add(gradientDelta));
        }
    }

    static SliceSpec normalizeSlice(int[] inputShape, int[] starts, int[] ends, int[] axes, int[] steps) {
        if (starts == null || ends == null) {
            throw new IllegalArgumentException("slice starts and ends cannot be null.");
        }
        if (starts.length != ends.length) {
            throw new IllegalArgumentException("slice starts and ends length mismatch.");
        }
        int count = starts.length;
        int rank = inputShape.length;
        int[] normalizedAxes = axes == null || axes.length == 0 ? defaultAxes(count) : axes.clone();
        int[] normalizedSteps = steps == null || steps.length == 0 ? ones(count) : steps.clone();
        if (normalizedAxes.length != count || normalizedSteps.length != count) {
            throw new IllegalArgumentException("slice starts, ends, axes, and steps must have matching lengths.");
        }
        int[] outShape = inputShape.clone();
        int[] normalizedStarts = new int[count];
        int[] normalizedEnds = new int[count];
        boolean[] seen = new boolean[rank];
        for (int i = 0; i < count; i++) {
            int axis = tensor.layout.TensorLayoutTransform.normalizeAxis(normalizedAxes[i], rank);
            if (seen[axis]) {
                throw new IllegalArgumentException("slice axes cannot contain duplicates.");
            }
            seen[axis] = true;
            int step = normalizedSteps[i];
            if (step <= 0) {
                throw new IllegalArgumentException("slice currently supports positive steps only.");
            }
            int dim = inputShape[axis];
            int start = starts[i] < 0 ? starts[i] + dim : starts[i];
            int end = ends[i] < 0 ? ends[i] + dim : ends[i];
            start = Math.max(0, Math.min(start, dim));
            end = Math.max(0, Math.min(end, dim));
            int length = start >= end ? 0 : ((end - start + step - 1) / step);
            if (length <= 0) {
                throw new IllegalArgumentException("slice cannot produce empty dimensions.");
            }
            normalizedAxes[i] = axis;
            normalizedSteps[i] = step;
            normalizedStarts[i] = start;
            normalizedEnds[i] = end;
            outShape[axis] = length;
        }
        return new SliceSpec(normalizedStarts, normalizedEnds, normalizedAxes, normalizedSteps, outShape);
    }

    static int[] defaultAxes(int count) {
        int[] out = new int[count];
        for (int i = 0; i < count; i++) {
            out[i] = i;
        }
        return out;
    }

    static int[] allAxes(int rank) {
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            out[i] = i;
        }
        return out;
    }

    static int[] ones(int count) {
        int[] out = new int[count];
        java.util.Arrays.fill(out, 1);
        return out;
    }

    static boolean allOnes(int[] values) {
        for (int value : values) {
            if (value != 1) {
                return false;
            }
        }
        return true;
    }

    static int[] normalizePads(int[] pads, int rank, String name) {
        if (pads == null || pads.length != rank) {
            throw new IllegalArgumentException("pad " + name + " length must match input rank.");
        }
        int[] out = pads.clone();
        for (int value : out) {
            if (value < 0) {
                throw new IllegalArgumentException("pad " + name + " values must be non-negative.");
            }
        }
        return out;
    }

    record SliceSpec(int[] starts, int[] ends, int[] axes, int[] steps, int[] outputShape) {
    }
}
