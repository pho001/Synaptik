package tensor.ops.layout;

import tensor.Tensor;

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
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }
}
