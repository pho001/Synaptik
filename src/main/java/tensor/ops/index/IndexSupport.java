package tensor.ops.index;

import tensor.DataType;
import tensor.Tensor;

final class IndexSupport {
    private IndexSupport() {
    }

    static void validateGatherIndicesShape(int[] indicesShape, int[] expectedShape) {
        if (indicesShape.length != expectedShape.length) {
            throw new IllegalArgumentException("gather indices shape must equal input shape without gathered axis.");
        }
        for (int i = 0; i < indicesShape.length; i++) {
            if (indicesShape[i] != expectedShape[i]) {
                throw new IllegalArgumentException("gather indices shape must equal input shape without gathered axis.");
            }
        }
    }

    static void validateTakeAlongAxisShape(int[] inputShape, int[] indicesShape, int axis) {
        if (indicesShape.length != inputShape.length) {
            throw new IllegalArgumentException("takeAlongAxis indices rank must match input rank.");
        }
        for (int i = 0; i < inputShape.length; i++) {
            if (i == axis) {
                continue;
            }
            if (indicesShape[i] != inputShape[i]) {
                throw new IllegalArgumentException("takeAlongAxis indices must match input shape on all non-axis dimensions.");
            }
        }
    }

    static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    static int[] reduceStrides(int[] strides, int axis) {
        if (strides.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[strides.length - 1];
        for (int i = 0, j = 0; i < strides.length; i++) {
            if (i != axis) {
                reduced[j++] = strides[i];
            }
        }
        return reduced;
    }

    static Tensor constantIndexTensor(int[] shape, int index) {
        int[] data = new int[elementCount(shape)];
        java.util.Arrays.fill(data, index);
        return new Tensor(data, shape, null, "select_indices", DataType.INT32);
    }

    static int normalizeIndex(int index, int axisSize) {
        int normalized = index < 0 ? index + axisSize : index;
        if (normalized < 0 || normalized >= axisSize) {
            throw new IndexOutOfBoundsException(
                    "Index out of bounds for selected axis. index=" + index + ", axisSize=" + axisSize
            );
        }
        return normalized;
    }

    static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }

    private static int elementCount(int[] shape) {
        int size = 1;
        for (int dimension : shape) {
            size *= dimension;
        }
        return size;
    }
}
