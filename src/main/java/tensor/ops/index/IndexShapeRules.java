package tensor.ops.index;

import tensor.DataType;
import tensor.Tensor;

final class IndexShapeRules {
    private IndexShapeRules() {
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

    static void validateScatterElementsShape(int[] dataShape, int[] indicesShape, int[] updatesShape, int axis) {
        if (indicesShape.length != dataShape.length) {
            throw new IllegalArgumentException("scatterElements indices rank must match data rank.");
        }
        if (updatesShape.length != indicesShape.length) {
            throw new IllegalArgumentException("scatterElements updates rank must match indices rank.");
        }
        for (int i = 0; i < indicesShape.length; i++) {
            if (indicesShape[i] != updatesShape[i]) {
                throw new IllegalArgumentException("scatterElements updates shape must equal indices shape.");
            }
            if (i != axis && indicesShape[i] != dataShape[i]) {
                throw new IllegalArgumentException("scatterElements indices must match data shape on all non-axis dimensions.");
            }
        }
    }

    static void validateScatterNdShape(int[] dataShape, int[] indicesShape, int[] updatesShape, int batchDims) {
        validateGatherNdShape(dataShape, indicesShape, batchDims);
        int tupleRank = indicesShape[indicesShape.length - 1];
        int expectedRank = indicesShape.length - 1 + dataShape.length - batchDims - tupleRank;
        if (updatesShape.length != expectedRank) {
            if (expectedRank == 0 && updatesShape.length == 1 && updatesShape[0] == 1) {
                return;
            }
            throw new IllegalArgumentException("scatterNd updates shape must equal indices.shape[:-1] + data.shape[batchDims + indices.shape[-1]:].");
        }
        int p = 0;
        for (int i = 0; i < indicesShape.length - 1; i++) {
            if (updatesShape[p++] != indicesShape[i]) {
                throw new IllegalArgumentException("scatterNd updates prefix shape must match indices prefix shape.");
            }
        }
        for (int i = batchDims + tupleRank; i < dataShape.length; i++) {
            if (updatesShape[p++] != dataShape[i]) {
                throw new IllegalArgumentException("scatterNd updates suffix shape must match indexed data slice shape.");
            }
        }
    }

    static void validateScatterAxisAddShape(int[] dataShape, int[] indicesShape, int[] updatesShape, int axis) {
        int[] expectedUpdatesShape = gatherAxisOutputShape(dataShape, indicesShape, axis);
        if (updatesShape.length != expectedUpdatesShape.length) {
            throw new IllegalArgumentException("scatterAxisAdd updates shape must match gatherAxis output shape.");
        }
        for (int i = 0; i < updatesShape.length; i++) {
            if (updatesShape[i] != expectedUpdatesShape[i]) {
                throw new IllegalArgumentException("scatterAxisAdd updates shape must match gatherAxis output shape.");
            }
        }
    }

    static int[] gatherNdOutputShape(int[] dataShape, int[] indicesShape, int batchDims) {
        validateGatherNdShape(dataShape, indicesShape, batchDims);
        int tupleRank = indicesShape[indicesShape.length - 1];
        int outputRank = indicesShape.length - 1 + dataShape.length - batchDims - tupleRank;
        if (outputRank == 0) {
            return new int[]{1};
        }
        int[] outputShape = new int[outputRank];
        int p = 0;
        for (int i = 0; i < indicesShape.length - 1; i++) {
            outputShape[p++] = indicesShape[i];
        }
        for (int i = batchDims + tupleRank; i < dataShape.length; i++) {
            outputShape[p++] = dataShape[i];
        }
        return outputShape;
    }

    static void validateGatherNdShape(int[] dataShape, int[] indicesShape, int batchDims) {
        if (indicesShape.length == 0) {
            throw new IllegalArgumentException("gatherNd indices rank must be at least 1.");
        }
        if (batchDims < 0 || batchDims >= indicesShape.length) {
            throw new IllegalArgumentException("gatherNd batchDims must be in [0, indices rank).");
        }
        if (batchDims > dataShape.length) {
            throw new IllegalArgumentException("gatherNd batchDims cannot exceed data rank.");
        }
        for (int i = 0; i < batchDims; i++) {
            if (indicesShape[i] != dataShape[i]) {
                throw new IllegalArgumentException("gatherNd batch dimensions must match data leading dimensions.");
            }
        }
        int tupleRank = indicesShape[indicesShape.length - 1];
        if (tupleRank <= 0 || batchDims + tupleRank > dataShape.length) {
            throw new IllegalArgumentException("gatherNd final indices dimension must be in [1, data rank - batchDims].");
        }
    }

    static int[] gatherAxisOutputShape(int[] dataShape, int[] indicesShape, int axis) {
        int[] out = new int[dataShape.length + indicesShape.length - 1];
        int p = 0;
        for (int i = 0; i < axis; i++) {
            out[p++] = dataShape[i];
        }
        for (int dim : indicesShape) {
            out[p++] = dim;
        }
        for (int i = axis + 1; i < dataShape.length; i++) {
            out[p++] = dataShape[i];
        }
        return out;
    }

    static boolean isFloating(DataType type) {
        return type == DataType.FLOAT64 || type == DataType.FLOAT32 || type == DataType.BFLOAT16;
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

    private static int elementCount(int[] shape) {
        int size = 1;
        for (int dimension : shape) {
            size *= dimension;
        }
        return size;
    }
}
