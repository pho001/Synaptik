package backend.cpu.kernels.index;

import operations.index.ScatterReduction;
import tensor.DataType;
import tensor.Tensor;

final class IndexValidation {
    private IndexValidation() {
    }

    static void validateGather(Tensor input, Tensor indices, Tensor out, int dimension) {
        int[] inputShape = input.getShapeUnsafe();
        if (dimension < 0 || dimension >= inputShape.length) {
            throw new IllegalArgumentException("Gather dimension out of bounds: " + dimension);
        }
        validateIndexTensor(indices);
        int[] expectedOutShape = reduceShape(inputShape, dimension);
        validateShape(indices.getShapeUnsafe(), expectedOutShape, "Gather indices shape must equal input shape without gathered axis.");
        validateShape(out.getShapeUnsafe(), expectedOutShape, "Gather output shape must equal indices shape.");
        if (input.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("Gather output dtype must match input dtype.");
        }
    }

    static void validateGatherAxis(Tensor input, Tensor indices, Tensor out, int axis) {
        int[] inputShape = input.getShapeUnsafe();
        if (axis < 0 || axis >= inputShape.length) {
            throw new IllegalArgumentException("gatherAxis axis out of bounds: " + axis);
        }
        validateIndexTensor(indices);
        validateShape(out.getShapeUnsafe(), gatherAxisOutputShape(inputShape, indices.getShapeUnsafe(), axis),
                "gatherAxis output shape mismatch.");
        if (input.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("gatherAxis output dtype must match input dtype.");
        }
    }

    static void validateGatherAxisGrad(Tensor indices, Tensor outGrad, Tensor node, int axis) {
        int[] nodeShape = node.getShapeUnsafe();
        if (axis < 0 || axis >= nodeShape.length) {
            throw new IllegalArgumentException("gatherAxisGrad axis out of bounds: " + axis);
        }
        validateIndexTensor(indices);
        validateShape(outGrad.getShapeUnsafe(), gatherAxisOutputShape(nodeShape, indices.getShapeUnsafe(), axis),
                "gatherAxisGrad outGrad shape mismatch.");
        if (outGrad.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("gatherAxisGrad output dtype must match outGrad dtype.");
        }
        validateFloating(node.getDataType(), "gatherAxisGrad requires floating output dtype.");
    }

    static void validateScatterAxisAdd(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis) {
        validateShape(out.getShapeUnsafe(), data.getShapeUnsafe(), "scatterAxisAdd output shape must equal data shape.");
        validateGatherAxisGrad(indices, updates, out, axis);
        if (data.getDataType() != updates.getDataType() || data.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("scatterAxisAdd requires matching dtypes for data, updates and output.");
        }
        validateFloating(data.getDataType(), "scatterAxisAdd requires floating numeric tensors.");
    }

    static void validateGatherNd(Tensor input, Tensor indices, Tensor out, int batchDims) {
        validateIndexTensor(indices);
        int[] inputShape = input.getShapeUnsafe();
        int[] indicesShape = indices.getShapeUnsafe();
        validateShape(out.getShapeUnsafe(), gatherNdOutputShape(inputShape, indicesShape, batchDims),
                "gatherNd output shape mismatch.");
        if (input.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("gatherNd output dtype must match input dtype.");
        }
    }

    static void validateGatherNdGrad(Tensor indices, Tensor outGrad, Tensor node, int batchDims) {
        validateIndexTensor(indices);
        int[] nodeShape = node.getShapeUnsafe();
        int[] indicesShape = indices.getShapeUnsafe();
        validateShape(outGrad.getShapeUnsafe(), gatherNdOutputShape(nodeShape, indicesShape, batchDims),
                "gatherNdGrad outGrad shape mismatch.");
        if (outGrad.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("gatherNdGrad output dtype must match outGrad dtype.");
        }
        validateFloating(node.getDataType(), "gatherNdGrad requires floating output dtype.");
    }

    static void validateGatherGrad(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        int[] nodeShape = node.getShapeUnsafe();
        if (dimension < 0 || dimension >= nodeShape.length) {
            throw new IllegalArgumentException("GatherGrad dimension out of bounds: " + dimension);
        }
        validateIndexTensor(indices);
        int[] expectedGradShape = reduceShape(nodeShape, dimension);
        validateShape(indices.getShapeUnsafe(), expectedGradShape, "GatherGrad indices shape must equal gradient shape.");
        validateShape(outGrad.getShapeUnsafe(), expectedGradShape, "GatherGrad outGrad shape must equal indices shape.");
        if (outGrad.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("GatherGrad output dtype must match outGrad dtype.");
        }
    }

    static void validateScatterAdd(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension) {
        int[] baseShape = base.getShapeUnsafe();
        if (dimension < 0 || dimension >= baseShape.length) {
            throw new IllegalArgumentException("scatterAdd dimension out of bounds: " + dimension);
        }
        validateIndexTensor(indices);
        int[] expectedSrcShape = reduceShape(baseShape, dimension);
        validateShape(indices.getShapeUnsafe(), expectedSrcShape, "scatterAdd indices shape must equal base shape without scattered axis.");
        validateShape(src.getShapeUnsafe(), expectedSrcShape, "scatterAdd source shape must equal indices shape.");
        validateShape(out.getShapeUnsafe(), baseShape, "scatterAdd output shape must equal base shape.");
        validateFloating(base.getDataType(), "scatterAdd requires floating numeric tensors.");
        validateFloating(src.getDataType(), "scatterAdd requires floating numeric tensors.");
        validateFloating(out.getDataType(), "scatterAdd requires floating numeric tensors.");
        if (base.getDataType() != src.getDataType() || base.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("scatterAdd requires matching dtypes for base, src and output.");
        }
    }

    static ScatterReduction validateScatterElements(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            int axis,
            ScatterReduction reduction
    ) {
        ScatterReduction effectiveReduction = reduction == null ? ScatterReduction.NONE : reduction;
        int[] dataShape = data.getShapeUnsafe();
        if (axis < 0 || axis >= dataShape.length) {
            throw new IllegalArgumentException("scatterElements axis out of bounds: " + axis);
        }
        validateIndexTensor(indices);
        int[] indicesShape = indices.getShapeUnsafe();
        int[] updatesShape = updates.getShapeUnsafe();
        validateShape(out.getShapeUnsafe(), dataShape, "scatterElements output shape must equal data shape.");
        if (indicesShape.length != dataShape.length) {
            throw new IllegalArgumentException("scatterElements indices rank must match data rank.");
        }
        validateShape(updatesShape, indicesShape, "scatterElements updates shape must equal indices shape.");
        for (int i = 0; i < indicesShape.length; i++) {
            if (i != axis && indicesShape[i] != dataShape[i]) {
                throw new IllegalArgumentException("scatterElements indices must match data shape on all non-axis dimensions.");
            }
        }
        if (data.getDataType() != updates.getDataType() || data.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("scatterElements requires matching dtypes for data, updates and output.");
        }
        if (data.getDataType() == DataType.BOOL && effectiveReduction != ScatterReduction.NONE) {
            throw new IllegalArgumentException("scatterElements BOOL tensors support only NONE reduction.");
        }
        return effectiveReduction;
    }

    static ScatterReduction validateScatterNd(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            ScatterReduction reduction,
            int batchDims
    ) {
        ScatterReduction effectiveReduction = reduction == null ? ScatterReduction.NONE : reduction;
        validateIndexTensor(indices);
        int[] dataShape = data.getShapeUnsafe();
        int[] indicesShape = indices.getShapeUnsafe();
        int[] updatesShape = updates.getShapeUnsafe();
        validateShape(out.getShapeUnsafe(), dataShape, "scatterNd output shape must equal data shape.");
        validateGatherNdShape(dataShape, indicesShape, batchDims);
        int tupleRank = indicesShape[indicesShape.length - 1];
        int expectedRank = indicesShape.length - 1 + dataShape.length - batchDims - tupleRank;
        if (updatesShape.length != expectedRank) {
            if (expectedRank == 0 && updatesShape.length == 1 && updatesShape[0] == 1) {
                validateScatterNdDType(data, updates, out, effectiveReduction);
                return effectiveReduction;
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
        validateScatterNdDType(data, updates, out, effectiveReduction);
        return effectiveReduction;
    }

    static void validateTakeAlongAxis(Tensor input, Tensor indices, Tensor out, int dimension) {
        int[] inputShape = input.getShapeUnsafe();
        if (dimension < 0 || dimension >= inputShape.length) {
            throw new IllegalArgumentException("takeAlongAxis dimension out of bounds: " + dimension);
        }
        validateIndexTensor(indices);
        int[] indicesShape = indices.getShapeUnsafe();
        if (indicesShape.length != inputShape.length) {
            throw new IllegalArgumentException("takeAlongAxis indices rank must match input rank.");
        }
        for (int i = 0; i < inputShape.length; i++) {
            if (i != dimension && indicesShape[i] != inputShape[i]) {
                throw new IllegalArgumentException("takeAlongAxis indices must match input shape on all non-axis dimensions.");
            }
        }
        validateShape(out.getShapeUnsafe(), indicesShape, "takeAlongAxis output shape must equal indices shape.");
        if (input.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("takeAlongAxis output dtype must match input dtype.");
        }
    }

    static void validateTakeAlongAxisGrad(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        int[] nodeShape = node.getShapeUnsafe();
        if (dimension < 0 || dimension >= nodeShape.length) {
            throw new IllegalArgumentException("takeAlongAxisGrad dimension out of bounds: " + dimension);
        }
        validateIndexTensor(indices);
        int[] gradShape = outGrad.getShapeUnsafe();
        validateShape(indices.getShapeUnsafe(), gradShape, "takeAlongAxisGrad indices shape must equal outGrad shape.");
        if (gradShape.length != nodeShape.length) {
            throw new IllegalArgumentException("takeAlongAxisGrad outGrad rank must match input rank.");
        }
        for (int i = 0; i < nodeShape.length; i++) {
            if (i != dimension && gradShape[i] != nodeShape[i]) {
                throw new IllegalArgumentException("takeAlongAxisGrad outGrad shape must match input shape on all non-axis dimensions.");
            }
        }
        if (outGrad.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("takeAlongAxisGrad output dtype must match outGrad dtype.");
        }
    }

    static void validateShape(int[] actual, int[] expected, String message) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(message);
        }
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != expected[i]) {
                throw new IllegalArgumentException(message);
            }
        }
    }

    private static void validateIndexTensor(Tensor indices) {
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("Gather indices must be numeric integral values.");
        }
    }

    private static void validateFloating(DataType dataType, String message) {
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(message);
        }
    }

    private static void validateScatterNdDType(Tensor data, Tensor updates, Tensor out, ScatterReduction reduction) {
        if (data.getDataType() != updates.getDataType() || data.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("scatterNd requires matching dtypes for data, updates and output.");
        }
        if (data.getDataType() == DataType.BOOL && reduction != ScatterReduction.NONE) {
            throw new IllegalArgumentException("scatterNd BOOL tensors support only NONE reduction.");
        }
    }

    private static int[] reduceShape(int[] shape, int axis) {
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

    private static int[] gatherAxisOutputShape(int[] dataShape, int[] indicesShape, int axis) {
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

    private static int[] gatherNdOutputShape(int[] dataShape, int[] indicesShape, int batchDims) {
        validateGatherNdShape(dataShape, indicesShape, batchDims);
        int tupleRank = indicesShape[indicesShape.length - 1];
        int outputRank = indicesShape.length - 1 + dataShape.length - batchDims - tupleRank;
        if (outputRank == 0) {
            return new int[]{1};
        }
        int[] out = new int[outputRank];
        int p = 0;
        for (int i = 0; i < indicesShape.length - 1; i++) {
            out[p++] = indicesShape[i];
        }
        for (int i = batchDims + tupleRank; i < dataShape.length; i++) {
            out[p++] = dataShape[i];
        }
        return out;
    }

    private static void validateGatherNdShape(int[] dataShape, int[] indicesShape, int batchDims) {
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
}
