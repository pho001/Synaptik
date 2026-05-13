package tensor.ops.index;

import operations.index.gather;
import operations.index.gatherAxis;
import operations.index.gatherNd;
import operations.index.ScatterReduction;
import operations.index.scatterAdd;
import operations.index.scatterAxisAdd;
import operations.index.scatterElements;
import operations.index.scatterNd;
import operations.layout.select;
import operations.index.takeAlongAxis;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorLayoutTransform;
import tensor.TensorPrimitiveBuilder;

import java.util.List;

/**
 * Tensor indexing and indexed update operations.
 *
 * <p>Index tensors must contain numeric integral values and may be stored as a
 * floating or {@link DataType#INT32} tensor, depending on how they were created.
 * Boolean indices are rejected. Operations build graph tensors and do not mutate
 * source tensors.</p>
 */
public final class TensorIndexOps {
    private TensorIndexOps() {
    }

    /**
     * Returns a view selecting one index from one dimension.
     *
     * @param input source tensor; must be non-null
     * @param dimension axis to select from; negative axes are normalized
     * @param index element index on {@code dimension}; negative indices are normalized
     * @return view with the selected axis removed
     * @throws IllegalArgumentException if {@code input} is null or the dimension/index is invalid
     */
    public static Tensor select(Tensor input, int dimension, int index) {
        if (input == null) {
            throw new IllegalArgumentException("select input cannot be null");
        }
        int[] inputShape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, inputShape.length);
        int normalizedIndex = IndexSupport.normalizeIndex(index, inputShape[normalizedDimension]);
        int[] outShape = IndexSupport.reduceShape(inputShape, normalizedDimension);
        int[] outStrides = IndexSupport.reduceStrides(input.getStridesUnsafe(), normalizedDimension);
        int outStorageOffset = input.getStorageOffsetUnsafe() + normalizedIndex * input.getStridesUnsafe()[normalizedDimension];

        Tensor out = TensorPrimitiveBuilder.unaryView(
                input,
                outShape,
                outStrides,
                outStorageOffset,
                new select(normalizedDimension, normalizedIndex),
                "select",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor zeroBase = Tensor.zerosLike(input);
            Tensor indices = IndexSupport.constantIndexTensor(IndexSupport.reduceShape(input.getShapeUnsafe(), normalizedDimension), normalizedIndex);
            Tensor grad = zeroBase.scatterAdd(indices, outGrad, normalizedDimension);
            IndexSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    /**
     * Gathers one element per index from a dimension, removing that dimension.
     *
     * <p>For an input of shape {@code [2, 3]} gathered along dimension {@code 1},
     * indices of shape {@code [2]} produce shape {@code [2]}.</p>
     *
     * @param input source tensor; must be non-null
     * @param indices numeric integral index tensor with shape equal to input shape
     *                without {@code dimension}
     * @param dimension input axis to gather from; negative axes are normalized
     * @return gathered tensor with dtype matching {@code input}
     * @throws IllegalArgumentException if inputs are null, indices are BOOL, shape
     *                                  validation fails, or dimension is invalid
     */
    public static Tensor gather(Tensor input, Tensor indices, int dimension) {
        if (input == null || indices == null) {
            throw new IllegalArgumentException("gather inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("gather indices must be numeric integral values.");
        }
        int[] inputShape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, inputShape.length);
        int[] outputShape = IndexSupport.reduceShape(inputShape, normalizedDimension);
        IndexSupport.validateGatherIndicesShape(indices.getShape(), outputShape);

        Tensor out = TensorPrimitiveBuilder.binary(
                input,
                indices,
                outputShape,
                new gather(normalizedDimension),
                "gather",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = Tensor.zerosLike(input).scatterAdd(indices, outGrad, normalizedDimension);
            IndexSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    /**
     * ONNX-style gather that inserts the index tensor shape at the gathered axis.
     *
     * <p>For data shape {@code [A, B, C]}, indices shape {@code [I, J]}, and
     * axis {@code 1}, the output shape is {@code [A, I, J, C]}.</p>
     */
    public static Tensor gatherAxis(Tensor input, Tensor indices, int axis) {
        if (input == null || indices == null) {
            throw new IllegalArgumentException("gatherAxis inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("gatherAxis indices must be numeric integral values.");
        }
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, input.getShapeUnsafe().length);
        int[] outputShape = gatherAxisOutputShape(input.getShapeUnsafe(), indices.getShapeUnsafe(), normalizedAxis);
        Tensor out = TensorPrimitiveBuilder.binary(
                input,
                indices,
                outputShape,
                new gatherAxis(normalizedAxis),
                "gatherAxis",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad() && isFloating(input.getDataType()));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad() || !isFloating(input.getDataType())) {
                return;
            }
            Tensor grad = scatterAxisAdd(Tensor.zerosLike(input), indices, outGrad, normalizedAxis);
            IndexSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    /**
     * ONNX GatherND with batch_dims=0: tuple-index read with slice-preserving suffix.
     */
    public static Tensor gatherNd(Tensor input, Tensor indices) {
        return gatherNd(input, indices, 0);
    }

    /**
     * ONNX GatherND: tuple-index read with optional leading batch dimensions.
     */
    public static Tensor gatherNd(Tensor input, Tensor indices, int batchDims) {
        if (input == null || indices == null) {
            throw new IllegalArgumentException("gatherNd inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("gatherNd indices must be numeric integral values.");
        }
        int[] outputShape = gatherNdOutputShape(input.getShapeUnsafe(), indices.getShapeUnsafe(), batchDims);
        Tensor out = TensorPrimitiveBuilder.binary(
                input,
                indices,
                outputShape,
                new gatherNd(batchDims),
                "gatherNd",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad() && isFloating(input.getDataType()));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad() || !isFloating(input.getDataType())) {
                return;
            }
            Tensor grad = scatterNd(Tensor.zerosLike(input), indices, outGrad, ScatterReduction.ADD, batchDims);
            IndexSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    /**
     * Adds source values into a copy-shaped output at indexed positions.
     *
     * <p>{@code base} and {@code src} must be floating tensors of the same dtype.
     * The returned tensor has {@code base}'s shape; {@code base} itself is not
     * modified.</p>
     *
     * @param base floating base tensor; must be non-null
     * @param indices numeric integral index tensor with shape equal to base shape
     *                without {@code dimension}
     * @param src floating source tensor with the same shape as {@code indices}
     * @param dimension target axis in {@code base}; negative axes are normalized
     * @return tensor containing base values plus scattered additions
     * @throws IllegalArgumentException if inputs are null, dtypes are invalid or
     *                                  mismatched, shapes are invalid, or axis is invalid
     */
    public static Tensor scatterAdd(Tensor base, Tensor indices, Tensor src, int dimension) {
        if (base == null || indices == null || src == null) {
            throw new IllegalArgumentException("scatterAdd inputs cannot be null");
        }
        if (base.getDataType() == DataType.BOOL || src.getDataType() == DataType.BOOL
                || base.getDataType() == DataType.INT32 || src.getDataType() == DataType.INT32
                || base.getDataType() == DataType.INT64 || src.getDataType() == DataType.INT64) {
            throw new IllegalArgumentException("scatterAdd requires floating numeric base and source tensors.");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("scatterAdd indices must be numeric integral values.");
        }
        if (base.getDataType() != src.getDataType()) {
            throw new IllegalArgumentException("scatterAdd requires base and source tensors to have matching dtypes.");
        }
        int[] baseShape = base.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, baseShape.length);
        int[] expectedSrcShape = IndexSupport.reduceShape(baseShape, normalizedDimension);
        IndexSupport.validateGatherIndicesShape(indices.getShape(), expectedSrcShape);
        IndexSupport.validateGatherIndicesShape(src.getShape(), expectedSrcShape);

        Tensor out = TensorPrimitiveBuilder.ternary(
                base,
                indices,
                src,
                base.getShape().clone(),
                new scatterAdd(normalizedDimension),
                "scatterAdd",
                base.getDataType()
        );
        out.setRequiresGrad(base.getRequiresGrad() || src.getRequiresGrad());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (base.getRequiresGrad()) {
                IndexSupport.accumulateGradient(base, outGrad);
            }
            if (src.getRequiresGrad()) {
                IndexSupport.accumulateGradient(src, outGrad.gather(indices, normalizedDimension));
            }
        });
        return out;
    }

    /**
     * Writes update values into a copy of {@code data} using rank-preserving axis indices.
     */
    public static Tensor scatterElements(Tensor data, Tensor indices, Tensor updates, int axis, ScatterReduction reduction) {
        if (data == null || indices == null || updates == null) {
            throw new IllegalArgumentException("scatterElements inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("scatterElements indices must be numeric integral values.");
        }
        if (data.getDataType() != updates.getDataType()) {
            throw new IllegalArgumentException("scatterElements requires data and updates to have matching dtypes.");
        }
        ScatterReduction effectiveReduction = reduction == null ? ScatterReduction.NONE : reduction;
        if (data.getDataType() == DataType.BOOL && effectiveReduction != ScatterReduction.NONE) {
            throw new IllegalArgumentException("scatterElements BOOL tensors support only NONE reduction.");
        }
        int[] dataShape = data.getShape();
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, dataShape.length);
        validateScatterElementsShape(dataShape, indices.getShapeUnsafe(), updates.getShapeUnsafe(), normalizedAxis);
        boolean differentiable = isFloating(data.getDataType())
                && (data.getRequiresGrad() || updates.getRequiresGrad());
        if (differentiable && effectiveReduction != ScatterReduction.NONE && effectiveReduction != ScatterReduction.ADD) {
            throw new UnsupportedOperationException("scatterElements backward supports only NONE and ADD reductions.");
        }

        Tensor out = TensorPrimitiveBuilder.ternary(
                data,
                indices,
                updates,
                dataShape.clone(),
                new scatterElements(normalizedAxis, effectiveReduction),
                "scatterElements",
                data.getDataType()
        );
        out.setRequiresGrad(differentiable);
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !isFloating(data.getDataType())) {
                return;
            }
            if (data.getRequiresGrad()) {
                Tensor dataGrad = switch (effectiveReduction) {
                    case NONE -> outGrad.scatterElements(indices, Tensor.zerosLike(updates), normalizedAxis, ScatterReduction.NONE);
                    case ADD -> outGrad;
                    case MUL, MAX, MIN -> throw new UnsupportedOperationException("scatterElements backward supports only NONE and ADD reductions.");
                };
                IndexSupport.accumulateGradient(data, dataGrad);
            }
            if (updates.getRequiresGrad()) {
                Tensor updatesGrad = outGrad.takeAlongAxis(indices, normalizedAxis);
                IndexSupport.accumulateGradient(updates, updatesGrad);
            }
        });
        return out;
    }

    /**
     * Writes update values into a copy of {@code data} using tuple indices.
     */
    public static Tensor scatterNd(Tensor data, Tensor indices, Tensor updates, ScatterReduction reduction) {
        return scatterNd(data, indices, updates, reduction, 0);
    }

    /**
     * Writes update values into a copy of {@code data} using tuple indices and
     * optional leading batch dimensions.
     */
    public static Tensor scatterNd(Tensor data, Tensor indices, Tensor updates, ScatterReduction reduction, int batchDims) {
        if (data == null || indices == null || updates == null) {
            throw new IllegalArgumentException("scatterNd inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("scatterNd indices must be numeric integral values.");
        }
        if (data.getDataType() != updates.getDataType()) {
            throw new IllegalArgumentException("scatterNd requires data and updates to have matching dtypes.");
        }
        ScatterReduction effectiveReduction = reduction == null ? ScatterReduction.NONE : reduction;
        if (data.getDataType() == DataType.BOOL && effectiveReduction != ScatterReduction.NONE) {
            throw new IllegalArgumentException("scatterNd BOOL tensors support only NONE reduction.");
        }
        validateScatterNdShape(data.getShapeUnsafe(), indices.getShapeUnsafe(), updates.getShapeUnsafe(), batchDims);
        boolean differentiable = isFloating(data.getDataType())
                && (data.getRequiresGrad() || updates.getRequiresGrad());
        if (differentiable && effectiveReduction != ScatterReduction.NONE && effectiveReduction != ScatterReduction.ADD) {
            throw new UnsupportedOperationException("scatterNd backward supports only NONE and ADD reductions.");
        }

        Tensor out = TensorPrimitiveBuilder.ternary(
                data,
                indices,
                updates,
                data.getShape().clone(),
                new scatterNd(effectiveReduction, batchDims),
                "scatterNd",
                data.getDataType()
        );
        out.setRequiresGrad(differentiable);
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !isFloating(data.getDataType())) {
                return;
            }
            if (data.getRequiresGrad()) {
                Tensor dataGrad = switch (effectiveReduction) {
                    case NONE -> scatterNd(outGrad, indices, Tensor.zerosLike(updates), ScatterReduction.NONE, batchDims);
                    case ADD -> outGrad;
                    case MUL, MAX, MIN -> throw new UnsupportedOperationException("scatterNd backward supports only NONE and ADD reductions.");
                };
                IndexSupport.accumulateGradient(data, dataGrad);
            }
            if (updates.getRequiresGrad()) {
                Tensor updatesGrad = outGrad.gatherNd(indices, batchDims);
                IndexSupport.accumulateGradient(updates, updatesGrad);
            }
        });
        return out;
    }

    /**
     * Adds rank-changing axis updates into a copy of {@code data}.
     */
    public static Tensor scatterAxisAdd(Tensor data, Tensor indices, Tensor updates, int axis) {
        if (data == null || indices == null || updates == null) {
            throw new IllegalArgumentException("scatterAxisAdd inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("scatterAxisAdd indices must be numeric integral values.");
        }
        if (data.getDataType() != updates.getDataType()) {
            throw new IllegalArgumentException("scatterAxisAdd requires data and updates to have matching dtypes.");
        }
        if (!isFloating(data.getDataType())) {
            throw new IllegalArgumentException("scatterAxisAdd requires floating numeric data and updates.");
        }
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, data.getShapeUnsafe().length);
        validateScatterAxisAddShape(data.getShapeUnsafe(), indices.getShapeUnsafe(), updates.getShapeUnsafe(), normalizedAxis);
        Tensor out = TensorPrimitiveBuilder.ternary(
                data,
                indices,
                updates,
                data.getShape().clone(),
                new scatterAxisAdd(normalizedAxis),
                "scatterAxisAdd",
                data.getDataType()
        );
        out.setRequiresGrad(data.getRequiresGrad() || updates.getRequiresGrad());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !isFloating(data.getDataType())) {
                return;
            }
            if (data.getRequiresGrad()) {
                IndexSupport.accumulateGradient(data, outGrad);
            }
            if (updates.getRequiresGrad()) {
                IndexSupport.accumulateGradient(updates, outGrad.gatherAxis(indices, normalizedAxis));
            }
        });
        return out;
    }

    private static void validateScatterElementsShape(int[] dataShape, int[] indicesShape, int[] updatesShape, int axis) {
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

    private static void validateScatterNdShape(int[] dataShape, int[] indicesShape, int[] updatesShape, int batchDims) {
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

    private static void validateScatterAxisAddShape(int[] dataShape, int[] indicesShape, int[] updatesShape, int axis) {
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

    private static int[] gatherNdOutputShape(int[] dataShape, int[] indicesShape, int batchDims) {
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

    private static boolean isFloating(DataType type) {
        return type == DataType.FLOAT64 || type == DataType.FLOAT32 || type == DataType.BFLOAT16;
    }

    /**
     * Gathers values using an index tensor with the desired output shape.
     *
     * <p>The index tensor shape must match the input shape except at
     * {@code dimension}; the result shape equals {@code indices.getShape()}.</p>
     *
     * @param input source tensor; must be non-null
     * @param indices numeric integral index tensor; must be non-null and non-BOOL
     * @param dimension axis to gather from; negative axes are normalized
     * @return tensor with the same shape as {@code indices} and dtype matching {@code input}
     * @throws IllegalArgumentException if inputs are null, indices are BOOL, shape
     *                                  validation fails, or dimension is invalid
     */
    public static Tensor takeAlongAxis(Tensor input, Tensor indices, int dimension) {
        if (input == null || indices == null) {
            throw new IllegalArgumentException("takeAlongAxis inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("takeAlongAxis indices must be numeric integral values.");
        }
        int[] inputShape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, inputShape.length);
        IndexSupport.validateTakeAlongAxisShape(inputShape, indices.getShape(), normalizedDimension);

        Tensor out = TensorPrimitiveBuilder.binary(
                input,
                indices,
                indices.getShape().clone(),
                new takeAlongAxis(normalizedDimension),
                "takeAlongAxis",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = Tensor.zerosLike(input).scatterElements(indices, outGrad, normalizedDimension, ScatterReduction.ADD);
            IndexSupport.accumulateGradient(input, grad);
        });
        return out;
    }
}
