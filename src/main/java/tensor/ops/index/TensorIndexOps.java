package tensor.ops.index;

import operations.index.gather;
import operations.index.gatherGrad;
import operations.index.scatterAdd;
import operations.layout.select;
import operations.index.takeAlongAxis;
import operations.index.takeAlongAxisGrad;
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
            Tensor grad = TensorPrimitiveBuilder.binaryNoGrad(
                    indices,
                    outGrad,
                    input.getShape().clone(),
                    new gatherGrad(normalizedDimension),
                    "gather_grad",
                    input.getDataType()
            );
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
                || base.getDataType() == DataType.INT32 || src.getDataType() == DataType.INT32) {
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
            Tensor grad = TensorPrimitiveBuilder.binaryNoGrad(
                    indices,
                    outGrad,
                    input.getShape().clone(),
                    new takeAlongAxisGrad(normalizedDimension),
                    "take_along_axis_grad",
                    input.getDataType()
            );
            IndexSupport.accumulateGradient(input, grad);
        });
        return out;
    }
}
