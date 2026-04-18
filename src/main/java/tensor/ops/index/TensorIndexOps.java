package tensor.ops.index;

import operations.index.gather;
import operations.index.gatherGrad;
import operations.index.scatterAdd;
import operations.layout.select;
import operations.index.takeAlongAxis;
import operations.index.takeAlongAxisGrad;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorLayoutTransform;
import tensor.TensorPrimitiveBuilder;

import java.util.List;

public final class TensorIndexOps {
    private TensorIndexOps() {
    }

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
        out.setBackwardFunction(() -> {
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
        out.setBackwardFunction(() -> {
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
        out.setBackwardFunction(() -> {
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
        out.setBackwardFunction(() -> {
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
