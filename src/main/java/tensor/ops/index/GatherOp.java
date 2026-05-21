package tensor.ops.index;

import operations.index.gather;
import operations.index.gatherAxis;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definitions for axis gather and take aliases.
 */
public final class GatherOp {
    private GatherOp() {
    }

    public static Tensor build(Tensor input, Tensor indices, int dimension) {
        if (input == null || indices == null) {
            throw new IllegalArgumentException("gather inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("gather indices must be numeric integral values.");
        }
        int[] inputShape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, inputShape.length);
        int[] outputShape = IndexShapeRules.reduceShape(inputShape, normalizedDimension);
        IndexShapeRules.validateGatherIndicesShape(indices.getShape(), outputShape);

        Tensor out = TensorPrimitiveBuilder.binary(
                input,
                indices,
                outputShape,
                new gather(normalizedDimension),
                "gather",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad());
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = Tensor.zerosLike(input).scatterAdd(indices, outGrad, normalizedDimension);
            context.accumulate(input, grad);
        });
        return out;
    }

    public static Tensor buildAxis(Tensor input, Tensor indices, int axis) {
        if (input == null || indices == null) {
            throw new IllegalArgumentException("gatherAxis inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("gatherAxis indices must be numeric integral values.");
        }
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, input.getShapeUnsafe().length);
        int[] outputShape = IndexShapeRules.gatherAxisOutputShape(input.getShapeUnsafe(), indices.getShapeUnsafe(), normalizedAxis);
        Tensor out = TensorPrimitiveBuilder.binary(
                input,
                indices,
                outputShape,
                new gatherAxis(normalizedAxis),
                "gatherAxis",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad() && IndexShapeRules.isFloating(input.getDataType()));
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad() || !IndexShapeRules.isFloating(input.getDataType())) {
                return;
            }
            Tensor grad = ScatterAxisAddOp.build(Tensor.zerosLike(input), indices, outGrad, normalizedAxis);
            context.accumulate(input, grad);
        });
        return out;
    }

    public static Tensor take(Tensor input, int axis, Tensor indices) {
        return buildAxis(input, indices, axis);
    }

    public static Tensor take(Tensor input, int axis, int[] indices) {
        if (indices == null || indices.length == 0) {
            throw new IllegalArgumentException("take indices cannot be null/empty.");
        }
        Tensor indexTensor = new Tensor(indices.clone(), new int[]{indices.length}, null, "take_indices", DataType.INT32);
        return take(input, axis, indexTensor);
    }
}
