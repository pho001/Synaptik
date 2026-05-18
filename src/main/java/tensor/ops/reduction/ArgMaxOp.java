package tensor.ops.reduction;

import operations.reduction.ArgMaxTiePolicy;
import operations.reduction.argMax;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorLayoutTransform;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for {@code argMax}.
 */
public final class ArgMaxOp {
    private ArgMaxOp() {
    }

    public static Tensor build(Tensor input, int dimension) {
        return build(input, dimension, false);
    }

    public static Tensor build(Tensor input, int dimension, boolean keepDims) {
        return build(input, dimension, keepDims, ArgMaxTiePolicy.FIRST_INDEX);
    }

    public static Tensor build(Tensor input, int dimension, boolean keepDims, ArgMaxTiePolicy tiePolicy) {
        if (input == null) {
            throw new IllegalArgumentException("argMax input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("argMax requires numeric input.");
        }
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        return TensorPrimitiveBuilder.unaryNoGrad(
                input,
                ReductionSupport.reduceShape(input.getShape(), normalizedDimension, keepDims),
                new argMax(normalizedDimension, keepDims, tiePolicy),
                "argmax",
                DataType.INT64
        );
    }
}
