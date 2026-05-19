package tensor.ops.reduction;

import operations.reduction.cumSum;
import tensor.DataType;
import tensor.Tensor;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for {@code cumSum}.
 */
public final class CumSumOp {
    private CumSumOp() {
    }

    public static Tensor build(Tensor input, int axis) {
        return build(input, axis, false, false);
    }

    public static Tensor build(Tensor input, int axis, boolean exclusive, boolean reverse) {
        if (input == null) {
            throw new IllegalArgumentException("cumSum input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("cumSum requires floating or INT32 input.");
        }
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, input.getShape().length);
        return TensorPrimitiveBuilder.unaryNoGrad(
                input,
                input.getShape(),
                new cumSum(normalizedAxis, exclusive, reverse),
                "cumsum",
                input.getDataType()
        );
    }
}
