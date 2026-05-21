package tensor.ops.dtype;

import operations.dtype.cast;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for explicit dtype casts.
 */
public final class CastOp {
    private CastOp() {
    }

    public static Tensor build(Tensor input, DataType targetType) {
        if (input == null) {
            throw new IllegalArgumentException("cast input cannot be null");
        }
        if (targetType == null) {
            throw new IllegalArgumentException("cast target type cannot be null");
        }
        if (input.getDataType() == targetType) {
            return input;
        }
        Tensor out = TensorPrimitiveBuilder.unary(input, input.getShape(), new cast(targetType), "cast", targetType);
        if (!input.getDataType().isFloating() || !targetType.isFloating()) {
            out.setRequiresGrad(false);
            return out;
        }
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = build(outGrad, input.getDataType());
            context.accumulate(input, grad);
        });
        return out;
    }
}
