package tensor.ops.dtype;

import operations.dtype.cast;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Explicit dtype conversion graph operations.
 */
public final class TensorDTypeOps {
    private TensorDTypeOps() {
    }

    public static Tensor cast(Tensor input, DataType targetType) {
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
        if (!isFloating(input.getDataType()) || !isFloating(targetType)) {
            out.setRequiresGrad(false);
            return out;
        }
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = cast(outGrad, input.getDataType());
            if (input.getGradient() == null) {
                TensorInternalAccess.setGradient(input, grad);
            } else {
                TensorInternalAccess.setGradient(input, input.getGradient().add(grad));
            }
        });
        return out;
    }

    private static boolean isFloating(DataType type) {
        return type == DataType.FLOAT64 || type == DataType.FLOAT32 || type == DataType.BFLOAT16;
    }
}
