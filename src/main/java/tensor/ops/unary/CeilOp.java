package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.ceil;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code ceil}.
 */
public final class CeilOp {
    private CeilOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new ceil();
        return TensorPrimitiveBuilder.unaryNoGrad(input, input.getShape(), op, "ceil", TensorDTypes.requireFloating(input.getDataType()));
    }
}
