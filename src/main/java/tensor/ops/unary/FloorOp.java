package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.floor;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code floor}.
 */
public final class FloorOp {
    private FloorOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new floor();
        return TensorPrimitiveBuilder.unaryNoGrad(input, input.getShape(), op, "floor", TensorDataTypeUtil.unary(input));
    }
}
