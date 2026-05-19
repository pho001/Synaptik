package tensor.ops.layout;

import operations.Operation;
import operations.layout.contiguous;
import tensor.Tensor;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for {@code contiguous}.
 */
public final class ContiguousOp {
    private ContiguousOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new contiguous();
        return TensorPrimitiveBuilder.unary(input, input.getShape(), op, "contiguous", input.getDataType());
    }
}
