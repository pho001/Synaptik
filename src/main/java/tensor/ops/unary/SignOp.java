package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.sign;
import tensor.Tensor;
import tensor.dtype.TensorDataTypeUtil;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code sign}.
 */
public final class SignOp {
    private SignOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new sign();
        return TensorPrimitiveBuilder.unaryNoGrad(input, input.getShape(), op, "sign", TensorDataTypeUtil.unary(input));
    }
}
