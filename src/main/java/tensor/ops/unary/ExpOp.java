package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.exp;
import tensor.Tensor;
import tensor.dtype.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code exp}.
 */
public final class ExpOp {
    private ExpOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new exp();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "exp", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(out));
        });
        return out;
    }
}
