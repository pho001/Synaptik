package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.sigmoid;
import tensor.Tensor;
import tensor.dtype.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code sigmoid}.
 */
public final class SigmoidOp {
    private SigmoidOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new sigmoid();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "sigmoid", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(out).mul(Tensor.onesLike(out).sub(out)));
        });
        return out;
    }
}
