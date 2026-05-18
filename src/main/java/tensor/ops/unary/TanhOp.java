package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.tanh;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code tanh}.
 */
public final class TanhOp {
    private TanhOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new tanh();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "tanh", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(Tensor.onesLike(out).sub(out.mul(out))));
        });
        return out;
    }
}
