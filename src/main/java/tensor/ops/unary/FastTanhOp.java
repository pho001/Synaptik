package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.fastTanh;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for approximate elementwise {@code fastTanh}.
 */
public final class FastTanhOp {
    private FastTanhOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new fastTanh();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "fastTanh", TensorDataTypeUtil.unary(input));
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
