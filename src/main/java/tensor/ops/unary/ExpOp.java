package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.exp;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
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
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "exp", TensorDTypes.requireFloating(input.getDataType()));
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            context.accumulate(input, outGrad.mul(out));
        });
        return out;
    }
}
