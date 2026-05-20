package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.sigmoid;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
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
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "sigmoid", TensorDTypes.requireFloating(input.getDataType()));
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            context.accumulate(input, outGrad.mul(out).mul(Tensor.onesLike(out).sub(out)));
        });
        return out;
    }
}
