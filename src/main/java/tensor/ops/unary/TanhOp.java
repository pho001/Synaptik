package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.tanh;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code tanh}.
 */
public final class TanhOp {
    private TanhOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new tanh();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "tanh", TensorDTypes.requireFloating(input.getDataType()));
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            context.accumulate(input, outGrad.mul(Tensor.onesLike(out).sub(out.mul(out))));
        });
        return out;
    }
}
