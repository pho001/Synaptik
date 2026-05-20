package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.sqrt;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code sqrt}.
 */
public final class SqrtOp {
    private SqrtOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new sqrt();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "sqrt", TensorDTypes.requireFloating(input.getDataType()));

        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            context.accumulate(input, outGrad.mul(0.5).mul(out.inv()));
        });
        return out;
    }
}
