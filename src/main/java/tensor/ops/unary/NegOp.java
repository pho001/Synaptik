package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.neg;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code neg}.
 */
public final class NegOp {
    private NegOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new neg();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "neg", TensorDTypes.requireFloating(input.getDataType()));
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            context.accumulate(input, outGrad.neg());
        });
        return out;
    }
}
