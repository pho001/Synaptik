package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.neg;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code neg}.
 */
public final class NegOp {
    private NegOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new neg();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "neg", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.neg());
        });
        return out;
    }
}
