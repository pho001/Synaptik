package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.inv;
import tensor.Tensor;
import tensor.dtype.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise reciprocal {@code inv}.
 */
public final class InvOp {
    private InvOp() {
    }

    public static Tensor build(Tensor input) {
        if (input.getOperation() != null && input.getOperation().opType() == Operation.OpType.INV) {
            return input.getPrevTensors().get(0);
        }

        Operation op = new inv();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "inv", TensorDataTypeUtil.unary(input));

        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.neg().mul(out.mul(out)));
        });
        return out;
    }
}
