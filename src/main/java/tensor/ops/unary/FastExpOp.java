package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.fastExp;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for approximate elementwise {@code fastExp}.
 */
public final class FastExpOp {
    private FastExpOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new fastExp();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "fastExp", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(out));
        });
        return out;
    }
}
