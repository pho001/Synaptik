package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.log;
import tensor.Tensor;
import tensor.dtype.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code log}.
 */
public final class LogOp {
    private LogOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new log();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "log", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.div(input));
        });
        return out;
    }
}
