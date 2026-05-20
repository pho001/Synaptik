package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.log;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
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
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "log", TensorDTypes.requireFloating(input.getDataType()));
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            context.accumulate(input, outGrad.div(input));
        });
        return out;
    }
}
