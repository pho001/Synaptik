package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.fastExp;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for approximate elementwise {@code fastExp}.
 */
public final class FastExpOp {
    private FastExpOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new fastExp();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "fastExp", TensorDTypes.requireFloating(input.getDataType()));
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
