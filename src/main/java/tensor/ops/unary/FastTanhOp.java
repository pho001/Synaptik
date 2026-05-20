package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.fastTanh;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for approximate elementwise {@code fastTanh}.
 */
public final class FastTanhOp {
    private FastTanhOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new fastTanh();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "fastTanh", TensorDTypes.requireFloating(input.getDataType()));
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
