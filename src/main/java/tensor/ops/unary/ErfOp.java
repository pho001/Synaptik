package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.erf;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code erf}.
 */
public final class ErfOp {
    private ErfOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new erf();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "erf", TensorDTypes.requireFloating(input.getDataType()));
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor scale = Tensor.scalar(2.0d / Math.sqrt(Math.PI), input.getDataType());
            Tensor gradForInput = outGrad.mul(scale).mul(input.mul(input).neg().exp());
            context.accumulate(input, gradForInput);
        });
        return out;
    }
}
