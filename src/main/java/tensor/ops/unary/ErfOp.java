package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.erf;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code erf}.
 */
public final class ErfOp {
    private ErfOp() {
    }

    public static Tensor build(Tensor input) {
        Operation op = new erf();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "erf", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor scale = Tensor.scalar(2.0d / Math.sqrt(Math.PI), input.getDataType());
            Tensor gradForInput = outGrad.mul(scale).mul(input.mul(input).neg().exp());
            UnarySupport.accumulateGradient(input, gradForInput);
        });
        return out;
    }
}
