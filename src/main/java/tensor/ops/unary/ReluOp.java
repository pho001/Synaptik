package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.relu;
import tensor.Tensor;
import tensor.dtype.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code relu}.
 */
public final class ReluOp {
    private ReluOp() {
    }

    public static Tensor build(Tensor input) {
        UnarySupport.requireNumeric(input, "relu");

        Operation op = new relu();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "relu", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor zero = Tensor.scalar(0.0, input.getDataType());
            Tensor gradForInput = Tensor.where(input.greaterThan(zero), outGrad, Tensor.zerosLike(outGrad));
            UnarySupport.accumulateGradient(input, gradForInput);
        });
        return out;
    }
}
