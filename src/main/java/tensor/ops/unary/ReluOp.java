package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.relu;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code relu}.
 */
public final class ReluOp {
    private ReluOp() {
    }

    public static Tensor build(Tensor input) {
        UnaryNumericRules.requireNumeric(input, "relu");

        Operation op = new relu();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "relu", TensorDTypes.requireFloating(input.getDataType()));
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor zero = Tensor.scalar(0.0, input.getDataType());
            Tensor gradForInput = Tensor.where(input.greaterThan(zero), outGrad, Tensor.zerosLike(outGrad));
            context.accumulate(input, gradForInput);
        });
        return out;
    }
}
