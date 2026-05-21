package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.abs;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code abs}.
 */
public final class AbsOp {
    private AbsOp() {
    }

    public static Tensor build(Tensor input) {
        UnaryNumericRules.requireNumeric(input, "abs");

        Operation op = new abs();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "abs", TensorDTypes.requireFloating(input.getDataType()));
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor zero = Tensor.scalar(0.0, input.getDataType());
            Tensor sign = Tensor.where(
                    input.greaterThan(zero),
                    Tensor.onesLike(input),
                    Tensor.where(
                            input.lessThan(zero),
                            Tensor.onesLike(input).mul(-1.0),
                            Tensor.zerosLike(input)
                    )
            );
            context.accumulate(input, outGrad.mul(sign));
        });
        return out;
    }
}
