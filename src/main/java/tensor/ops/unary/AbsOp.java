package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.abs;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code abs}.
 */
public final class AbsOp {
    private AbsOp() {
    }

    public static Tensor build(Tensor input) {
        UnarySupport.requireNumeric(input, "abs");

        Operation op = new abs();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "abs", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
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
            UnarySupport.accumulateGradient(input, outGrad.mul(sign));
        });
        return out;
    }
}
