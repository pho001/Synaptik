package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.mulScalar;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for scalar elementwise {@code mul}.
 */
public final class MulScalarOp {
    private MulScalarOp() {
    }

    public static Tensor build(Tensor input, double scalar) {
        UnarySupport.requireNumeric(input, "mulScalar");

        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        double scalarForGrad = isF32 ? (float) scalar : scalar;
        if (Double.compare(scalarForGrad, 0.0d) == 0) {
            return Tensor.zerosLike(input);
        }
        if (Double.compare(scalarForGrad, 1.0d) == 0) {
            return input;
        }
        if (Double.compare(scalarForGrad, -1.0d) == 0) {
            return input.neg();
        }

        Operation op = isF32 ? new mulScalar((float) scalar) : new mulScalar(scalar);
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "* constant", TensorDataTypeUtil.unary(input));

        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(scalarForGrad));
        });

        return out;
    }
}
