package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.pow;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for scalar elementwise {@code pow}.
 */
public final class PowScalarOp {
    private PowScalarOp() {
    }

    public static Tensor build(Tensor input, double exponent) {
        UnarySupport.requireNumeric(input, "pow");

        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        double exponentForGrad = isF32 ? (float) exponent : exponent;
        if (Double.compare(exponentForGrad, 0.0d) == 0) {
            return Tensor.onesLike(input);
        }
        if (Double.compare(exponentForGrad, 1.0d) == 0) {
            return input;
        }
        if (Double.compare(exponentForGrad, -1.0d) == 0) {
            return input.inv();
        }
        if (Double.compare(exponentForGrad, 2.0d) == 0) {
            return input.mul(input);
        }

        Operation op = isF32 ? new pow((float) exponent) : new pow(exponent);
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "pow", TensorDTypes.requireFloating(input.getDataType()));

        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor gradForInput = outGrad
                    .mul(exponentForGrad)
                    .mul(input.pow(exponentForGrad - 1.0));
            context.accumulate(input, gradForInput);
        });

        return out;
    }
}
