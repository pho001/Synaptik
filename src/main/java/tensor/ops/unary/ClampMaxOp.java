package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.clampMax;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code clampMax}.
 */
public final class ClampMaxOp {
    private ClampMaxOp() {
    }

    public static Tensor build(Tensor input, double maxValue) {
        UnarySupport.requireNumeric(input, "clampMax");
        if (maxValue == Double.POSITIVE_INFINITY) {
            return input;
        }
        if (input.getOperation() instanceof clampMax inner) {
            return input.getPrevTensors().get(0).clampMax(Math.min(inner.getMaxValue(), maxValue));
        }
        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        Operation op = isF32 ? new clampMax((float) maxValue) : new clampMax(maxValue);
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "clampMax", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor upper = Tensor.scalar(maxValue, input.getDataType());
            Tensor gradForInput = Tensor.where(input.lessOrEqual(upper), outGrad, Tensor.zerosLike(outGrad));
            UnarySupport.accumulateGradient(input, gradForInput);
        });
        return out;
    }
}
