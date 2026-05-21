package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.clampMin;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code clampMin}.
 */
public final class ClampMinOp {
    private ClampMinOp() {
    }

    public static Tensor build(Tensor input, double minValue) {
        UnaryNumericRules.requireNumeric(input, "clampMin");
        if (minValue == Double.NEGATIVE_INFINITY) {
            return input;
        }
        if (input.getOperation() instanceof clampMin inner) {
            return input.getPrevTensors().get(0).clampMin(Math.max(inner.getMinValue(), minValue));
        }
        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        Operation op = isF32 ? new clampMin((float) minValue) : new clampMin(minValue);
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "clampMin", TensorDTypes.requireFloating(input.getDataType()));
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor lower = Tensor.scalar(minValue, input.getDataType());
            Tensor gradForInput = Tensor.where(input.greaterOrEqual(lower), outGrad, Tensor.zerosLike(outGrad));
            context.accumulate(input, gradForInput);
        });
        return out;
    }
}
