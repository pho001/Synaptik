package tensor.ops.binary;

import operations.Operation;
import operations.elementwise.binary.min;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code min}.
 */
public final class MinOp {
    private MinOp() {
    }

    /**
     * Computes the elementwise minimum of two tensors with broadcasting.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted elementwise minimum with promoted floating dtype
     */
    public static Tensor build(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new min(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "min",
                TensorDTypes.promoteFloating(first.getDataType(), second.getDataType()), null);
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                Tensor gradRaw = BinarySupport.minMaxElementwiseGrad(first, second, outGrad, false, true);
                context.accumulate(first, TensorBroadcastOps.sumToShape(gradRaw, first.getShape()));
            }
            if (second.getRequiresGrad()) {
                Tensor gradRaw = BinarySupport.minMaxElementwiseGrad(first, second, outGrad, false, false);
                context.accumulate(second, TensorBroadcastOps.sumToShape(gradRaw, second.getShape()));
            }
        });
        return out;
    }
}
