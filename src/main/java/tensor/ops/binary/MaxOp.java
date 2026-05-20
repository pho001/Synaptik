package tensor.ops.binary;

import graph.compile.intent.BackendIntentPropagator;
import operations.Operation;
import operations.elementwise.binary.max;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.dtype.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code max}.
 */
public final class MaxOp {
    private MaxOp() {
    }

    /**
     * Computes the elementwise maximum of two tensors with broadcasting.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted elementwise maximum with promoted floating dtype
     */
    public static Tensor build(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new max(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "max",
                TensorDataTypeUtil.binary(first, second), null);
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                Tensor gradRaw = BinarySupport.minMaxElementwiseGrad(first, second, outGrad, true, true);
                BackendIntentPropagator.preserve(gradRaw, out);
                BinarySupport.accumulateGradient(first, TensorBroadcastOps.sumToShape(gradRaw, first.getShape()));
            }
            if (second.getRequiresGrad()) {
                Tensor gradRaw = BinarySupport.minMaxElementwiseGrad(first, second, outGrad, true, false);
                BackendIntentPropagator.preserve(gradRaw, out);
                BinarySupport.accumulateGradient(second, TensorBroadcastOps.sumToShape(gradRaw, second.getShape()));
            }
        });
        return out;
    }
}
