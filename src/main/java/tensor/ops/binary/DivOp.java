package tensor.ops.binary;

import operations.Operation;
import operations.elementwise.binary.div;
import tensor.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code div}.
 */
public final class DivOp {
    private DivOp() {
    }

    /**
     * Divides one tensor by another elementwise with broadcasting.
     *
     * @param first numerator; must be non-null and floating numeric
     * @param second denominator; must be non-null and floating numeric
     * @return broadcasted quotient tensor with promoted floating dtype
     */
    public static Tensor build(Tensor first, Tensor second) {
        if (BinarySupport.isScalarConstant(first, 1.0d)) {
            return second.inv();
        }
        if (BinarySupport.isScalarConstant(second, 1.0d)) {
            return first;
        }
        if (BinarySupport.isScalarConstant(second, -1.0d)) {
            return first.neg();
        }
        if (BinarySupport.isNonZeroScalarConstant(second)) {
            return first.mul(1.0d / second.scalarAsDouble());
        }

        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new div(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "/",
                TensorDataTypeUtil.binary(first, second), null);
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                BinarySupport.accumulateGradient(first, TensorBroadcastOps.sumToShape(outGrad.div(second), first.getShape()));
            }
            if (second.getRequiresGrad()) {
                Tensor grad = outGrad.neg().mul(first).div(second.pow(2));
                BinarySupport.accumulateGradient(second, TensorBroadcastOps.sumToShape(grad, second.getShape()));
            }
        });
        return out;
    }
}
