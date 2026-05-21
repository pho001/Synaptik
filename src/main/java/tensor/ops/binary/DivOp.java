package tensor.ops.binary;

import operations.Operation;
import operations.elementwise.binary.div;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

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
        if (BinaryScalarRules.isScalarConstant(first, 1.0d)) {
            return second.inv();
        }
        if (BinaryScalarRules.isScalarConstant(second, 1.0d)) {
            return first;
        }
        if (BinaryScalarRules.isScalarConstant(second, -1.0d)) {
            return first.neg();
        }
        if (BinaryScalarRules.isNonZeroScalarConstant(second)) {
            return first.mul(1.0d / second.scalarAsDouble());
        }

        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new div(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "/",
                TensorDTypes.promoteFloating(first.getDataType(), second.getDataType()), null);
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                context.accumulate(first, TensorBroadcastOps.sumToShape(outGrad.div(second), first.getShape()));
            }
            if (second.getRequiresGrad()) {
                Tensor grad = outGrad.neg().mul(first).div(second.pow(2));
                context.accumulate(second, TensorBroadcastOps.sumToShape(grad, second.getShape()));
            }
        });
        return out;
    }
}
