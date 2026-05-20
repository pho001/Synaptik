package tensor.ops.binary;

import operations.Operation;
import operations.elementwise.binary.sub;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code sub}.
 */
public final class SubOp {
    private SubOp() {
    }

    /**
     * Subtracts one tensor from another elementwise with broadcasting.
     *
     * @param first minuend; must be non-null and floating numeric
     * @param second subtrahend; must be non-null and floating numeric
     * @return broadcasted difference tensor with promoted floating dtype
     */
    public static Tensor build(Tensor first, Tensor second) {
        if (BinarySupport.isScalarConstant(second, 0.0d)) {
            return first;
        }
        if (BinarySupport.isScalarConstant(first, 0.0d)) {
            return second.neg();
        }

        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new sub(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "-",
                TensorDTypes.promoteFloating(first.getDataType(), second.getDataType()), null);
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                context.accumulate(first, TensorBroadcastOps.sumToShape(outGrad, first.getShape()));
            }
            if (second.getRequiresGrad()) {
                context.accumulate(second, TensorBroadcastOps.sumToShape(outGrad.neg(), second.getShape()));
            }
        });
        return out;
    }
}
