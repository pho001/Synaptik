package tensor.ops.binary;

import operations.Operation;
import operations.elementwise.binary.add;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.dtype.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code add}.
 *
 * <p>This keeps the public operation contract next to the primitive descriptor
 * it creates, while backend execution remains owned by backend packages.</p>
 */
public final class AddOp {
    private AddOp() {
    }

    /**
     * Adds two tensors elementwise with NumPy-style broadcasting.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted sum tensor with promoted floating dtype
     */
    public static Tensor build(Tensor first, Tensor second) {
        if (BinarySupport.isScalarConstant(first, 0.0d)) {
            return second;
        }
        if (BinarySupport.isScalarConstant(second, 0.0d)) {
            return first;
        }

        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new add(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "+",
                TensorDataTypeUtil.binary(first, second), null);
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                BinarySupport.accumulateGradient(first, TensorBroadcastOps.sumToShape(outGrad, first.getShape()));
            }
            if (second.getRequiresGrad()) {
                BinarySupport.accumulateGradient(second, TensorBroadcastOps.sumToShape(outGrad, second.getShape()));
            }
        });
        return out;
    }
}
