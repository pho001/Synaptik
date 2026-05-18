package tensor.ops.binary;

import operations.Operation;
import operations.elementwise.binary.mul;
import tensor.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise {@code mul}.
 */
public final class MulOp {
    private MulOp() {
    }

    /**
     * Multiplies two tensors elementwise with broadcasting.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted product tensor with promoted floating dtype
     */
    public static Tensor build(Tensor first, Tensor second) {
        if (BinarySupport.isScalarConstant(first, 0.0d)) {
            return second.mul(0.0d);
        }
        if (BinarySupport.isScalarConstant(second, 0.0d)) {
            return first.mul(0.0d);
        }
        if (BinarySupport.isScalarConstant(first, 1.0d)) {
            return second;
        }
        if (BinarySupport.isScalarConstant(second, 1.0d)) {
            return first;
        }
        if (BinarySupport.isScalarConstant(first, -1.0d)) {
            return second.neg();
        }
        if (BinarySupport.isScalarConstant(second, -1.0d)) {
            return first.neg();
        }

        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new mul(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "*",
                TensorDataTypeUtil.binary(first, second), null);
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                BinarySupport.accumulateGradient(first, TensorBroadcastOps.sumToShape(outGrad.mul(second), first.getShape()));
            }
            if (second.getRequiresGrad()) {
                BinarySupport.accumulateGradient(second, TensorBroadcastOps.sumToShape(outGrad.mul(first), second.getShape()));
            }
        });
        return out;
    }
}
