package tensor.ops.binary;

import operations.Operation;
import operations.elementwise.binary.mul;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

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
        if (BinaryScalarRules.isScalarConstant(first, 0.0d)) {
            return second.mul(0.0d);
        }
        if (BinaryScalarRules.isScalarConstant(second, 0.0d)) {
            return first.mul(0.0d);
        }
        if (BinaryScalarRules.isScalarConstant(first, 1.0d)) {
            return second;
        }
        if (BinaryScalarRules.isScalarConstant(second, 1.0d)) {
            return first;
        }
        if (BinaryScalarRules.isScalarConstant(first, -1.0d)) {
            return second.neg();
        }
        if (BinaryScalarRules.isScalarConstant(second, -1.0d)) {
            return first.neg();
        }

        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new mul(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "*",
                TensorDTypes.promoteFloating(first.getDataType(), second.getDataType()), null);
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                context.accumulate(first, TensorBroadcastOps.sumToShape(outGrad.mul(second), first.getShape()));
            }
            if (second.getRequiresGrad()) {
                context.accumulate(second, TensorBroadcastOps.sumToShape(outGrad.mul(first), second.getShape()));
            }
        });
        return out;
    }
}
