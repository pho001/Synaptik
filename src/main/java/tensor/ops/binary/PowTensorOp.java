package tensor.ops.binary;

import operations.Operation;
import operations.elementwise.binary.powTensor;
import tensor.layout.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for tensor exponentiation {@code pow}.
 */
public final class PowTensorOp {
    private PowTensorOp() {
    }

    /**
     * Raises the left tensor to the elementwise exponent from the right tensor.
     *
     * @param first base tensor; must be floating and broadcast-compatible
     * @param second exponent tensor; must be floating and broadcast-compatible
     * @return broadcasted elementwise power tensor with promoted floating dtype
     */
    public static Tensor build(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new powTensor(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "pow",
                TensorDTypes.promoteFloating(first.getDataType(), second.getDataType()), null);
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                Tensor exponentMinusOne = second.sub(Tensor.scalar(1.0d, second.getDataType()));
                Tensor gradRaw = outGrad.mul(second).mul(first.pow(exponentMinusOne));
                context.accumulate(first, TensorBroadcastOps.sumToShape(gradRaw, first.getShape()));
            }
            if (second.getRequiresGrad()) {
                Tensor gradRaw = outGrad.mul(out).mul(first.log());
                context.accumulate(second, TensorBroadcastOps.sumToShape(gradRaw, second.getShape()));
            }
        });
        return out;
    }
}
