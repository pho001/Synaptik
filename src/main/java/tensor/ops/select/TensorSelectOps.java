package tensor.ops.select;

import operations.elementwise.where.where;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;
import tensor.WhereBroadcastPlan;
import tensor.WhereBroadcastPlanner;

/**
 * Conditional selection operations.
 *
 * <p>The public API currently provides a differentiable {@code where}
 * operation: gradients flow only into selected floating branches, never into
 * the boolean condition.</p>
 */
public final class TensorSelectOps {
    private TensorSelectOps() {
    }

    /**
     * Selects values from two branch tensors according to a boolean condition.
     *
     * <p>All three tensors are broadcast together. Branch tensors must be
     * floating numeric tensors and may have different floating dtypes, which are
     * promoted for the result.</p>
     *
     * @param condition BOOL tensor; true selects {@code ifTrue}, false selects {@code ifFalse}
     * @param ifTrue floating tensor used where {@code condition} is true
     * @param ifFalse floating tensor used where {@code condition} is false
     * @return broadcasted selected tensor with promoted floating dtype
     * @throws IllegalArgumentException if any input is null, condition is not BOOL,
     *                                  branches are not floating, or shapes cannot broadcast
     */
    public static Tensor where(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        if (condition == null || ifTrue == null || ifFalse == null) {
            throw new IllegalArgumentException("where inputs cannot be null");
        }
        if (condition.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("where condition must have BOOL dtype.");
        }
        if (ifTrue.getDataType() == DataType.BOOL || ifFalse.getDataType() == DataType.BOOL
                || ifTrue.getDataType() == DataType.INT32 || ifFalse.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("where branches must have floating numeric dtypes.");
        }

        WhereBroadcastPlan plan = WhereBroadcastPlanner.plan(condition, ifTrue, ifFalse);
        Tensor out = TensorPrimitiveBuilder.ternary(
                condition,
                ifTrue,
                ifFalse,
                plan.outShape(),
                new where(),
                "where",
                TensorDataTypeUtil.promote(ifTrue.getDataType(), ifFalse.getDataType())
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            if (ifTrue.getRequiresGrad()) {
                Tensor gradRaw = where(condition, outGrad, Tensor.zerosLike(outGrad));
                Tensor grad = TensorBroadcastOps.sumToShape(gradRaw, ifTrue.getShape());
                SelectSupport.accumulateGradient(ifTrue, grad);
            }
            if (ifFalse.getRequiresGrad()) {
                Tensor gradRaw = where(condition, Tensor.zerosLike(outGrad), outGrad);
                Tensor grad = TensorBroadcastOps.sumToShape(gradRaw, ifFalse.getShape());
                SelectSupport.accumulateGradient(ifFalse, grad);
            }
        });
        return out;
    }
}
