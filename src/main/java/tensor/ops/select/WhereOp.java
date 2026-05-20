package tensor.ops.select;

import operations.elementwise.where.where;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.layout.WhereBroadcastPlan;
import tensor.layout.WhereBroadcastPlanner;

/**
 * Graph-building definition for conditional {@code where}.
 */
public final class WhereOp {
    private WhereOp() {
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
    public static Tensor build(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        if (condition == null || ifTrue == null || ifFalse == null) {
            throw new IllegalArgumentException("where inputs cannot be null");
        }
        if (condition.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("where condition must have BOOL dtype.");
        }
        if (ifTrue.getDataType() == DataType.BOOL || ifFalse.getDataType() == DataType.BOOL
                || ifTrue.getDataType() == DataType.INT32 || ifFalse.getDataType() == DataType.INT32
                || ifTrue.getDataType() == DataType.INT64 || ifFalse.getDataType() == DataType.INT64) {
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
                TensorDTypes.promoteFloating(ifTrue.getDataType(), ifFalse.getDataType())
        );
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            if (ifTrue.getRequiresGrad()) {
                Tensor gradRaw = build(condition, outGrad, Tensor.zerosLike(outGrad));
                Tensor grad = TensorBroadcastOps.sumToShape(gradRaw, ifTrue.getShape());
                context.accumulate(ifTrue, grad);
            }
            if (ifFalse.getRequiresGrad()) {
                Tensor gradRaw = build(condition, Tensor.zerosLike(outGrad), outGrad);
                Tensor grad = TensorBroadcastOps.sumToShape(gradRaw, ifFalse.getShape());
                context.accumulate(ifFalse, grad);
            }
        });
        return out;
    }
}
