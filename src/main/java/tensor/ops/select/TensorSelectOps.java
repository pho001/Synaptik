package tensor.ops.select;

import operations.elementwise.where.where;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.TensorDataTypeUtil;
import tensor.TensorPrimitiveBuilder;
import tensor.WhereBroadcastPlan;
import tensor.WhereBroadcastPlanner;

public final class TensorSelectOps {
    private TensorSelectOps() {
    }

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
        out.setBackwardFunction(() -> {
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
