package tensor;

import operations.where;

import java.util.List;

final class TensorSelectOps {
    private TensorSelectOps() {}

    static Tensor where(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
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
        DataType outputType = TensorDataTypeUtil.promote(ifTrue.getDataType(), ifFalse.getDataType());
        Tensor out = new Tensor(plan.outShape(), List.of(condition, ifTrue, ifFalse), new where(), "where", outputType);
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;

            if (ifTrue.getRequiresGrad()) {
                Tensor gradRaw = where(condition, outGrad, Tensor.zerosLike(outGrad));
                Tensor grad = TensorBroadcastOps.sumToShape(gradRaw, ifTrue.getShape());
                accumulateGradient(ifTrue, grad);
            }
            if (ifFalse.getRequiresGrad()) {
                Tensor gradRaw = where(condition, Tensor.zerosLike(outGrad), outGrad);
                Tensor grad = TensorBroadcastOps.sumToShape(gradRaw, ifFalse.getShape());
                accumulateGradient(ifFalse, grad);
            }
        });
        return out;
    }

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }
}
