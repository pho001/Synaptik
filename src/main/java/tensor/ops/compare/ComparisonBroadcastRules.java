package tensor.ops.compare;

import operations.Operation;
import tensor.layout.BroadcastPlan;
import tensor.DataType;
import tensor.Tensor;
import tensor.internal.TensorPrimitiveBuilder;

final class ComparisonBroadcastRules {
    private ComparisonBroadcastRules() {
    }

    static Tensor compare(Tensor first, Tensor second, BroadcastPlan plan, Operation op, String label) {
        if (first == null || second == null) {
            throw new IllegalArgumentException("compare inputs cannot be null");
        }
        if (first.getDataType() == DataType.BOOL || second.getDataType() == DataType.BOOL
                || first.getDataType() == DataType.INT32 || second.getDataType() == DataType.INT32
                || first.getDataType() == DataType.INT64 || second.getDataType() == DataType.INT64) {
            throw new IllegalArgumentException("Comparison ops require floating numeric inputs.");
        }
        return TensorPrimitiveBuilder.binaryNoGrad(first, second, plan.outShape(), op, label, DataType.BOOL);
    }
}
