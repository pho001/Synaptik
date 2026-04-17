package tensor.ops.compare;

import operations.Operation;
import operations.equalTo;
import operations.greaterOrEqual;
import operations.greaterThan;
import operations.lessOrEqual;
import operations.lessThan;
import operations.notEqualTo;
import tensor.BroadcastPlan;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.TensorPrimitiveBuilder;

public final class TensorCompareOps {
    private TensorCompareOps() {
    }

    public static Tensor greaterThan(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new greaterThan(plan), "gt");
    }

    public static Tensor lessThan(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new lessThan(plan), "lt");
    }

    public static Tensor greaterOrEqual(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new greaterOrEqual(plan), "ge");
    }

    public static Tensor lessOrEqual(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new lessOrEqual(plan), "le");
    }

    public static Tensor equalTo(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new equalTo(plan), "eq");
    }

    public static Tensor notEqualTo(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new notEqualTo(plan), "ne");
    }

    private static Tensor compare(Tensor first, Tensor second, BroadcastPlan plan, Operation op, String label) {
        if (first == null || second == null) {
            throw new IllegalArgumentException("compare inputs cannot be null");
        }
        if (first.getDataType() == DataType.BOOL || second.getDataType() == DataType.BOOL
                || first.getDataType() == DataType.INT32 || second.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("Comparison ops require floating numeric inputs.");
        }
        return TensorPrimitiveBuilder.binaryNoGrad(first, second, plan.outShape(), op, label, DataType.BOOL);
    }
}
