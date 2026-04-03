package tensor;

import operations.Operation;
import operations.equalTo;
import operations.greaterThan;
import operations.greaterOrEqual;
import operations.lessThan;
import operations.lessOrEqual;
import operations.notEqualTo;

import java.util.List;

final class TensorCompareOps {
    private TensorCompareOps() {}

    static Tensor greaterThan(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new greaterThan(plan), "gt");
    }

    static Tensor lessThan(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new lessThan(plan), "lt");
    }

    static Tensor greaterOrEqual(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new greaterOrEqual(plan), "ge");
    }

    static Tensor lessOrEqual(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new lessOrEqual(plan), "le");
    }

    static Tensor equalTo(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new equalTo(plan), "eq");
    }

    static Tensor notEqualTo(Tensor first, Tensor second) {
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
        Tensor out = new Tensor(plan.outShape(), List.of(first, second), op, label, DataType.BOOL);
        out.setRequiresGrad(false);
        return out;
    }
}
