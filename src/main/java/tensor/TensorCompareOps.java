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
        return compare(first, second, new greaterThan(TensorBroadcastOps.planBinary(first, second)), "gt");
    }

    static Tensor lessThan(Tensor first, Tensor second) {
        return compare(first, second, new lessThan(TensorBroadcastOps.planBinary(first, second)), "lt");
    }

    static Tensor greaterOrEqual(Tensor first, Tensor second) {
        return compare(first, second, new greaterOrEqual(TensorBroadcastOps.planBinary(first, second)), "ge");
    }

    static Tensor lessOrEqual(Tensor first, Tensor second) {
        return compare(first, second, new lessOrEqual(TensorBroadcastOps.planBinary(first, second)), "le");
    }

    static Tensor equalTo(Tensor first, Tensor second) {
        return compare(first, second, new equalTo(TensorBroadcastOps.planBinary(first, second)), "eq");
    }

    static Tensor notEqualTo(Tensor first, Tensor second) {
        return compare(first, second, new notEqualTo(TensorBroadcastOps.planBinary(first, second)), "ne");
    }

    private static Tensor compare(Tensor first, Tensor second, Operation op, String label) {
        if (first == null || second == null) {
            throw new IllegalArgumentException("compare inputs cannot be null");
        }
        if (first.getDataType() == DataType.BOOL || second.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("Comparison ops require numeric inputs.");
        }
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Tensor out = new Tensor(plan.outShape(), List.of(first, second), op, label, DataType.BOOL);
        out.setRequiresGrad(false);
        return out;
    }
}
