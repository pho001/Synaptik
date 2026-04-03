package tensor;

import operations.logicalAnd;
import operations.logicalNot;
import operations.logicalOr;

import java.util.List;

final class TensorBoolOps {
    private TensorBoolOps() {}

    static Tensor logicalAnd(Tensor first, Tensor second) {
        return binaryBool(first, second, new logicalAnd(TensorBroadcastOps.planBinary(first, second)), "logical_and");
    }

    static Tensor logicalOr(Tensor first, Tensor second) {
        return binaryBool(first, second, new logicalOr(TensorBroadcastOps.planBinary(first, second)), "logical_or");
    }

    static Tensor logicalNot(Tensor input) {
        if (input == null) {
            throw new IllegalArgumentException("logicalNot input cannot be null");
        }
        if (input.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("logicalNot requires BOOL input.");
        }
        Tensor out = new Tensor(input.getShape().clone(), List.of(input), new logicalNot(), "logical_not", DataType.BOOL);
        out.setRequiresGrad(false);
        return out;
    }

    private static Tensor binaryBool(Tensor first, Tensor second, operations.Operation op, String label) {
        if (first == null || second == null) {
            throw new IllegalArgumentException("logical bool inputs cannot be null");
        }
        if (first.getDataType() != DataType.BOOL || second.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("logical bool ops require BOOL inputs.");
        }
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Tensor out = new Tensor(plan.outShape(), List.of(first, second), op, label, DataType.BOOL);
        out.setRequiresGrad(false);
        return out;
    }
}
