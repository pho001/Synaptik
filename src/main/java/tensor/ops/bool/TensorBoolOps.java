package tensor.ops.bool;

import operations.elementwise.logical.logicalAnd;
import operations.elementwise.logical.logicalNot;
import operations.elementwise.logical.logicalOr;
import tensor.BroadcastPlan;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.TensorPrimitiveBuilder;

public final class TensorBoolOps {
    private TensorBoolOps() {
    }

    public static Tensor logicalAnd(Tensor first, Tensor second) {
        return binaryBool(first, second, new logicalAnd(TensorBroadcastOps.planBinary(first, second)), "logical_and");
    }

    public static Tensor logicalOr(Tensor first, Tensor second) {
        return binaryBool(first, second, new logicalOr(TensorBroadcastOps.planBinary(first, second)), "logical_or");
    }

    public static Tensor logicalNot(Tensor input) {
        if (input == null) {
            throw new IllegalArgumentException("logicalNot input cannot be null");
        }
        if (input.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("logicalNot requires BOOL input.");
        }
        return TensorPrimitiveBuilder.unaryNoGrad(input, input.getShape().clone(), new logicalNot(), "logical_not", DataType.BOOL);
    }

    private static Tensor binaryBool(Tensor first, Tensor second, operations.Operation op, String label) {
        if (first == null || second == null) {
            throw new IllegalArgumentException("logical bool inputs cannot be null");
        }
        if (first.getDataType() != DataType.BOOL || second.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("logical bool ops require BOOL inputs.");
        }
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return TensorPrimitiveBuilder.binaryNoGrad(first, second, plan.outShape(), op, label, DataType.BOOL);
    }
}
