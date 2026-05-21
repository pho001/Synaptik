package tensor.ops.bool;

import operations.Operation;
import tensor.layout.BroadcastPlan;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.internal.TensorPrimitiveBuilder;

final class BooleanBroadcastRules {
    private BooleanBroadcastRules() {
    }

    static BroadcastPlan validateBinary(Tensor first, Tensor second) {
        if (first == null || second == null) {
            throw new IllegalArgumentException("logical bool inputs cannot be null");
        }
        if (first.getDataType() != DataType.BOOL || second.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("logical bool ops require BOOL inputs.");
        }
        return TensorBroadcastOps.planBinary(first, second);
    }

    static Tensor binary(Tensor first, Tensor second, BroadcastPlan plan, Operation op, String label) {
        return TensorPrimitiveBuilder.binaryNoGrad(first, second, plan.outShape(), op, label, DataType.BOOL);
    }
}
