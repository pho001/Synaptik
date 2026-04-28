package tensor.ops.bool;

import operations.elementwise.logical.logicalAnd;
import operations.elementwise.logical.logicalNot;
import operations.elementwise.logical.logicalOr;
import tensor.BroadcastPlan;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.TensorPrimitiveBuilder;

/**
 * Logical operations for {@link DataType#BOOL} tensors.
 *
 * <p>Binary logical operations broadcast their inputs and return non-gradient
 * boolean tensors. Inputs are validated as boolean tensors and are not mutated.</p>
 */
public final class TensorBoolOps {
    private TensorBoolOps() {
    }

    /**
     * Computes elementwise logical AND with broadcasting.
     *
     * @param first left boolean tensor; must be non-null
     * @param second right boolean tensor; must be non-null
     * @return broadcasted boolean result
     * @throws NullPointerException if either input is null
     * @throws IllegalArgumentException if an input is non-boolean or not broadcast-compatible
     */
    public static Tensor logicalAnd(Tensor first, Tensor second) {
        return binaryBool(first, second, new logicalAnd(TensorBroadcastOps.planBinary(first, second)), "logical_and");
    }

    /**
     * Computes elementwise logical OR with broadcasting.
     *
     * @param first left boolean tensor; must be non-null
     * @param second right boolean tensor; must be non-null
     * @return broadcasted boolean result
     * @throws NullPointerException if either input is null
     * @throws IllegalArgumentException if an input is non-boolean or not broadcast-compatible
     */
    public static Tensor logicalOr(Tensor first, Tensor second) {
        return binaryBool(first, second, new logicalOr(TensorBroadcastOps.planBinary(first, second)), "logical_or");
    }

    /**
     * Computes elementwise logical NOT.
     *
     * @param input boolean tensor; must be non-null
     * @return boolean tensor with the same shape as {@code input}
     * @throws IllegalArgumentException if {@code input} is null or not BOOL
     */
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
