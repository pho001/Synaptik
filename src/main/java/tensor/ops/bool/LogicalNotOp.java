package tensor.ops.bool;

import operations.elementwise.logical.logicalNot;
import tensor.DataType;
import tensor.Tensor;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for elementwise boolean {@code logical_not}.
 */
public final class LogicalNotOp {
    private LogicalNotOp() {
    }

    /**
     * Computes elementwise logical NOT.
     *
     * @param input boolean tensor; must be non-null
     * @return boolean tensor with the same shape as {@code input}
     * @throws IllegalArgumentException if {@code input} is null or not BOOL
     */
    public static Tensor build(Tensor input) {
        if (input == null) {
            throw new IllegalArgumentException("logicalNot input cannot be null");
        }
        if (input.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("logicalNot requires BOOL input.");
        }
        return TensorPrimitiveBuilder.unaryNoGrad(input, input.getShape().clone(), new logicalNot(), "logical_not", DataType.BOOL);
    }
}
