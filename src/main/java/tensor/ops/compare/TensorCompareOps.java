package tensor.ops.compare;

import operations.Operation;
import operations.elementwise.compare.equalTo;
import operations.elementwise.compare.greaterOrEqual;
import operations.elementwise.compare.greaterThan;
import operations.elementwise.compare.lessOrEqual;
import operations.elementwise.compare.lessThan;
import operations.elementwise.compare.notEqualTo;
import tensor.BroadcastPlan;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.TensorPrimitiveBuilder;

/**
 * Elementwise comparison operations for floating tensors.
 *
 * <p>Inputs are broadcast using the same rules as numeric binary operations.
 * Results are non-differentiable {@link DataType#BOOL} tensors and inputs are
 * not mutated.</p>
 */
public final class TensorCompareOps {
    private TensorCompareOps() {
    }

    /**
     * Compares whether each element of {@code first} is greater than {@code second}.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     * @throws NullPointerException if either input is null
     * @throws IllegalArgumentException if inputs are non-floating, integral,
     *                                  boolean, or not broadcast-compatible
     */
    public static Tensor greaterThan(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new greaterThan(plan), "gt");
    }

    /**
     * Compares whether each element of {@code first} is less than {@code second}.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     * @throws NullPointerException if either input is null
     * @throws IllegalArgumentException if inputs are non-floating, integral,
     *                                  boolean, or not broadcast-compatible
     */
    public static Tensor lessThan(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new lessThan(plan), "lt");
    }

    /**
     * Compares whether each element of {@code first} is greater than or equal to {@code second}.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     * @throws NullPointerException if either input is null
     * @throws IllegalArgumentException if inputs are non-floating, integral,
     *                                  boolean, or not broadcast-compatible
     */
    public static Tensor greaterOrEqual(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new greaterOrEqual(plan), "ge");
    }

    /**
     * Compares whether each element of {@code first} is less than or equal to {@code second}.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     * @throws NullPointerException if either input is null
     * @throws IllegalArgumentException if inputs are non-floating, integral,
     *                                  boolean, or not broadcast-compatible
     */
    public static Tensor lessOrEqual(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new lessOrEqual(plan), "le");
    }

    /**
     * Compares whether corresponding broadcasted elements are equal.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     * @throws NullPointerException if either input is null
     * @throws IllegalArgumentException if inputs are non-floating, integral,
     *                                  boolean, or not broadcast-compatible
     */
    public static Tensor equalTo(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        return compare(first, second, plan, new equalTo(plan), "eq");
    }

    /**
     * Compares whether corresponding broadcasted elements are unequal.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return broadcasted boolean tensor
     * @throws NullPointerException if either input is null
     * @throws IllegalArgumentException if inputs are non-floating, integral,
     *                                  boolean, or not broadcast-compatible
     */
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
