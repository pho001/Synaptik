package operations.reduction;

import operations.Operation;

/**
 * Computes cumulative sums along one axis.
 */
public final class cumSum implements Operation {
    private final int axis;
    private final boolean exclusive;
    private final boolean reverse;

    public cumSum(int axis) {
        this(axis, false, false);
    }

    public cumSum(int axis, boolean exclusive, boolean reverse) {
        this.axis = axis;
        this.exclusive = exclusive;
        this.reverse = reverse;
    }

    @Override
    public OpType opType() {
        return OpType.CUMSUM;
    }

    @Override
    public OpArityClass arityClass() {
        return OpArityClass.REDUCTION;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.REDUCTION;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.MEDIUM;
    }

    @Override
    public OpControlTrait controlTrait() {
        return OpControlTrait.NONE;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "cumSum";
    }

    public int getAxis() {
        return axis;
    }

    public boolean isExclusive() {
        return exclusive;
    }

    public boolean isReverse() {
        return reverse;
    }
}
