package operations.layout;

import operations.Operation;

import java.util.Arrays;

/**
 * Pads every tensor axis with a constant scalar value.
 */
public final class pad implements Operation {
    private final int[] before;
    private final int[] after;
    private final double constantValue;

    public pad(int[] before, int[] after, double constantValue) {
        this.before = before == null ? new int[0] : before.clone();
        this.after = after == null ? new int[0] : after.clone();
        this.constantValue = constantValue;
    }

    @Override
    public OpType opType() {
        return OpType.PAD;
    }

    @Override
    public OpArityClass arityClass() {
        return OpArityClass.LAYOUT;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.LAYOUT;
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
        return OpResultKind.SHAPE_VIEW;
    }

    @Override
    public String getExpression() {
        return "pad(before=" + Arrays.toString(before)
                + ",after=" + Arrays.toString(after)
                + ",value=" + constantValue + ")";
    }

    public int[] getBefore() {
        return before.clone();
    }

    public int[] getAfter() {
        return after.clone();
    }

    public double getConstantValue() {
        return constantValue;
    }
}
