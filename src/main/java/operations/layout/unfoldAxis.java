package operations.layout;

import operations.Operation;

/**
 * Materializes 1-D sliding windows along one tensor axis.
 */
public final class unfoldAxis implements Operation {
    private final int axis;
    private final int size;
    private final int step;

    public unfoldAxis(int axis, int size, int step) {
        if (size <= 0) {
            throw new IllegalArgumentException("unfold size must be positive.");
        }
        if (step <= 0) {
            throw new IllegalArgumentException("unfold step must be positive.");
        }
        this.axis = axis;
        this.size = size;
        this.step = step;
    }

    public int getAxis() {
        return axis;
    }

    public int getSize() {
        return size;
    }

    public int getStep() {
        return step;
    }

    @Override
    public OpType opType() {
        return OpType.UNFOLD_AXIS;
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
        return OpComputationalCost.TRIVIAL;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.SHAPE_VIEW;
    }

    @Override
    public String getExpression() {
        return "unfold(axis=" + axis + ", size=" + size + ", step=" + step + ")";
    }
}
