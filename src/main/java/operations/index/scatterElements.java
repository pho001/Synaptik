package operations.index;

import operations.Operation;

/**
 * Rank-preserving functional scatter along one axis.
 */
public final class scatterElements implements Operation {
    private final int axis;
    private final ScatterReduction reduction;

    public scatterElements(int axis, ScatterReduction reduction) {
        this.axis = axis;
        this.reduction = reduction == null ? ScatterReduction.NONE : reduction;
    }

    @Override
    public OpType opType() {
        return OpType.SCATTER_ELEMENTS;
    }

    @Override
    public String getExpression() {
        return "scatterElements(axis=" + axis + ",reduction=" + reduction + ")";
    }

    public int getAxis() {
        return axis;
    }

    public ScatterReduction getReduction() {
        return reduction;
    }
}
