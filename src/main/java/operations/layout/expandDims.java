package operations.layout;

import operations.Operation;

/**
 * Inserts a singleton dimension at the supplied axis.
 *
 * <p>This is a shape-only layout operation; tensor data is not copied by the
 * descriptor itself.</p>
 */
public final class expandDims implements Operation {
    private final int axis;

    /**
     * Creates a descriptor for one axis.
     *
     * @param axis axis at which the singleton dimension is interpreted
     */
    public expandDims(int axis) {
        this.axis = axis;
    }

    /**
     * Returns the axis used by this layout descriptor.
     *
     * @return axis supplied by the tensor front end
     */
    public int getAxis() {
        return axis;
    }

    @Override
    public OpType opType() {
        return OpType.EXPAND_DIMS;
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
    public OpControlTrait controlTrait() {
        return OpControlTrait.NONE;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.SHAPE_VIEW;
    }

    @Override
    public String getExpression() {
        return "expandDims(" + axis + ")";
    }
}
