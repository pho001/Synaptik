package operations.reduction;

import operations.Operation;

/**
 * Multiplies tensor values along one dimension.
 */
public final class reduceProd implements Operation {
    private final int dimension;
    private final boolean keepDims;

    public reduceProd(int dimension) {
        this(dimension, false);
    }

    public reduceProd(int dimension, boolean keepDims) {
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    @Override
    public OpType opType() {
        return OpType.REDUCE_PROD;
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
        return "reduceProd";
    }

    public int getDimension() {
        return dimension;
    }

    public boolean keepDims() {
        return keepDims;
    }
}
