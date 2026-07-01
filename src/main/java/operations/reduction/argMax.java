package operations.reduction;

import operations.Operation;

/**
 * Returns the first maximum-value index along one dimension.
 */
public final class argMax implements Operation {
    private final int dimension;
    private final boolean keepDims;
    private final ArgMaxTiePolicy tiePolicy;

    public argMax(int dimension, boolean keepDims) {
        this(dimension, keepDims, ArgMaxTiePolicy.FIRST_INDEX);
    }

    public argMax(int dimension, boolean keepDims, ArgMaxTiePolicy tiePolicy) {
        this.dimension = dimension;
        this.keepDims = keepDims;
        this.tiePolicy = tiePolicy == null ? ArgMaxTiePolicy.FIRST_INDEX : tiePolicy;
    }

    @Override
    public OpType opType() {
        return OpType.ARGMAX;
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
    public OpResultKind resultKind() {
        return OpResultKind.INDEX;
    }

    @Override
    public String getExpression() {
        return "argMax";
    }

    public int getDimension() {
        return dimension;
    }

    public boolean keepDims() {
        return keepDims;
    }

    public ArgMaxTiePolicy tiePolicy() {
        return tiePolicy;
    }
}
