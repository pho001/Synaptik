package operations.layout;

import operations.Operation;

/**
 * Represents an identity operation that preserves shape, dtype, and values.
 */
public final class noop implements Operation {
    @Override
    public OpType opType() {
        return OpType.NOOP;
    }

    @Override
    public OpArityClass arityClass() {
        return OpArityClass.SPECIAL;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.SPECIAL;
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
        return OpResultKind.UNKNOWN;
    }

    @Override
    public String getExpression() {
        return "noop";
    }
}
