package operations.elementwise.unary;

import operations.Operation;

/**
 * Computes the elementwise sign: -1 for negative values, 0 for zero, and 1 for positive values.
 */
public final class sign implements Operation {
    @Override
    public OpType opType() {
        return OpType.SIGN;
    }

    @Override
    public String getExpression() {
        return "sign";
    }
}
