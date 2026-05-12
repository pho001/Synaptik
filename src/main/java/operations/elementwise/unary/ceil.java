package operations.elementwise.unary;

import operations.Operation;

/**
 * Computes the elementwise ceiling.
 */
public final class ceil implements Operation {
    @Override
    public OpType opType() {
        return OpType.CEIL;
    }

    @Override
    public String getExpression() {
        return "ceil";
    }
}
