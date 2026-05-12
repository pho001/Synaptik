package operations.elementwise.unary;

import operations.Operation;

/**
 * Computes the elementwise Gaussian error function.
 */
public final class erf implements Operation {
    @Override
    public OpType opType() {
        return OpType.ERF;
    }

    @Override
    public String getExpression() {
        return "erf";
    }
}
