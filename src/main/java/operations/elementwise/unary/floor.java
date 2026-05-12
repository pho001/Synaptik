package operations.elementwise.unary;

import operations.Operation;

/**
 * Computes the elementwise floor.
 */
public final class floor implements Operation {
    @Override
    public OpType opType() {
        return OpType.FLOOR;
    }

    @Override
    public String getExpression() {
        return "floor";
    }
}
