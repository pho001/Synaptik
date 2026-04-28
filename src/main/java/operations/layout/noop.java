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
    public String getExpression() {
        return "noop";
    }
}
