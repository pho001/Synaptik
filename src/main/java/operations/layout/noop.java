package operations.layout;

import operations.Operation;

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
