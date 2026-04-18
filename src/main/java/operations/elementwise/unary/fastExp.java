package operations.elementwise.unary;

import operations.Operation;

public final class fastExp implements Operation {
    @Override
    public OpType opType() {
        return OpType.FAST_EXP;
    }

    @Override
    public String getExpression() {
        return "fastExp";
    }
}
