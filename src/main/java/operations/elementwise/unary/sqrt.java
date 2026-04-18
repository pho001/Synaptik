package operations.elementwise.unary;

import operations.Operation;

public final class sqrt implements Operation {
    @Override
    public OpType opType() {
        return OpType.SQRT;
    }

    @Override
    public String getExpression() {
        return "sqrt";
    }

    @Override
    public boolean isCheap() {
        return false;
    }
}
