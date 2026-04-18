package operations.elementwise.unary;

import operations.Operation;

public final class neg implements Operation {
    @Override
    public OpType opType() {
        return OpType.NEG;
    }

    @Override
    public String getExpression() {
        return "neg";
    }

    @Override
    public boolean isCheap() {
        return true;
    }
}
