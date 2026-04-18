package operations.elementwise.unary;

import operations.Operation;

public final class tanh implements Operation {
    @Override
    public OpType opType() {
        return OpType.TANH;
    }

    @Override
    public String getExpression() {
        return "tanh";
    }
}
