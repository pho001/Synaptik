package operations.elementwise.unary;

import operations.Operation;

public final class relu implements Operation {
    @Override
    public OpType opType() {
        return OpType.RELU;
    }

    @Override
    public String getExpression() {
        return "relu";
    }
}
