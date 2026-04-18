package operations.elementwise.unary;

import operations.Operation;

public final class sigmoid implements Operation {
    @Override
    public OpType opType() {
        return OpType.SIGMOID;
    }

    @Override
    public String getExpression() {
        return "sigmoid";
    }
}
