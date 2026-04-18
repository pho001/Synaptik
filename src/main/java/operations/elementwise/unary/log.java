package operations.elementwise.unary;

import operations.Operation;

public final class log implements Operation {
    @Override
    public OpType opType() {
        return OpType.LOG;
    }

    @Override
    public String getExpression() {
        return "log";
    }
}
