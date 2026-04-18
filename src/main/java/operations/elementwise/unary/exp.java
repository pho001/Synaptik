package operations.elementwise.unary;

import operations.Operation;

public final class exp implements Operation {
    @Override
    public OpType opType() {
        return OpType.EXP;
    }

    @Override
    public String getExpression() {
        return "exp";
    }
}
