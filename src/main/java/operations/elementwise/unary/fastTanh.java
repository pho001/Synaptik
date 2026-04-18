package operations.elementwise.unary;

import operations.Operation;

public final class fastTanh implements Operation {
    @Override
    public OpType opType() {
        return OpType.FAST_TANH;
    }

    @Override
    public String getExpression() {
        return "fastTanh";
    }
}
