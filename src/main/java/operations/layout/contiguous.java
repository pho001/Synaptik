package operations.layout;

import operations.Operation;

public final class contiguous implements Operation {
    @Override
    public OpType opType() {
        return OpType.CONTIGUOUS;
    }

    @Override
    public String getExpression() {
        return "contiguous";
    }
}
