package operations.linalg;

import operations.Operation;

public final class matmul implements Operation {
    @Override
    public OpType opType() {
        return OpType.MATMUL;
    }

    @Override
    public String getExpression() {
        return "matmul";
    }
}
