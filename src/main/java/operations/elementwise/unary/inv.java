package operations.elementwise.unary;

import operations.Operation;

public final class inv implements Operation {
    @Override
    public OpType opType() {
        return OpType.INV;
    }

    @Override
    public String getExpression() {
        return "inv";
    }
}
