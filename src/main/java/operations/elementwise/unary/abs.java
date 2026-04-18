package operations.elementwise.unary;
import operations.Operation;

public final class abs implements Operation {
    @Override
    public OpType opType() {
        return OpType.ABS;
    }

    @Override
    public String getExpression() {
        return "abs";
    }
}
