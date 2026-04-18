package operations.elementwise.logical;
import operations.Operation;

public final class logicalNot implements Operation {
    @Override
    public OpType opType() {
        return OpType.LOGICAL_NOT;
    }

    @Override
    public String getExpression() {
        return "logicalNot";
    }
}
