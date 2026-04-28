package operations.elementwise.logical;
import operations.Operation;

/**
 * Inverts a boolean-compatible tensor elementwise.
 *
 * <p>The result has the same shape as the input and boolean dtype.</p>
 */
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
