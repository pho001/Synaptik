package operations.elementwise.unary;
import operations.Operation;

/**
 * Computes the elementwise absolute value.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
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
