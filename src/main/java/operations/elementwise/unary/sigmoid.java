package operations.elementwise.unary;

import operations.Operation;

/**
 * Applies the sigmoid activation elementwise.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class sigmoid implements Operation {
    @Override
    public OpType opType() {
        return OpType.SIGMOID;
    }

    @Override
    public String getExpression() {
        return "sigmoid";
    }
}
