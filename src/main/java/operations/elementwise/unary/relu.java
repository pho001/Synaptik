package operations.elementwise.unary;

import operations.Operation;

/**
 * Applies the rectified linear unit activation elementwise.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class relu implements Operation {
    @Override
    public OpType opType() {
        return OpType.RELU;
    }

    @Override
    public String getExpression() {
        return "relu";
    }
}
