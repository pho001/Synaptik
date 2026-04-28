package operations.elementwise.unary;

import operations.Operation;

/**
 * Computes the elementwise square root.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class sqrt implements Operation {
    @Override
    public OpType opType() {
        return OpType.SQRT;
    }

    @Override
    public String getExpression() {
        return "sqrt";
    }

    @Override
    public boolean isCheap() {
        return false;
    }
}
