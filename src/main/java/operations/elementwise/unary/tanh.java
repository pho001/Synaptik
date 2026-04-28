package operations.elementwise.unary;

import operations.Operation;

/**
 * Computes the elementwise hyperbolic tangent.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class tanh implements Operation {
    @Override
    public OpType opType() {
        return OpType.TANH;
    }

    @Override
    public String getExpression() {
        return "tanh";
    }
}
