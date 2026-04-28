package operations.elementwise.unary;

import operations.Operation;

/**
 * Computes an approximate elementwise hyperbolic tangent.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class fastTanh implements Operation {
    @Override
    public OpType opType() {
        return OpType.FAST_TANH;
    }

    @Override
    public String getExpression() {
        return "fastTanh";
    }
}
