package operations.elementwise.unary;

import operations.Operation;

/**
 * Computes the elementwise exponential.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class exp implements Operation {
    @Override
    public OpType opType() {
        return OpType.EXP;
    }

    @Override
    public String getExpression() {
        return "exp";
    }
}
