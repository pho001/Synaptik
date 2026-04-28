package operations.elementwise.unary;

import operations.Operation;

/**
 * Computes the elementwise reciprocal.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class inv implements Operation {
    @Override
    public OpType opType() {
        return OpType.INV;
    }

    @Override
    public String getExpression() {
        return "inv";
    }
}
