package operations.elementwise.unary;

import operations.Operation;

/**
 * Negates each element.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class neg implements Operation {
    @Override
    public OpType opType() {
        return OpType.NEG;
    }

    @Override
    public String getExpression() {
        return "neg";
    }

    @Override
    public boolean isCheap() {
        return true;
    }
}
