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
    public OpArityClass arityClass() {
        return OpArityClass.ELEMENT_WISE;
    }

    @Override
    public boolean isFusable() {
        return true;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.ARITHMETIC;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.MEDIUM;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "inv";
    }
}
