package operations.elementwise.where;
import operations.Operation;

/**
 * Selects elements from two value tensors according to a boolean condition.
 *
 * <p>The condition and value operands are broadcast by the surrounding tensor
 * operation before execution; the result has the broadcasted value dtype and
 * shape.</p>
 */
public final class where implements Operation {
    @Override
    public OpType opType() {
        return OpType.WHERE;
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
        return OpSemanticFamily.SELECTION;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.CHEAP;
    }

    @Override
    public OpControlTrait controlTrait() {
        return OpControlTrait.SELECT_MASK;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "where";
    }
}
