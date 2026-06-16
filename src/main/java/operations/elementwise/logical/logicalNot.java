package operations.elementwise.logical;
import operations.Operation;

/**
 * Inverts a boolean-compatible tensor elementwise.
 *
 * <p>The result has the same shape as the input and boolean dtype.</p>
 */
public final class logicalNot implements Operation {
    @Override
    public OpType opType() {
        return OpType.LOGICAL_NOT;
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
        return OpSemanticFamily.LOGICAL;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.CHEAP;
    }

    @Override
    public OpControlTrait controlTrait() {
        return OpControlTrait.BOOL_LOGIC;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.BOOLEAN;
    }

    @Override
    public String getExpression() {
        return "logicalNot";
    }
}
