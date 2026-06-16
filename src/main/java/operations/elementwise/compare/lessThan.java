package operations.elementwise.compare;
import operations.Operation;

import tensor.layout.BroadcastPlan;

/**
 * Performs an elementwise less-than comparison.
 *
 * <p>Inputs follow the supplied {@link BroadcastPlan}; the result is a boolean
 * tensor with the broadcasted output shape.</p>
 */
public final class lessThan implements Operation {
    private final BroadcastPlan broadcastPlan;

    /**
     * Creates a comparison descriptor.
     *
     * @param broadcastPlan precomputed broadcast metadata for the operands
     */
    public lessThan(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    /**
     * Returns the broadcast metadata attached to this comparison.
     *
     * @return broadcast plan for the compared operands
     */
    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.LT;
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
        return OpSemanticFamily.COMPARISON;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.CHEAP;
    }

    @Override
    public OpControlTrait controlTrait() {
        return OpControlTrait.BRANCHLESS;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.BOOLEAN;
    }

    @Override
    public String getExpression() {
        return "lessThan";
    }
}
