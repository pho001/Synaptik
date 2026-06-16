package operations.elementwise.logical;
import operations.Operation;

import tensor.layout.BroadcastPlan;

/**
 * Performs elementwise logical OR over boolean-compatible tensors.
 *
 * <p>Inputs follow the supplied {@link BroadcastPlan}; the result is a boolean
 * tensor with the broadcasted output shape.</p>
 */
public final class logicalOr implements Operation {
    private final BroadcastPlan broadcastPlan;

    /**
     * Creates a logical descriptor.
     *
     * @param broadcastPlan precomputed broadcast metadata for the operands
     */
    public logicalOr(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    /**
     * Returns the broadcast metadata attached to this logical operation.
     *
     * @return broadcast plan for the operands
     */
    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.LOGICAL_OR;
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
        return "logicalOr";
    }
}
