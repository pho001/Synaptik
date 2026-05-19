package operations.elementwise.compare;
import operations.Operation;

import tensor.layout.BroadcastPlan;

/**
 * Performs an elementwise inequality comparison.
 *
 * <p>Inputs follow the supplied {@link BroadcastPlan}; the result is a boolean
 * tensor with the broadcasted output shape.</p>
 */
public final class notEqualTo implements Operation {
    private final BroadcastPlan broadcastPlan;

    /**
     * Creates a comparison descriptor.
     *
     * @param broadcastPlan precomputed broadcast metadata for the operands
     */
    public notEqualTo(BroadcastPlan broadcastPlan) {
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
        return OpType.NE;
    }

    @Override
    public String getExpression() {
        return "notEqualTo";
    }
}
