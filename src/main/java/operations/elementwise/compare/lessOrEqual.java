package operations.elementwise.compare;
import operations.Operation;

import tensor.layout.BroadcastPlan;

/**
 * Performs an elementwise less-than-or-equal comparison.
 *
 * <p>Inputs follow the supplied {@link BroadcastPlan}; the result is a boolean
 * tensor with the broadcasted output shape.</p>
 */
public final class lessOrEqual implements Operation {
    private final BroadcastPlan broadcastPlan;

    /**
     * Creates a comparison descriptor.
     *
     * @param broadcastPlan precomputed broadcast metadata for the operands
     */
    public lessOrEqual(BroadcastPlan broadcastPlan) {
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
        return OpType.LE;
    }

    @Override
    public String getExpression() {
        return "lessOrEqual";
    }
}
