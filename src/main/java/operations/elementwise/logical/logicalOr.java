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
    public String getExpression() {
        return "logicalOr";
    }
}
