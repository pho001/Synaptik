package operations.elementwise.binary;
import operations.Operation;

import tensor.layout.BroadcastPlan;

/**
 * Routes an upstream gradient through the selected operand of an elementwise
 * maximum operation.
 *
 * <p>The descriptor records the original broadcast relationship and whether
 * this gradient belongs to the first or second input; backend kernels handle any
 * required reduction over broadcast axes.</p>
 */
public final class maxGrad implements Operation {
    private final BroadcastPlan broadcastPlan;
    private final boolean forFirstInput;

    /**
     * Creates a maximum gradient descriptor.
     *
     * @param broadcastPlan broadcast metadata from the forward operation
     * @param forFirstInput {@code true} for the first operand gradient,
     *        {@code false} for the second operand gradient
     */
    public maxGrad(BroadcastPlan broadcastPlan, boolean forFirstInput) {
        this.broadcastPlan = broadcastPlan;
        this.forFirstInput = forFirstInput;
    }

    /**
     * Returns the forward broadcast metadata used to shape the gradient.
     *
     * @return broadcast plan from the paired forward operation
     */
    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    /**
     * Indicates which forward operand this gradient targets.
     *
     * @return {@code true} for the first input, {@code false} for the second
     */
    public boolean isForFirstInput() {
        return forFirstInput;
    }

    @Override
    public OpType opType() {
        return OpType.MAX_GRAD;
    }

    @Override
    public String getExpression() {
        return forFirstInput ? "max_grad_a" : "max_grad_b";
    }
}
