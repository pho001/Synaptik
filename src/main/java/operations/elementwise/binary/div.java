package operations.elementwise.binary;

import operations.Operation;
import tensor.BroadcastPlan;

/**
 * Divides the left tensor by the right tensor elementwise.
 *
 * <p>Operands use standard tensor broadcasting when a {@link BroadcastPlan}
 * is supplied; otherwise inputs are expected to already share the same output
 * shape. Arithmetic result dtype is resolved by the surrounding tensor/backend
 * contract.</p>
 */
public final class div implements Operation {
    private final BroadcastPlan broadcastPlan;

    /**
     * Creates an unplanned elementwise {@code /} descriptor.
     */
    public div() {
        this(null);
    }

    /**
     * Creates an elementwise {@code /} descriptor with an optional broadcast plan.
     *
     * @param broadcastPlan precomputed broadcast metadata, or {@code null} when
     *        no explicit plan is attached
     */
    public div(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    /**
     * Returns the broadcast metadata attached to this descriptor.
     *
     * @return broadcast plan, or {@code null} for unplanned same-shape execution
     */
    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public OpType opType() {
        return OpType.DIV;
    }

    @Override
    public String getExpression() {
        return "/";
    }
}
