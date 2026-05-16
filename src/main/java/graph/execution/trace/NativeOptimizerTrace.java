package graph.execution.trace;

import tensor.DataType;

/**
 * Optimizer route diagnostics captured for one trainable parameter update.
 *
 * @param optimizer optimizer implementation name
 * @param route backend/storage route used for the update
 * @param dataType parameter dtype
 * @param parameterNodeId compiled trainable parameter node id
 * @param gradientNodeId compiled gradient node id
 * @param elementCount updated element count
 * @param fallbackReason fallback or ineligibility reason, blank for native success
 */
public record NativeOptimizerTrace(
        String optimizer,
        String route,
        DataType dataType,
        int parameterNodeId,
        int gradientNodeId,
        int elementCount,
        String fallbackReason
) {
    public NativeOptimizerTrace {
        optimizer = optimizer == null ? "" : optimizer;
        route = route == null ? "" : route;
        elementCount = Math.max(0, elementCount);
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
    }
}
