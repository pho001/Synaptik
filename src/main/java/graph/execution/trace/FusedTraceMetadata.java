package graph.execution.trace;

/**
 * Fused operation metadata for a step.
 *
 * @param numericContract fused numeric storage/compute/output contract
 * @param lowCostHint whether the fused op was classified as low-cost
 * @param dispatchFamily dispatch family label
 * @param schedulerSignature scheduler signature used for dispatch
 * @param executionBackend backend selected for fused execution
 * @param fusedNodeCount number of nodes represented by the fused op
 * @param fusedInputCount number of fused inputs
 * @param vectorFallbackReason reason fused vector dispatch fell back, or NONE
 */
public record FusedTraceMetadata(
        String numericContract,
        boolean lowCostHint,
        String dispatchFamily,
        String schedulerSignature,
        String executionBackend,
        int fusedNodeCount,
        int fusedInputCount,
        String vectorFallbackReason
) {
}
