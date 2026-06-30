package trace.compile;

/** Immutable diagnostic snapshot of a materialization-aware partition cost. */
public record MaterializationCostTrace(
        String preset,
        int boundaryCount,
        long estimatedTransferBytes,
        long layoutFallbackBytes,
        long estimatedComputeWork,
        long avoidedIntermediateBytes,
        double dispatchCost,
        double finalScore,
        String reasonCode,
        String fallbackMode,
        String layoutClass
) {
    public MaterializationCostTrace {
        preset = preset == null ? "" : preset;
        boundaryCount = Math.max(0, boundaryCount);
        estimatedTransferBytes = Math.max(0L, estimatedTransferBytes);
        layoutFallbackBytes = Math.max(0L, layoutFallbackBytes);
        estimatedComputeWork = Math.max(0L, estimatedComputeWork);
        avoidedIntermediateBytes = Math.max(0L, avoidedIntermediateBytes);
        dispatchCost = Math.max(0.0d, dispatchCost);
        reasonCode = reasonCode == null ? "" : reasonCode;
        fallbackMode = fallbackMode == null ? "" : fallbackMode;
        layoutClass = layoutClass == null ? "" : layoutClass;
    }
}
