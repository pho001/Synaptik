package graph.optimizer.memory;

import graph.optimizer.region.MaterializationDecision;
import graph.optimizer.region.RegionValueRef;

import java.util.Objects;

public record MaterializationPlanEntry(
        RegionValueRef valueRef,
        MaterializationDecision decision,
        boolean requiredMaterialized,
        boolean allocatesStorage
) {
    public MaterializationPlanEntry {
        valueRef = Objects.requireNonNull(valueRef, "valueRef cannot be null");
        decision = Objects.requireNonNull(decision, "decision cannot be null");
    }
}
