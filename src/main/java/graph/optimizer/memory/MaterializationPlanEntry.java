package graph.optimizer.memory;

import graph.optimizer.region.MaterializationDecision;
import graph.optimizer.region.RegionValueRef;

import java.util.Objects;

/**
 * Materialization decision for one optimized region value.
 *
 * @param valueRef region value reference
 * @param decision planner decision for storage materialization
 * @param requiredMaterialized whether graph semantics require materialized storage
 * @param allocatesStorage whether runtime binding should allocate storage for this value
 */
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
