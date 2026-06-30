package planning.memory;

import planning.region.MaterializationDecision;
import planning.value.GraphValueRef;

import java.util.Objects;

/**
 * Materialization decision for one planned region value.
 *
 * @param valueRef graph value reference
 * @param decision planner decision for storage materialization
 * @param requiredMaterialized whether graph semantics require materialized storage
 * @param allocatesStorage whether runtime binding should allocate storage for this value
 */
public record MaterializationPlanEntry(
        GraphValueRef valueRef,
        MaterializationDecision decision,
        boolean requiredMaterialized,
        boolean allocatesStorage
) {
    public MaterializationPlanEntry {
        valueRef = Objects.requireNonNull(valueRef, "valueRef cannot be null");
        decision = Objects.requireNonNull(decision, "decision cannot be null");
    }
}
