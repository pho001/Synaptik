package planning.memory;

import planning.region.MaterializationDecision;
import planning.value.GraphValueRef;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

/**
 * Producer and consumer flow for one region value.
 *
 * @param valueRef graph value reference
 * @param decision materialization decision
 * @param producerRegionId producing region id
 * @param producerUnitId producing unit id
 * @param consumerRegionIds consuming region ids
 * @param consumerUnitIds consuming unit ids
 */
public record StructuralValueFlow(
        GraphValueRef valueRef,
        MaterializationDecision decision,
        String producerRegionId,
        String producerUnitId,
        List<String> consumerRegionIds,
        List<String> consumerUnitIds
) {
    public StructuralValueFlow {
        valueRef = Objects.requireNonNull(valueRef, "valueRef cannot be null");
        decision = Objects.requireNonNull(decision, "decision cannot be null");
        consumerRegionIds = List.copyOf(consumerRegionIds == null ? List.of() : new LinkedHashSet<>(consumerRegionIds));
        consumerUnitIds = List.copyOf(consumerUnitIds == null ? List.of() : new LinkedHashSet<>(consumerUnitIds));
    }

    /**
     * Returns whether any consumer is in a different region from the producer.
     *
     * @return {@code true} for cross-region flow
     */
    public boolean hasCrossRegionConsumer() {
        if (producerRegionId == null || producerRegionId.isBlank()) {
            return false;
        }
        for (String consumerRegionId : consumerRegionIds) {
            if (consumerRegionId != null
                    && !consumerRegionId.isBlank()
                    && !producerRegionId.equals(consumerRegionId)) {
                return true;
            }
        }
        return false;
    }
}
