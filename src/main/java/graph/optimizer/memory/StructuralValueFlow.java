package graph.optimizer.memory;

import graph.optimizer.region.MaterializationDecision;
import graph.optimizer.region.RegionValueRef;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

public record StructuralValueFlow(
        RegionValueRef valueRef,
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
