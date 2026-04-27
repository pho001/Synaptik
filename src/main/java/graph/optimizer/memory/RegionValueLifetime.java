package graph.optimizer.memory;

import graph.optimizer.region.MaterializationDecision;
import graph.optimizer.region.RegionValueRef;
import graph.optimizer.region.ValueTypeContract;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

public record RegionValueLifetime(
        RegionValueRef valueRef,
        int birthStep,
        int lastUseStep,
        int elementCount,
        MaterializationDecision decision,
        ValueTypeContract typeContract,
        String producerRegionId,
        String producerUnitId,
        List<String> consumerRegionIds,
        List<String> consumerUnitIds
) {
    public RegionValueLifetime {
        valueRef = Objects.requireNonNull(valueRef, "valueRef cannot be null");
        if (birthStep < 0 || lastUseStep < birthStep || elementCount < 0) {
            throw new IllegalArgumentException("Invalid lifetime bounds");
        }
        decision = Objects.requireNonNull(decision, "decision cannot be null");
        typeContract = Objects.requireNonNull(typeContract, "typeContract cannot be null");
        consumerRegionIds = List.copyOf(consumerRegionIds == null ? List.of() : new LinkedHashSet<>(consumerRegionIds));
        consumerUnitIds = List.copyOf(consumerUnitIds == null ? List.of() : new LinkedHashSet<>(consumerUnitIds));
    }

    public boolean isCrossRegion() {
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
