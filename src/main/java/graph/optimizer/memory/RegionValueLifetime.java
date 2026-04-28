package graph.optimizer.memory;

import graph.optimizer.region.MaterializationDecision;
import graph.optimizer.region.RegionValueRef;
import graph.optimizer.region.ValueTypeContract;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

/**
 * Lifetime of a region value across optimized execution units.
 *
 * @param valueRef region value reference
 * @param birthStep unit step where the value is produced
 * @param lastUseStep last unit or graph step that consumes the value
 * @param elementCount value size in elements
 * @param decision materialization decision
 * @param typeContract dtype contract for storage and transport
 * @param producerRegionId producing region id
 * @param producerUnitId producing unit id
 * @param consumerRegionIds consuming region ids
 * @param consumerUnitIds consuming unit ids
 */
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

    /**
     * Returns whether this value crosses from one region to another.
     *
     * @return {@code true} when any consumer region differs from the producer region
     */
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
