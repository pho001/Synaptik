package graph.optimizer.memory;

import graph.optimizer.region.MaterializationDecision;
import graph.optimizer.region.RegionValueRef;
import tensor.DataType;

import java.util.Objects;

public record RegionHandoffRequirement(
        RegionValueRef valueRef,
        String producerRegionId,
        String producerUnitId,
        String consumerRegionId,
        String consumerUnitId,
        DataType transportType,
        MaterializationDecision decision
) {
    public RegionHandoffRequirement {
        valueRef = Objects.requireNonNull(valueRef, "valueRef cannot be null");
        transportType = Objects.requireNonNull(transportType, "transportType cannot be null");
        decision = Objects.requireNonNull(decision, "decision cannot be null");
    }
}
