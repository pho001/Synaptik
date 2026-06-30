package planning.memory;

import planning.region.MaterializationDecision;
import planning.value.GraphValueRef;
import tensor.DataType;

import java.util.Objects;

/**
 * Requirement for moving a region value from a producer unit to a consumer unit.
 *
 * @param valueRef graph value reference
 * @param producerRegionId producer region id
 * @param producerUnitId producer execution unit id
 * @param consumerRegionId consumer region id
 * @param consumerUnitId consumer execution unit id
 * @param transportType dtype used across the handoff
 * @param decision materialization decision for the handoff value
 */
public record RegionHandoffRequirement(
        GraphValueRef valueRef,
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
