package planning.memory;

import planning.partition.execution.MaterializationDecision;
import planning.value.GraphValueRef;
import tensor.DataType;

import java.util.Objects;

/**
 * Requirement for moving a partition value from a producer unit to a consumer unit.
 *
 * @param valueRef graph value reference
 * @param producerPartitionId producer partition id
 * @param producerUnitId producer execution unit id
 * @param consumerPartitionId consumer partition id
 * @param consumerUnitId consumer execution unit id
 * @param transportType dtype used across the handoff
 * @param decision materialization decision for the handoff value
 */
public record PartitionHandoffRequirement(
        GraphValueRef valueRef,
        String producerPartitionId,
        String producerUnitId,
        String consumerPartitionId,
        String consumerUnitId,
        DataType transportType,
        MaterializationDecision decision
) {
    public PartitionHandoffRequirement {
        valueRef = Objects.requireNonNull(valueRef, "valueRef cannot be null");
        transportType = Objects.requireNonNull(transportType, "transportType cannot be null");
        decision = Objects.requireNonNull(decision, "decision cannot be null");
    }
}
