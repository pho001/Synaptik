package planning.memory;

import planning.partition.execution.MaterializationDecision;
import planning.value.GraphValueRef;
import planning.partition.execution.ValueTypeContract;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

/**
 * Lifetime of a partition value across optimized execution units.
 *
 * @param valueRef graph value reference
 * @param birthStep unit step where the value is produced
 * @param lastUseStep last unit or graph step that consumes the value
 * @param elementCount value size in elements
 * @param decision materialization decision
 * @param typeContract dtype contract for storage and transport
 * @param producerPartitionId producing partition id
 * @param producerUnitId producing unit id
 * @param consumerPartitionIds consuming partition ids
 * @param consumerUnitIds consuming unit ids
 */
public record PartitionValueLifetime(
        GraphValueRef valueRef,
        int birthStep,
        int lastUseStep,
        int elementCount,
        MaterializationDecision decision,
        ValueTypeContract typeContract,
        String producerPartitionId,
        String producerUnitId,
        List<String> consumerPartitionIds,
        List<String> consumerUnitIds
) {
    public PartitionValueLifetime {
        valueRef = Objects.requireNonNull(valueRef, "valueRef cannot be null");
        if (birthStep < 0 || lastUseStep < birthStep || elementCount < 0) {
            throw new IllegalArgumentException("Invalid lifetime bounds");
        }
        decision = Objects.requireNonNull(decision, "decision cannot be null");
        typeContract = Objects.requireNonNull(typeContract, "typeContract cannot be null");
        consumerPartitionIds = List.copyOf(consumerPartitionIds == null ? List.of() : new LinkedHashSet<>(consumerPartitionIds));
        consumerUnitIds = List.copyOf(consumerUnitIds == null ? List.of() : new LinkedHashSet<>(consumerUnitIds));
    }

    /**
     * Returns whether this value crosses from one partition to another.
     *
     * @return {@code true} when any consumer partition differs from the producer partition
     */
    public boolean isCrossPartition() {
        if (producerPartitionId == null || producerPartitionId.isBlank()) {
            return false;
        }
        for (String consumerPartitionId : consumerPartitionIds) {
            if (consumerPartitionId != null
                    && !consumerPartitionId.isBlank()
                    && !producerPartitionId.equals(consumerPartitionId)) {
                return true;
            }
        }
        return false;
    }
}
