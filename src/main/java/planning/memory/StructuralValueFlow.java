package planning.memory;

import planning.partition.execution.MaterializationDecision;
import planning.value.GraphValueRef;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

/**
 * Producer and consumer flow for one partition value.
 *
 * @param valueRef graph value reference
 * @param decision materialization decision
 * @param producerPartitionId producing partition id
 * @param producerUnitId producing unit id
 * @param consumerPartitionIds consuming partition ids
 * @param consumerUnitIds consuming unit ids
 */
public record StructuralValueFlow(
        GraphValueRef valueRef,
        MaterializationDecision decision,
        String producerPartitionId,
        String producerUnitId,
        List<String> consumerPartitionIds,
        List<String> consumerUnitIds
) {
    public StructuralValueFlow {
        valueRef = Objects.requireNonNull(valueRef, "valueRef cannot be null");
        decision = Objects.requireNonNull(decision, "decision cannot be null");
        consumerPartitionIds = List.copyOf(consumerPartitionIds == null ? List.of() : new LinkedHashSet<>(consumerPartitionIds));
        consumerUnitIds = List.copyOf(consumerUnitIds == null ? List.of() : new LinkedHashSet<>(consumerUnitIds));
    }

    /**
     * Returns whether any consumer is in a different partition from the producer.
     *
     * @return {@code true} for cross-partition flow
     */
    public boolean hasCrossPartitionConsumer() {
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
