package planning.memory;

import tensor.DataType;

import java.util.ArrayList;
import java.util.List;

final class PartitionHandoffPlanner {
    private PartitionHandoffPlanner() {
    }

    static List<PartitionHandoffRequirement> plan(List<PartitionValueLifetime> lifetimes) {
        ArrayList<PartitionHandoffRequirement> requirements = new ArrayList<>();
        for (PartitionValueLifetime lifetime : lifetimes) {
            for (int i = 0; i < lifetime.consumerPartitionIds().size(); i++) {
                String consumerPartitionId = lifetime.consumerPartitionIds().get(i);
                if (consumerPartitionId == null
                        || consumerPartitionId.isBlank()
                        || consumerPartitionId.equals(lifetime.producerPartitionId())) {
                    continue;
                }
                String consumerUnitId = i < lifetime.consumerUnitIds().size()
                        ? lifetime.consumerUnitIds().get(i)
                        : null;
                requirements.add(new PartitionHandoffRequirement(
                        lifetime.valueRef(),
                        lifetime.producerPartitionId(),
                        lifetime.producerUnitId(),
                        consumerPartitionId,
                        consumerUnitId,
                        transportTypeFor(lifetime),
                        lifetime.decision()
                ));
            }
        }
        return List.copyOf(requirements);
    }

    private static DataType transportTypeFor(PartitionValueLifetime lifetime) {
        return switch (lifetime.decision()) {
            case CONTINUE -> lifetime.typeContract().transportType();
            case MATERIALIZE, VIRTUALIZE -> lifetime.typeContract().storageType();
        };
    }
}
