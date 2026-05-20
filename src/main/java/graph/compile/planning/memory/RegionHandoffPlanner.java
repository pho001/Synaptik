package graph.compile.planning.memory;

import tensor.DataType;

import java.util.ArrayList;
import java.util.List;

final class RegionHandoffPlanner {
    private RegionHandoffPlanner() {
    }

    static List<RegionHandoffRequirement> plan(List<RegionValueLifetime> lifetimes) {
        ArrayList<RegionHandoffRequirement> requirements = new ArrayList<>();
        for (RegionValueLifetime lifetime : lifetimes) {
            for (int i = 0; i < lifetime.consumerRegionIds().size(); i++) {
                String consumerRegionId = lifetime.consumerRegionIds().get(i);
                if (consumerRegionId == null
                        || consumerRegionId.isBlank()
                        || consumerRegionId.equals(lifetime.producerRegionId())) {
                    continue;
                }
                String consumerUnitId = i < lifetime.consumerUnitIds().size()
                        ? lifetime.consumerUnitIds().get(i)
                        : null;
                requirements.add(new RegionHandoffRequirement(
                        lifetime.valueRef(),
                        lifetime.producerRegionId(),
                        lifetime.producerUnitId(),
                        consumerRegionId,
                        consumerUnitId,
                        transportTypeFor(lifetime),
                        lifetime.decision()
                ));
            }
        }
        return List.copyOf(requirements);
    }

    private static DataType transportTypeFor(RegionValueLifetime lifetime) {
        return switch (lifetime.decision()) {
            case CONTINUE -> lifetime.typeContract().transportType();
            case MATERIALIZE, VIRTUALIZE -> lifetime.typeContract().storageType();
        };
    }
}
