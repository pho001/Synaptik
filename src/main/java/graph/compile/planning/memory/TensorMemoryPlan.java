package graph.compile.planning.memory;

import tensor.Tensor;

import java.util.Map;
import java.util.Objects;

public record TensorMemoryPlan(
        Map<Tensor, NodeLifetime> lifetimes,
        Map<Tensor, ReusableInterval> reusableIntervals,
        Map<Tensor, Integer> slotByOwner,
        Map<Integer, Integer> slotSizes
) {
    public TensorMemoryPlan {
        lifetimes = Map.copyOf(Objects.requireNonNull(lifetimes, "lifetimes cannot be null"));
        reusableIntervals = Map.copyOf(Objects.requireNonNull(reusableIntervals, "reusableIntervals cannot be null"));
        slotByOwner = Map.copyOf(Objects.requireNonNull(slotByOwner, "slotByOwner cannot be null"));
        slotSizes = Map.copyOf(Objects.requireNonNull(slotSizes, "slotSizes cannot be null"));
    }

    public static TensorMemoryPlan empty() {
        return new TensorMemoryPlan(Map.of(), Map.of(), Map.of(), Map.of());
    }
}
