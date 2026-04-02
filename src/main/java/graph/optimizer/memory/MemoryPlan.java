package graph.optimizer.memory;

import tensor.Tensor;

import java.util.Map;
import java.util.Objects;

public final class MemoryPlan {
    private final Map<Tensor, NodeLifetime> lifetimes;

    public MemoryPlan(Map<Tensor, NodeLifetime> lifetimes) {
        this.lifetimes = Map.copyOf(Objects.requireNonNull(lifetimes, "lifetimes cannot be null"));
    }

    public NodeLifetime lifetimeOf(Tensor tensor) {
        NodeLifetime lifetime = lifetimes.get(tensor);
        if (lifetime == null) {
            throw new IllegalArgumentException("Missing lifetime for tensor: " + tensor.getLabel());
        }
        return lifetime;
    }

    public Tensor storageOwnerOf(Tensor tensor) {
        return lifetimeOf(tensor).storageOwner();
    }

    public int lastReadIndexOf(Tensor tensor) {
        return lifetimeOf(tensor).lastReadIndex();
    }

    public boolean isReusableOwner(Tensor tensor) {
        NodeLifetime lifetime = lifetimeOf(tensor);
        if (lifetime.storageOwner() != tensor) {
            return false;
        }
        return switch (lifetime.role()) {
            case FORWARD_TEMP, BACKWARD_TEMP -> true;
            default -> false;
        };
    }
}
