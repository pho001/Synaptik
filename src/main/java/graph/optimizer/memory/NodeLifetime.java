package graph.optimizer.memory;

import tensor.Tensor;

public record NodeLifetime(
        int birthIndex,
        int lastReadIndex,
        MemoryRole role,
        Tensor storageOwner
) {
}
