package graph.compile.planning.memory;

import tensor.Tensor;

/**
 * Lifetime of a tensor's storage in graph execution order.
 *
 * @param birthIndex graph index where the tensor is produced
 * @param lastReadIndex last graph index that reads the storage owner
 * @param role memory role used for reuse policy
 * @param storageOwner tensor that owns the underlying storage
 */
public record NodeLifetime(
        int birthIndex,
        int lastReadIndex,
        MemoryRole role,
        Tensor storageOwner
) {
}
