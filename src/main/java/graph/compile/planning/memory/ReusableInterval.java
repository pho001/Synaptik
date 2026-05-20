package graph.compile.planning.memory;

import tensor.DataType;
import tensor.Tensor;

/**
 * Reusable storage interval for a tensor storage owner.
 *
 * @param owner tensor that owns storage for the interval
 * @param birthIndex graph index where the owner is produced
 * @param lastReadIndex last graph index that reads the owner
 * @param size element count required by the interval
 * @param dataType dtype stored in the interval
 * @param role memory role used for slot compatibility
 */
public record ReusableInterval(
        Tensor owner,
        int birthIndex,
        int lastReadIndex,
        int size,
        DataType dataType,
        MemoryRole role
) {
}
