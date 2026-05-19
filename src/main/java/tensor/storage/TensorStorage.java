package tensor.storage;

import tensor.DataType;
import tensor.Tensor;

/**
 * Raw storage backing a {@link Tensor}.
 *
 * <p>Implementations expose typed arrays directly for performance. The arrays
 * are mutable and are not synchronized; callers that share a storage instance
 * across threads must provide their own external synchronization. Mutating data
 * through a typed array accessor does not automatically increment the version,
 * so callers must invoke {@link #markModified()} when bypassing tensor setters.</p>
 */
public interface TensorStorage {
    /**
     * Returns the element type stored by this buffer.
     *
     * @return non-null dtype associated with the storage implementation
     */
    DataType getType();

    /**
     * Returns the number of physical elements in this storage buffer.
     *
     * @return non-negative element count
     */
    int getSize();

    /**
     * Returns a monotonically increasing mutation counter.
     *
     * @return current storage version, incremented by {@link #markModified()}
     */
    long version();

    /**
     * Records an in-place mutation of this storage.
     *
     * <p>This method does not change any element value by itself. It exists so
     * view/cache users can detect writes made through tensor setters or direct
     * array access.</p>
     */
    void markModified();
}
