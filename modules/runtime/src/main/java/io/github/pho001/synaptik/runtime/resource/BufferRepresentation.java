package io.github.pho001.synaptik.runtime.resource;

/**
 * Identifies the lifecycle role of one concrete backend-owned buffer representation.
 *
 * <p>A concrete backend implements this nominal role with its own physical storage type. The
 * shared Runtime layer may retain the representation and request cleanup, but this interface
 * deliberately exposes no storage, access, backend, device, transfer, validity, or residency
 * operation. Different implementations need not share a physical representation model.
 *
 * <p>When a representation is supplied to a run as run-owned, successful {@code RunState}
 * construction transfers cleanup responsibility to that run. A borrowed representation remains
 * caller-owned and is never closed by the run. Implementations define the physical cleanup
 * mechanics and may report cleanup failure with an unchecked exception or error.
 */
public interface BufferRepresentation extends AutoCloseable {
    /**
     * Releases the physical buffer resources owned by this representation.
     *
     * <p>The method declares no checked failure. Thread-safety and the effect of invoking the
     * implementation directly more than once are properties of the concrete backend type;
     * {@code RunState} invokes each representation it still owns at most once.
     *
     * @throws RuntimeException if the concrete backend reports an unchecked cleanup failure
     * @throws Error if the concrete backend reports a cleanup error
     */
    @Override
    void close();
}
