package io.github.pho001.synaptik.runtime.resource;

/**
 * Identifies the lifecycle role of one concrete backend-owned workspace representation.
 *
 * <p>A workspace is per-run backend-local scratch rather than a logical graph value or a
 * transferable result. A concrete backend implements this nominal role with its own physical
 * scratch type. The shared Runtime layer may retain the representation and request cleanup, but
 * this interface deliberately exposes no storage, access, backend, device, transfer, validity,
 * or residency operation.
 *
 * <p>Every workspace representation supplied to a successfully constructed {@code RunState} is
 * run-owned. The run requests cleanup when it closes, while the concrete implementation owns the
 * physical release mechanics and may report cleanup failure with an unchecked exception or
 * error.
 */
public interface WorkspaceRepresentation extends AutoCloseable {
    /**
     * Releases the physical workspace resources owned by this representation.
     *
     * <p>The method declares no checked failure. Thread-safety and the effect of invoking the
     * implementation directly more than once are properties of the concrete backend type;
     * {@code RunState} invokes a successfully transferred workspace at most once.
     *
     * @throws RuntimeException if the concrete backend reports an unchecked cleanup failure
     * @throws Error if the concrete backend reports a cleanup error
     */
    @Override
    void close();
}
