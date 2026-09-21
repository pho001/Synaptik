package io.github.pho001.synaptik.runtime.resource;

/**
 * Marks an immutable physical resource retained for the lifetime of one prepared execution.
 *
 * <p>Concrete backends implement this nominal contract and own every physical release detail.
 * Runtime retains only the resource identity and invokes {@link #close()} once when the owning
 * prepared execution can no longer admit or serve a run. This contract exposes no physical
 * storage, backend value, lookup key, or access operation.
 *
 * <p>Implementations may report cleanup failure with an unchecked exception or error. They must
 * not require Runtime to identify the concrete backend or physical resource type.
 */
public interface PreparedResource extends AutoCloseable {
    /**
     * Releases the backend-owned physical resource.
     *
     * <p>The owning prepared execution invokes this method at most once, in deterministic reverse
     * acquisition order, after all admitted synchronous runs have released their leases.
     *
     * @throws RuntimeException if physical cleanup reports an unchecked failure
     * @throws Error if physical cleanup reports an error
     */
    @Override
    void close();
}
