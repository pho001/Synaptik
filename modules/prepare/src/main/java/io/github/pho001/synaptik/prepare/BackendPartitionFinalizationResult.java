package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import java.util.List;
import java.util.Objects;

/**
 * Carries one finalized executable and the persistent resources acquired for it atomically.
 *
 * <p>The resource list is an immutable snapshot in physical acquisition order. The finalizer
 * owns every listed resource until this value returns successfully from finalization; shared
 * Prepare owns them afterward until ownership is transferred to the completed prepared
 * execution. Entries are non-null, but duplicate identities are permitted at this value boundary
 * so the complete shared handoff can reject duplicates across all results consistently. This
 * value neither closes resources nor exposes backend-specific resource lookup.</p>
 *
 * @param executable exact non-null immutable executable to retain
 * @param resources non-null acquisition-ordered persistent resources to snapshot; entries must
 *     be non-null
 */
public record BackendPartitionFinalizationResult(
        PreparedExecutable executable,
        List<PreparedResource> resources) {
    /**
     * Validates the executable and snapshots the acquisition-ordered resource list.
     *
     * @param executable exact non-null executable to retain
     * @param resources non-null resource list to snapshot; entries must be non-null
     * @throws NullPointerException if the executable, list, or an indexed entry is null
     */
    public BackendPartitionFinalizationResult {
        Objects.requireNonNull(executable, "executable");
        Objects.requireNonNull(resources, "resources");
        for (int index = 0; index < resources.size(); index++) {
            Objects.requireNonNull(resources.get(index), "resources[" + index + "]");
        }
        resources = List.copyOf(resources);
    }

    /**
     * Creates a resource-free result retaining the supplied executable.
     *
     * @param executable exact non-null executable to retain
     * @throws NullPointerException if {@code executable} is null
     */
    public BackendPartitionFinalizationResult(PreparedExecutable executable) {
        this(executable, List.of());
    }
}
