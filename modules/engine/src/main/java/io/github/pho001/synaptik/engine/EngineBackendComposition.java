package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.List;

/**
 * Supplies the exact inward collaborations owned by one advanced Engine composition.
 *
 * <p>This package-private, Engine-owned seam exists for the current CPU composition and
 * deterministic lifecycle tests. Implementations and their returned collaborations are safe for
 * concurrent Engine calls. It is not a public backend service-provider interface and does not
 * imply that several complete backend schedules can be assembled together.</p>
 */
interface EngineBackendComposition extends AutoCloseable {
    /**
     * Returns the immutable ordered providers used for every compilation.
     *
     * @return a non-null immutable ordered list with non-null elements; ownership remains with
     *     this composition
     */
    List<BackendCapabilityProvider> capabilityProviders();

    /**
     * Returns the immutable ordered availability facts aligned with the providers.
     *
     * @return a non-null immutable ordered list with non-null elements aligned with
     *     {@link #capabilityProviders()}; ownership remains with this composition
     */
    List<BackendAvailabilitySnapshot> availabilitySnapshots();

    /**
     * Prepares one graph using collaborations owned by this composition.
     *
     * @param artifacts non-null immutable artifacts produced for this Engine; not mutated
     * @return a non-null immutable reusable Runtime recipe
     * @throws RuntimeException if preparation rejects the artifacts or backend work fails
     * @throws Error if backend work reports a fatal failure
     */
    PreparedExecution prepare(CompileArtifacts artifacts);

    /**
     * Creates a non-owning representation of caller-owned host storage.
     *
     * @param storage non-null live caller-owned storage retained but never closed by the
     *     representation
     * @return a new non-null borrowed representation
     * @throws RuntimeException if the storage or composition is invalid for borrowing
     * @throws Error if backend work reports a fatal failure
     */
    BufferRepresentation borrow(HostTensorStorage storage);

    /**
     * Closes resources owned by this composition after all Engine results have closed.
     *
     * @throws RuntimeException if cleanup reports an unchecked failure
     * @throws Error if cleanup reports a fatal failure
     */
    @Override
    void close();
}
