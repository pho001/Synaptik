package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
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
     * Copies one exact publication representation into detached canonical host bytes.
     * The synchronous call performs no caching or ownership change. The caller keeps the result
     * lease open and prevents source mutation or closure until return; implementations add no
     * atomic-snapshot guarantee for racing mutation.
     *
     * @param representation non-null borrowed representation kept open and owned by its result
     *     lease; not retained, mutated, transferred, or closed by this call
     * @param descriptor non-null exact fully static resolved publication descriptor
     * @param maximumBytes non-negative caller limit for the canonical payload
     * @return fresh non-null caller-owned canonical row-major big-endian bytes
     * @throws NullPointerException if {@code representation} or {@code descriptor} is null
     * @throws IllegalArgumentException if the limit, descriptor, representation, carrier,
     *     capacity, or BOOL encoding is unsupported or inconsistent
     * @throws IllegalStateException if composition or representation closure has begun or the
     *     representation is inaccessible to the calling thread
     * @throws ArithmeticException if checked count or address arithmetic overflows
     * @throws OutOfMemoryError if the otherwise valid fresh byte array cannot be allocated
     * @throws RuntimeException if physical copying reports another unchecked failure
     * @throws Error if copying reports a fatal failure
     */
    byte[] copyToCanonicalHostBytes(
            BufferRepresentation representation,
            TensorDescriptor descriptor,
            long maximumBytes);

    /**
     * Closes resources owned by this composition after all Engine results have closed.
     *
     * @throws RuntimeException if cleanup reports an unchecked failure
     * @throws Error if cleanup reports a fatal failure
     */
    @Override
    void close();
}
