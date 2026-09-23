package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContributor;
import io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.List;

/**
 * Supplies exact inward collaborations owned by one registered Engine backend.
 *
 * <p>This package-private, Engine-owned seam has current CPU and Metal implementations for the
 * ordinary explicit builder and deterministic lifecycle tests. Implementations and returned
 * collaborations are safe for concurrent Engine calls. It is not a public backend
 * service-provider interface. Shared Engine/Prepare composition combines positional partition
 * preparations and physical creation contributions into one complete schedule.</p>
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
     * Returns one fresh positional preparation for a partition owned by this backend.
     *
     * @return non-null immutable preparation whose collaborations borrow this adapter lifetime
     * @throws IllegalStateException if the underlying integration is closed
     */
    PartitionPreparation<?, ?> partitionPreparation();

    /**
     * Returns the retained physical representation-creation contributor.
     *
     * @return non-null immutable contributor with this adapter's exact backend identity
     * @throws IllegalStateException if the underlying integration is closed
     */
    PreparedScheduleContributor scheduleContributor();

    /**
     * Returns the backend's retained complete assembler for producerless constant geometry.
     *
     * @return non-null immutable assembler borrowing this adapter lifetime
     * @throws IllegalStateException if the underlying integration is closed
     */
    PreparedScheduleAssembler scheduleAssembler();

    /**
     * Reports whether this exact source adapter supports transfer to a direct destination adapter
     * for one logical descriptor.
     *
     * @param destination non-null exact registered destination adapter
     * @param descriptor non-null exact logical descriptor
     * @return whether the ordered path and descriptor are supported
     */
    boolean supportsTransferTo(
            EngineBackendComposition destination, TensorDescriptor descriptor);

    /**
     * Creates one immutable direct transfer recipe after successful capability preflight.
     *
     * @param destination non-null exact destination adapter
     * @param memoryPlan exact non-null shared memory plan
     * @param bufferIndex dense buffer position
     * @param sourceRepresentationIndex source owner representation position
     * @param destinationRepresentationIndex destination owner representation position
     * @param descriptor exact non-null logical descriptor
     * @return non-null immutable transfer recipe retaining no physical run resource
     * @throws IllegalArgumentException if path, descriptor, or coordinates are unsupported
     */
    PreparedBufferTransfer prepareTransferTo(
            EngineBackendComposition destination,
            PreparedMemoryPlan memoryPlan,
            int bufferIndex,
            int sourceRepresentationIndex,
            int destinationRepresentationIndex,
            TensorDescriptor descriptor);

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
     * Closes resources owned by this composition after all Engine results and retained prepared
     * handles have closed.
     *
     * @throws RuntimeException if cleanup reports an unchecked failure
     * @throws Error if cleanup reports a fatal failure
     */
    @Override
    void close();
}
