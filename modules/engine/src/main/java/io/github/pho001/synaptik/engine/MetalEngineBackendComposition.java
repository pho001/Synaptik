package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
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
import java.util.Objects;

/** Adapts one Engine-owned Metal integration without exposing Engine types to the backend. */
final class MetalEngineBackendComposition implements EngineBackendComposition {
    private final MetalBackendIntegration integration;
    private final List<BackendCapabilityProvider> capabilityProviders;
    private final List<BackendAvailabilitySnapshot> availabilitySnapshots;

    /**
     * Captures one immutable provider and availability snapshot from the transferred integration.
     *
     * @param integration non-null open integration whose ownership already transferred to Engine
     * @throws NullPointerException if {@code integration} is {@code null}
     */
    MetalEngineBackendComposition(MetalBackendIntegration integration) {
        this.integration = Objects.requireNonNull(integration, "integration");
        this.capabilityProviders = List.of(integration.capabilityProvider());
        this.availabilitySnapshots = List.of(integration.availabilitySnapshot());
    }

    /** @return the exact retained Metal integration for direct prepared transfer binding */
    MetalBackendIntegration integration() {
        return integration;
    }

    /** {@inheritDoc} */
    @Override
    public List<BackendCapabilityProvider> capabilityProviders() {
        return capabilityProviders;
    }

    /** {@inheritDoc} */
    @Override
    public List<BackendAvailabilitySnapshot> availabilitySnapshots() {
        return availabilitySnapshots;
    }

    /** {@inheritDoc} */
    @Override
    public PartitionPreparation<?, ?> partitionPreparation() {
        return integration.partitionPreparation();
    }

    /** {@inheritDoc} */
    @Override
    public PreparedScheduleContributor scheduleContributor() {
        return integration.scheduleContributor();
    }

    /** {@inheritDoc} */
    @Override
    public PreparedScheduleAssembler scheduleAssembler() {
        return integration.scheduleAssembler();
    }

    /** {@inheritDoc} */
    @Override
    public boolean supportsTransferTo(
            EngineBackendComposition destination, TensorDescriptor descriptor) {
        Objects.requireNonNull(destination, "destination");
        return destination instanceof CpuEngineBackendComposition
                && CpuMetalPreparedBufferTransfer.supports(descriptor);
    }

    /** {@inheritDoc} */
    @Override
    public PreparedBufferTransfer prepareTransferTo(
            EngineBackendComposition destination,
            PreparedMemoryPlan memoryPlan,
            int bufferIndex,
            int sourceRepresentationIndex,
            int destinationRepresentationIndex,
            TensorDescriptor descriptor) {
        if (!(destination instanceof CpuEngineBackendComposition cpu)) {
            throw new IllegalArgumentException(
                    "Metal transfer destination is not the registered CPU adapter");
        }
        return CpuMetalPreparedBufferTransfer.metalToCpu(
                cpu.integration(),
                integration,
                memoryPlan,
                bufferIndex,
                sourceRepresentationIndex,
                destinationRepresentationIndex,
                descriptor);
    }

    /** {@inheritDoc} */
    @Override
    public BufferRepresentation borrow(HostTensorStorage storage) {
        return integration.borrow(storage);
    }

    /** {@inheritDoc} */
    @Override
    public byte[] copyToCanonicalHostBytes(
            BufferRepresentation representation,
            TensorDescriptor descriptor,
            long maximumBytes) {
        return integration.copyToCanonicalHostBytes(
                representation, descriptor, maximumBytes);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        integration.close();
    }
}
