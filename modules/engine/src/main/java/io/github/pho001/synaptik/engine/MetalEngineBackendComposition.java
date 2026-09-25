package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalCompletePlanTuning;
import io.github.pho001.synaptik.backend.metal.MetalLocalWorkloadTuning;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContributor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
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

    /** @return the retained Metal-owned local-workload tuning collaboration */
    MetalLocalWorkloadTuning localWorkloadTuning() {
        return integration.localWorkloadTuning();
    }

    /** @return the retained Metal-owned complete-plan tuning collaboration */
    MetalCompletePlanTuning completePlanTuning() {
        return integration.completePlanTuning();
    }

    /**
     * Obtains the sole authoritative stable Prepare projection for Metal tuning.
     *
     * @param artifacts exact non-null Engine-owned compile result; inspected but not transferred
     * @return a new non-null validated context for the sole Metal partition
     * @throws NullPointerException if {@code artifacts} is {@code null}
     * @throws IllegalArgumentException if partition coverage is not exactly one
     * @throws IllegalStateException if the Metal integration is closed
     */
    PrepareContext<?> projectedContext(CompileArtifacts artifacts) {
        Objects.requireNonNull(artifacts, "artifacts");
        if (artifacts.partitions().size() != 1) {
            throw new IllegalArgumentException(
                    "Metal integration requires exactly one non-empty Metal partition");
        }
        PartitionPreparation<?, ?> preparation = integration.partitionPreparation();
        return GraphPreparation.project(
                artifacts, artifacts.partitions().getFirst(), preparation.backendInputs());
    }

    /**
     * Composes one Metal-owned partition preparation through complete shared graph preparation.
     *
     * @param artifacts exact non-null Engine-owned compile result; inspected synchronously
     * @param preparation exact non-null Metal-owned positional preparation
     * @return one non-null complete immutable prepared execution owned by the caller
     */
    PreparedExecution prepare(
            CompileArtifacts artifacts, PartitionPreparation<?, ?> preparation) {
        return GraphPreparation.prepare(
                artifacts, List.of(preparation), integration.scheduleAssembler());
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
