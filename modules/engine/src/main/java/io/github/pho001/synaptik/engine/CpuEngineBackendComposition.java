package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuCompletePlanTuning;
import io.github.pho001.synaptik.backend.cpu.CpuLocalWorkloadTuning;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.List;
import java.util.Objects;

/**
 * Adapts the one currently complete CPU lifecycle integration to Engine orchestration.
 *
 * <p>The adapter retains the exact CPU integration whose ownership was transferred to the
 * Engine. It delegates lowering, representation work, and complete one-partition schedule
 * assembly to CPU and shared Prepare without importing backend internals or deriving CPU
 * constant roles, descriptors, byte geometry, or initialization facts. Closing this composition
 * closes that integration; no other method transfers its ownership.</p>
 */
final class CpuEngineBackendComposition implements EngineBackendComposition {
    private final CpuBackendIntegration integration;
    private final List<BackendCapabilityProvider> capabilityProviders;
    private final List<BackendAvailabilitySnapshot> availabilitySnapshots;

    /**
     * Creates the package-private composition over one owned CPU adapter.
     *
     * @param integration non-null open CPU integration whose sole cleanup ownership has
     *     transferred to Engine
     * @throws NullPointerException if {@code integration} is null
     */
    CpuEngineBackendComposition(CpuBackendIntegration integration) {
        this.integration = Objects.requireNonNull(integration, "integration");
        capabilityProviders = List.of(integration.capabilityProvider());
        availabilitySnapshots = List.of(integration.availabilitySnapshot());
    }

    /**
     * Returns the retained CPU-owned local-workload tuning collaboration.
     *
     * @return the exact non-null collaboration owned by the retained CPU integration
     * @throws RuntimeException if the CPU integration cannot expose its cold collaboration
     * @throws Error if acquisition reports a fatal failure
     */
    CpuLocalWorkloadTuning localWorkloadTuning() {
        return integration.localWorkloadTuning();
    }

    /**
     * Returns the retained CPU-owned complete-plan tuning collaboration.
     *
     * @return the exact non-null collaboration owned by the retained CPU integration
     * @throws RuntimeException if the CPU integration cannot expose its cold collaboration
     * @throws Error if acquisition reports a fatal failure
     */
    CpuCompletePlanTuning completePlanTuning() {
        return integration.completePlanTuning();
    }

    /**
     * Obtains the sole authoritative stable Prepare projection for CPU tuning.
     *
     * @param artifacts exact non-null Engine-owned compile result; inspected but not transferred
     * @return a new non-null validated context for the sole CPU partition
     * @throws NullPointerException if {@code artifacts} is {@code null}
     * @throws IllegalArgumentException if partition coverage is not exactly one
     * @throws IllegalStateException if the CPU integration is closed
     */
    PrepareContext<?> projectedContext(CompileArtifacts artifacts) {
        Objects.requireNonNull(artifacts, "artifacts");
        if (artifacts.partitions().size() != 1) {
            throw new IllegalArgumentException(
                    "CPU integration requires exactly one non-empty CPU partition");
        }
        PartitionPreparation<?, ?> preparation = integration.partitionPreparation();
        return GraphPreparation.project(artifacts, artifacts.partitions().getFirst(),
                preparation.backendInputs());
    }

    /**
     * Composes one CPU-owned partition preparation through complete shared graph preparation.
     *
     * @param artifacts exact non-null Engine-owned compile result; inspected synchronously
     * @param preparation exact non-null CPU-owned positional preparation; ownership is unchanged
     * @return one non-null complete immutable prepared execution owned by the caller
     * @throws NullPointerException if an argument or backend result is {@code null}
     * @throws IllegalArgumentException if the preparation or returned schedule disagrees with
     *     the compile projection
     * @throws IllegalStateException if the CPU integration is closed
     */
    PreparedExecution prepare(
            CompileArtifacts artifacts, PartitionPreparation<?, ?> preparation) {
        return GraphPreparation.prepare(artifacts, List.of(preparation),
                integration.scheduleAssembler());
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
    public PreparedExecution prepare(CompileArtifacts artifacts) {
        return prepare(artifacts, integration.partitionPreparation());
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
        return integration.copyToCanonicalHostBytes(representation, descriptor, maximumBytes);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        integration.close();
    }
}
