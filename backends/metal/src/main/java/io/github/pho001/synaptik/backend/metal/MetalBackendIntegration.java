package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.Objects;

/**
 * Supported explicit lifecycle integration boundary for the Metal backend.
 *
 * <p>One integration owns one caller-configured native-library lookup and default-device context.
 * Opening validates a snapshotted {@link MetalBackendConfiguration}, loads the exact library,
 * validates its ABI, creates native ownership, and rolls back every partially opened native owner
 * before failing. The integration and its immutable collaborators may be used concurrently while
 * Engine coordinates their lifetime. It performs no library discovery, CPU fallback, backend
 * registration, or mixed-owner transfer.</p>
 */
public final class MetalBackendIntegration implements AutoCloseable {
    private final MetalBackendConfiguration configuration;
    private final MetalCapabilityProvider capabilityProvider;
    private final MetalBackendRuntime runtime;

    private MetalBackendIntegration(
            MetalBackendConfiguration configuration,
            MetalBackendRuntime runtime) {
        this.configuration = Objects.requireNonNull(configuration, "configuration");
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        this.capabilityProvider = new MetalCapabilityProvider();
    }

    /**
     * Opens one native Metal integration from an explicit backend-owned configuration.
     *
     * @param configuration non-null validated configuration; its immutable value is snapshotted
     * @return a new non-null open Metal integration owner
     * @throws NullPointerException if {@code configuration} is {@code null}
     * @throws IllegalArgumentException if its native library path is not absolute
     * @throws RuntimeException if library loading, ABI validation, or context creation fails
     * @throws Error if opening or partial-construction rollback reports a fatal failure
     */
    public static MetalBackendIntegration open(MetalBackendConfiguration configuration) {
        return open(configuration, MetalBackendRuntime::open, MetalBackendIntegration::new);
    }

    static MetalBackendIntegration open(
            MetalBackendConfiguration configuration,
            RuntimeOpener runtimeOpener,
            IntegrationPublisher integrationPublisher) {
        Objects.requireNonNull(configuration, "configuration");
        Objects.requireNonNull(runtimeOpener, "runtimeOpener");
        Objects.requireNonNull(integrationPublisher, "integrationPublisher");
        MetalBackendConfiguration snapshot =
                new MetalBackendConfiguration(configuration.nativeLibraryPath());
        MetalBackendRuntime runtime = runtimeOpener.open(snapshot.nativeLibraryPath());
        try {
            return Objects.requireNonNull(
                    integrationPublisher.publish(snapshot, runtime), "Metal integration");
        } catch (RuntimeException | Error failure) {
            suppressDistinct(failure, runtime::close);
            throw failure;
        }
    }

    /**
     * Returns the snapshotted configuration used to open this integration.
     *
     * @return the non-null immutable retained configuration
     */
    public MetalBackendConfiguration configuration() {
        return configuration;
    }

    /**
     * Returns the stable Metal capability provider.
     *
     * @return the same non-null immutable provider on every call
     */
    public MetalCapabilityProvider capabilityProvider() {
        return capabilityProvider;
    }

    /**
     * Returns the immutable availability captured when this integration opened.
     *
     * @return the same non-null snapshot containing the opened default Metal device
     */
    public BackendAvailabilitySnapshot availabilitySnapshot() {
        return runtime.availabilitySnapshot();
    }

    /**
     * Returns the retained Metal partition-preparation collaboration.
     *
     * @return a non-null shared Prepare role borrowing this integration lifetime
     * @throws IllegalStateException if native ownership is no longer available when used
     */
    public PartitionPreparation<?, ?> partitionPreparation() {
        return runtime.partitionPreparation();
    }

    /**
     * Returns the retained complete single-owner Metal schedule assembler.
     *
     * @return a non-null shared Prepare role borrowing this integration lifetime
     * @throws IllegalStateException if native ownership is no longer available when used
     */
    public PreparedScheduleAssembler scheduleAssembler() {
        return runtime.scheduleAssembler();
    }

    /**
     * Uploads caller-owned host storage into one Metal-owned borrowed-input representation.
     *
     * @param storage non-null live accessible FLOAT32 storage retained but never closed
     * @return a new non-null Metal representation whose ownership transfers to the caller
     * @throws NullPointerException if {@code storage} is {@code null}
     * @throws IllegalArgumentException if storage is not compatible with Metal ingress
     * @throws IllegalStateException if storage or this integration is closed or inaccessible
     * @throws RuntimeException if native allocation or upload fails
     * @throws Error if ingress or rollback reports a fatal failure
     */
    public BufferRepresentation borrow(HostTensorStorage storage) {
        return runtime.borrow(storage);
    }

    /**
     * Downloads one live Metal publication into detached canonical host bytes.
     *
     * @param representation non-null live representation owned by this integration
     * @param descriptor non-null exact publication descriptor
     * @param maximumBytes non-negative maximum canonical payload size
     * @return fresh non-null caller-owned row-major big-endian bytes
     * @throws NullPointerException if an object argument is {@code null}
     * @throws IllegalArgumentException if type, layout, representation, size, or limit is invalid
     * @throws IllegalStateException if the representation or integration is closed
     * @throws ArithmeticException if checked size arithmetic overflows
     * @throws RuntimeException if native download fails
     * @throws Error if copying reports a fatal failure
     */
    public byte[] copyToCanonicalHostBytes(
            BufferRepresentation representation,
            TensorDescriptor descriptor,
            long maximumBytes) {
        return runtime.copyToCanonicalHostBytes(representation, descriptor, maximumBytes);
    }

    /**
     * Closes native ownership after Engine has closed results and prepared handles.
     * Repeated calls are idempotent; context release may be deferred to a remaining child lease.
     *
     * @throws RuntimeException if native cleanup fails
     * @throws Error if cleanup reports a fatal failure
     */
    @Override
    public void close() {
        runtime.close();
    }

    @FunctionalInterface
    interface RuntimeOpener {
        MetalBackendRuntime open(java.nio.file.Path path);
    }

    @FunctionalInterface
    interface IntegrationPublisher {
        MetalBackendIntegration publish(
                MetalBackendConfiguration configuration, MetalBackendRuntime runtime);
    }

    private static void suppressDistinct(Throwable primary, Runnable cleanup) {
        try {
            cleanup.run();
        } catch (RuntimeException | Error failure) {
            if (failure != primary) primary.addSuppressed(failure);
        }
    }
}
