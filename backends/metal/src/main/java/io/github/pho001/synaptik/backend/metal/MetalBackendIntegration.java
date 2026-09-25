package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContributor;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Supported explicit lifecycle integration boundary for the Metal backend.
 *
 * <p>One integration owns one caller-configured native-library lookup and default-device context.
 * Opening validates a snapshotted {@link MetalBackendConfiguration}, loads the exact library,
 * validates its ABI, creates native ownership, and rolls back every partially opened native owner
 * before failing. The integration and its immutable collaborators may be used concurrently while
 * Engine coordinates their lifetime. It performs no library discovery, CPU fallback, backend
 * registration, or transfer scheduling. It exposes a physical schedule contributor and checked
 * native upload/download binders that Engine uses for the bounded mixed-owner CPU/Metal transfer
 * domain.</p>
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

    /**
     * Opens one native Metal integration with a caller-owned typed diagnostic observer.
     *
     * <p>The observer is retained but never closed. Its callback may be invoked concurrently.
     * Callback {@link RuntimeException RuntimeExceptions} disable later tracing without changing
     * backend work or outcomes. Callback {@link Error Errors} propagate from success reporting;
     * during failure reporting, the backend failure remains primary and receives a distinct
     * acyclic callback error as a suppressed failure.</p>
     *
     * @param configuration non-null validated configuration; its immutable value is snapshotted
     * @param observer non-null caller-owned thread-safe observer retained for this integration
     * @return a new non-null open traced Metal integration owner
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the native library path is not absolute
     * @throws RuntimeException if library loading, ABI validation, or context creation fails
     * @throws Error if opening, partial-construction rollback, or successful-outcome observation
     *     reports a fatal error
     */
    public static MetalBackendIntegration open(
            MetalBackendConfiguration configuration, MetalTraceObserver observer) {
        Objects.requireNonNull(configuration, "configuration");
        Objects.requireNonNull(observer, "observer");
        MetalTraceProducer traceProducer = new MetalTraceProducer(observer);
        return open(
                configuration,
                path -> MetalBackendRuntime.open(path, traceProducer),
                MetalBackendIntegration::new);
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
     * Returns the retained backend-local complete Metal schedule assembler.
     *
     * @return a non-null shared Prepare role borrowing this integration lifetime
     * @throws IllegalStateException if native ownership is no longer available when used
     */
    public PreparedScheduleAssembler scheduleAssembler() {
        return runtime.scheduleAssembler();
    }

    /**
     * Returns the retained Metal physical-creation contributor used by shared mixed-owner
     * assembly.
     *
     * @return non-null retained immutable contributor; ownership remains with this integration
     * @throws IllegalStateException if native ownership is no longer available when used
     */
    public PreparedScheduleContributor scheduleContributor() {
        return runtime.scheduleContributor();
    }

    /**
     * Uploads caller-owned FLOAT32 or INT32 host storage into one Metal-owned borrowed-input
     * representation without conversion.
     *
     * @param storage non-null live accessible FLOAT32 or INT32 storage retained but never closed
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
     * Reports whether a nominal representation is the exact live Metal side of a prepared
     * contiguous FLOAT32 transfer.
     *
     * @param representation non-null candidate Metal representation
     * @param descriptor non-null exact logical descriptor
     * @return whether representation type, context, layout, data type, and byte extent match
     * @throws NullPointerException if an object argument is null
     */
    public boolean acceptsContiguousFloat32Transfer(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        return runtime.acceptsContiguousFloat32Transfer(representation, descriptor);
    }

    /**
     * Cold-binds one exact Metal destination to a direct native upload action.
     *
     * <p>The returned action retains the typed destination directly. Invoking it performs exactly
     * one upload from the supplied live native host segment and allocates no staging storage.</p>
     *
     * @param representation non-null exact destination representation
     * @param descriptor non-null exact static canonical contiguous FLOAT32 descriptor
     * @return non-null immutable action retaining the typed destination and checked byte extent
     * @throws NullPointerException if an object argument is null
     * @throws IllegalArgumentException if type, context, layout, or byte extent is incompatible
     */
    public Consumer<MemorySegment> bindContiguousFloat32Upload(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        return runtime.bindContiguousFloat32Upload(representation, descriptor);
    }

    /**
     * Cold-binds one exact Metal source to a direct native download action.
     *
     * <p>The returned action retains the typed source directly. Invoking it performs exactly one
     * download into the supplied live writable native host segment and allocates no staging
     * storage.</p>
     *
     * @param representation non-null exact source representation
     * @param descriptor non-null exact static canonical contiguous FLOAT32 descriptor
     * @return non-null immutable action retaining the typed source and checked byte extent
     * @throws NullPointerException if an object argument is null
     * @throws IllegalArgumentException if type, context, layout, or byte extent is incompatible
     */
    public Consumer<MemorySegment> bindContiguousFloat32Download(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        return runtime.bindContiguousFloat32Download(representation, descriptor);
    }

    /**
     * Downloads one live Metal publication into detached canonical host bytes.
     *
     * <p>The descriptor is an ordinary canonical non-view FLOAT32 or BOOL publication, including
     * a locally produced rank-zero FLOAT32 reduction result, or an exact positive-rank logical
     * affine FLOAT32 view whose representation carries finalized-route authentication for a full
     * dense represented-order target. A rank-zero FLOAT32 result yields exactly four big-endian
     * bytes; BOOL retains exact row-major one-byte zero-or-one elements. This local BOOL
     * publication does not add BOOL ingress or a general BOOL consumer, and the path does not
     * widen the positive-rank-only CPU/Metal FLOAT32 transfer predicate.</p>
     *
     * @param representation non-null live representation owned by this integration
     * @param descriptor non-null exact canonical FLOAT32/BOOL or authenticated affine FLOAT32
     *     publication descriptor
     * @param maximumBytes non-negative maximum canonical payload size
     * @return fresh non-null caller-owned row-major canonical bytes
     * @throws NullPointerException if an object argument is {@code null}
     * @throws IllegalArgumentException if type, layout, authentication, representation, size, or
     *     limit is invalid
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
