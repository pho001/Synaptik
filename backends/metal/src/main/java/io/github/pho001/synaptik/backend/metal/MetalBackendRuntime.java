package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContributor;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/** Package-private native-lifecycle owner used only by {@link MetalBackendIntegration}. */
final class MetalBackendRuntime implements AutoCloseable {
    private static final ValueLayout.OfByte BYTE = ValueLayout.JAVA_BYTE;

    private final MetalDeviceContext context;
    private final BackendAvailabilitySnapshot availabilitySnapshot;
    private final PartitionPreparation<?, ?> partitionPreparation;
    private final PreparedScheduleAssembler scheduleAssembler;
    private final MetalLocalWorkloadTuning localWorkloadTuning;
    private final MetalCompletePlanTuning completePlanTuning;

    MetalBackendRuntime(MetalDeviceContext context) {
        this(context, null);
    }

    private MetalBackendRuntime(
            MetalDeviceContext context, MetalTraceProducer traceProducer) {
        this.context = Objects.requireNonNull(context, "context");
        var backendId = MetalCapabilityProvider.METAL_BACKEND_ID;
        this.availabilitySnapshot = new BackendAvailabilitySnapshot(
                backendId,
                Map.of(new BackendDeviceId(backendId, "default"), DeviceClass.ACCELERATOR));
        this.partitionPreparation = new PartitionPreparation<>(
                new MetalNegAnalysisInputs(context, traceProducer),
                new MetalNegPartitionPreparer(),
                new MetalNegPartitionFinalizer(context));
        this.scheduleAssembler = new MetalNegPreparedScheduleAssembler(context);
        this.localWorkloadTuning = new MetalLocalWorkloadTuning(context, traceProducer);
        this.completePlanTuning = new MetalCompletePlanTuning(localWorkloadTuning);
    }

    /**
     * Opens the exact caller-selected native bridge and its default-device context.
     *
     * @param absoluteLibraryPath non-null normalized absolute native bridge path
     * @return a new non-null runtime owner
     * @throws NullPointerException if {@code absoluteLibraryPath} is {@code null}
     * @throws IllegalArgumentException if the path is not absolute
     * @throws RuntimeException if native loading, ABI validation, or context creation fails
     * @throws Error if opening or rollback reports a fatal failure
     */
    static MetalBackendRuntime open(Path absoluteLibraryPath) {
        MetalDeviceContext context = MetalDeviceContext.open(absoluteLibraryPath);
        return takeOwnership(context, MetalBackendRuntime::new);
    }

    /**
     * Opens the exact caller-selected native bridge with one retained trace producer.
     *
     * @param absoluteLibraryPath non-null normalized absolute native bridge path
     * @param traceProducer non-null producer retained by preparation plans and executables
     * @return a new non-null traced runtime owner
     */
    static MetalBackendRuntime open(
            Path absoluteLibraryPath, MetalTraceProducer traceProducer) {
        Objects.requireNonNull(traceProducer, "traceProducer");
        MetalDeviceContext context = MetalDeviceContext.open(absoluteLibraryPath);
        return takeOwnership(
                context, acquired -> new MetalBackendRuntime(acquired, traceProducer));
    }

    static MetalBackendRuntime takeOwnership(
            MetalDeviceContext context, RuntimePublisher publisher) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(publisher, "publisher");
        try {
            return Objects.requireNonNull(publisher.publish(context), "Metal backend runtime");
        } catch (RuntimeException | Error failure) {
            suppressDistinct(failure, context::close);
            throw failure;
        }
    }

    /**
     * Returns the point-in-time availability captured after native context creation.
     *
     * @return the immutable snapshot captured immediately after native context creation
     */
    BackendAvailabilitySnapshot availabilitySnapshot() {
        return availabilitySnapshot;
    }

    /**
     * Returns the retained Metal partition-preparation collaboration.
     *
     * @return the retained immutable Metal partition-preparation collaboration
     */
    PartitionPreparation<?, ?> partitionPreparation() {
        return partitionPreparation;
    }

    /**
     * Returns the retained Metal local-workload tuning collaboration.
     *
     * @return the retained Metal local-workload tuning collaboration
     */
    MetalLocalWorkloadTuning localWorkloadTuning() {
        return localWorkloadTuning;
    }

    /**
     * Returns the retained Metal complete-plan tuning collaboration.
     *
     * @return the retained Metal complete-plan tuning collaboration
     */
    MetalCompletePlanTuning completePlanTuning() {
        return completePlanTuning;
    }

    /**
     * Returns the retained complete Metal schedule assembler.
     *
     * @return the retained immutable complete Metal schedule assembler
     */
    PreparedScheduleAssembler scheduleAssembler() {
        return scheduleAssembler;
    }

    /** @return the retained physical-creation contributor used by shared mixed-owner assembly */
    PreparedScheduleContributor scheduleContributor() {
        return (PreparedScheduleContributor) scheduleAssembler;
    }

    /**
     * Creates one Metal-owned input representation from caller-owned host storage.
     *
     * <p>The upload is complete before return. All six model data types transfer their exact
     * native represented bytes. BOOL is validated as zero-or-one before native mutation. The
     * representation retains the storage as a non-owning borrow and owns only its Metal buffer.</p>
     *
     * @param storage non-null live accessible host storage; never closed here
     * @return a new non-null Metal buffer representation owned by the caller
     * @throws NullPointerException if {@code storage} is {@code null}
     * @throws IllegalArgumentException if BOOL storage contains a byte other than zero or one
     * @throws IllegalStateException if storage or the Metal context is closed or inaccessible
     * @throws RuntimeException if native allocation or upload fails
     * @throws Error if allocation, upload, or rollback reports a fatal failure
     */
    BufferRepresentation borrow(HostTensorStorage storage) {
        Objects.requireNonNull(storage, "storage");
        MemorySegment source = storage.segment();
        if (!storage.isAlive() || !source.isAccessibleBy(Thread.currentThread())) {
            throw new IllegalStateException("host storage is not alive and accessible");
        }
        if (storage.dataType() == DataType.BOOL) {
            validateBooleanBytes(source, storage.byteSize());
        }
        MetalBufferRepresentation buffer =
                context.createBorrowedBuffer(storage.byteSize(), storage);
        try {
            if (source.isNative()) {
                buffer.upload(0L, source, 0L, storage.byteSize());
            } else {
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment staging = arena.allocate(storage.byteSize(), 1L);
                    staging.copyFrom(source);
                    buffer.upload(0L, staging, 0L, storage.byteSize());
                }
            }
            return buffer;
        } catch (RuntimeException | Error failure) {
            try {
                buffer.close();
            } catch (RuntimeException | Error cleanupFailure) {
                if (cleanupFailure != failure) failure.addSuppressed(cleanupFailure);
            }
            throw failure;
        }
    }

    /**
     * Reports whether one nominal representation is an exact live Metal side of a prepared
     * canonical transfer for any model data type.
     *
     * @param representation non-null candidate representation
     * @param descriptor non-null exact logical descriptor
     * @return whether representation type, context, layout, data type, and byte extent match
     */
    boolean acceptsCanonicalTransfer(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        Objects.requireNonNull(representation, "representation");
        Objects.requireNonNull(descriptor, "descriptor");
        if (!isCanonicalTransfer(descriptor)) {
            return false;
        }
        long byteCount = transferByteCount(descriptor);
        MetalBufferRepresentation metal =
                MetalPreparedSplatResource.readableBuffer(representation);
        return metal != null
                && metal.belongsTo(context)
                && metal.byteSize() == byteCount;
    }

    /**
     * Cold-binds one exact Metal destination to a direct canonical upload action.
     *
     * @param representation non-null candidate destination
     * @param descriptor non-null exact logical descriptor
     * @return non-null immutable action capturing the exact typed destination
     */
    Consumer<MemorySegment> bindCanonicalUpload(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        if (!(representation instanceof MetalBufferRepresentation metal)
                || !acceptsCanonicalTransfer(representation, descriptor)) {
            throw new IllegalArgumentException(
                    "Metal upload requires an exact writable canonical representation");
        }
        long byteCount = transferByteCount(descriptor);
        return source -> {
            if (descriptor.dataType() == DataType.BOOL) {
                validateBooleanBytes(source, byteCount);
            }
            metal.upload(0L, source, 0L, byteCount);
        };
    }

    /**
     * Cold-binds one exact Metal source to a direct canonical download action.
     *
     * @param representation non-null candidate source
     * @param descriptor non-null exact logical descriptor
     * @return non-null immutable action capturing the exact typed source
     */
    Consumer<MemorySegment> bindCanonicalDownload(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        if (!acceptsCanonicalTransfer(representation, descriptor)) {
            throw new IllegalArgumentException(
                    "Metal download requires an exact live canonical representation");
        }
        MetalBufferRepresentation metal =
                MetalPreparedSplatResource.readableBuffer(representation);
        long byteCount = transferByteCount(descriptor);
        return destination -> metal.download(0L, destination, 0L, byteCount);
    }

    /**
     * Downloads one canonical publication of any model data type into row-major big-endian bytes.
     *
     * <p>Canonical non-view rank-0..16 publications support all six model data types. BOOL bytes
     * must be zero or one. The existing authenticated positive-rank affine publication remains
     * FLOAT32-only. The logical descriptor is validated but never rewritten.</p>
     *
     * @param representation non-null live representation owned by this context
     * @param descriptor non-null exact canonical descriptor or authenticated affine FLOAT32
     * @param maximumBytes non-negative caller byte ceiling
     * @return fresh non-null canonical bytes
     * @throws NullPointerException if an object argument is {@code null}
     * @throws IllegalArgumentException if type, layout, size, representation, BOOL value, or limit
     *     is invalid
     * @throws IllegalStateException if the representation or context is closed
     * @throws ArithmeticException if checked size arithmetic overflows
     * @throws RuntimeException if native download fails
     */
    byte[] copyToCanonicalHostBytes(
            BufferRepresentation representation,
            TensorDescriptor descriptor,
            long maximumBytes) {
        Objects.requireNonNull(representation, "representation");
        Objects.requireNonNull(descriptor, "descriptor");
        if (maximumBytes < 0L) {
            throw new IllegalArgumentException("maximumBytes must be non-negative");
        }
        MetalBufferRepresentation metal =
                MetalPreparedSplatResource.readableBuffer(representation);
        if (metal == null || !metal.belongsTo(context)) {
            throw new IllegalArgumentException(
                    "representation must be a live readable buffer owned by this Metal integration");
        }
        LayoutDescriptor layout = descriptor.layout().orElse(null);
        int rank = descriptor.shape().rank();
        DataType dataType = descriptor.dataType();
        boolean canonical = rank >= 0
                && descriptor.shape().isFullyStatic()
                && rank <= 16
                && layout != null
                && layout.kind() == LayoutKind.DENSE_CONTIGUOUS
                && !layout.isView()
                && layout.storageOffset() == 0L;
        boolean authenticatedAffine = dataType == DataType.FLOAT32
                && descriptor.shape().isFullyStatic()
                && descriptor.shape().rank() >= 1
                && descriptor.shape().rank() <= 16
                && layout != null
                && layout.isView()
                && layout.storageOffset() == 0L
                && metal.authenticatesDenseAffinePublication(descriptor);
        if (!canonical && !authenticatedAffine) {
            throw new IllegalArgumentException(
                    "Metal materialization requires a canonical rank-0..16 descriptor"
                            + " or authenticated affine FLOAT32");
        }
        long elements = 1L;
        for (long dimension : descriptor.shape().toLongArray()) {
            if (dimension <= 0L) {
                throw new IllegalArgumentException(
                        "descriptor dimensions are outside the materialization domain");
            }
            elements = Math.multiplyExact(elements, dimension);
        }
        long byteCount = Math.multiplyExact(elements, dataType.byteWidth());
        if (byteCount > maximumBytes) {
            throw new IllegalArgumentException("canonical payload exceeds maximumBytes");
        }
        int length = Math.toIntExact(byteCount);
        if (metal.byteSize() < byteCount) {
            throw new IllegalArgumentException("Metal representation is smaller than descriptor");
        }
        byte[] result = new byte[length];
        try (Arena arena = Arena.ofConfined()) {
            int width = dataType.byteWidth();
            MemorySegment staging = arena.allocate(byteCount, width);
            metal.download(0L, staging, 0L, byteCount);
            MemorySegment destination = MemorySegment.ofArray(result);
            if (dataType == DataType.BOOL) {
                validateBooleanBytes(staging, byteCount);
                MemorySegment.copy(staging, 0L, destination, 0L, byteCount);
            } else if (ByteOrder.nativeOrder() == ByteOrder.BIG_ENDIAN || width == 1) {
                MemorySegment.copy(staging, 0L, destination, 0L, byteCount);
            } else {
                for (long offset = 0L; offset < byteCount; offset += width) {
                    for (int index = 0; index < width; index++) {
                        destination.set(BYTE, offset + index,
                                staging.get(BYTE, offset + width - 1L - index));
                    }
                }
            }
        }
        return result;
    }

    private static boolean isCanonicalTransfer(TensorDescriptor descriptor) {
        if (!descriptor.shape().isFullyStatic()
                || descriptor.shape().rank() > 16
                || descriptor.layout().isEmpty()
                || !descriptor.layout().orElseThrow().equals(
                        LayoutDescriptor.contiguous(descriptor.shape()))) {
            return false;
        }
        try {
            transferByteCount(descriptor);
            return true;
        } catch (IllegalArgumentException | ArithmeticException invalid) {
            return false;
        }
    }

    private static long transferByteCount(TensorDescriptor descriptor) {
        return Math.multiplyExact(
                descriptor.shape().knownElementCount().orElseThrow(),
                descriptor.dataType().byteWidth());
    }

    private static void validateBooleanBytes(MemorySegment source, long byteCount) {
        Objects.requireNonNull(source, "source");
        if (byteCount < 0L || source.byteSize() < byteCount) {
            throw new IllegalArgumentException("BOOL source is smaller than its transfer extent");
        }
        if (!source.isAccessibleBy(Thread.currentThread())) {
            throw new IllegalStateException("BOOL source is not accessible");
        }
        for (long offset = 0L; offset < byteCount; offset++) {
            byte value = source.get(BYTE, offset);
            if (value != 0 && value != 1) {
                throw new IllegalArgumentException("BOOL source contains a non-canonical byte");
            }
        }
    }

    /**
     * Closes the native context owner; child prepared/run resources retain deferred context leases.
     *
     * @throws RuntimeException if native cleanup fails
     * @throws Error if cleanup reports a fatal failure
     */
    @Override
    public void close() {
        context.close();
    }

    @FunctionalInterface
    interface RuntimePublisher {
        MetalBackendRuntime publish(MetalDeviceContext context);
    }

    private static void suppressDistinct(Throwable primary, Runnable cleanup) {
        try {
            cleanup.run();
        } catch (RuntimeException | Error failure) {
            if (failure != primary) primary.addSuppressed(failure);
        }
    }
}
