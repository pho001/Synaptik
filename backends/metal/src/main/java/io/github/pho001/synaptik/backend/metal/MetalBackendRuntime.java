package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.model.datatype.DataType;
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
     * <p>The upload is complete before return. All six model data types transfer their exact raw
     * physical storage bytes. This descriptor-free borrow does not interpret BOOL bytes: later
     * descriptor-aware execution, transfer, and publication paths validate logical BOOL positions
     * while leaving prefix and gap bytes uninterpreted. The representation retains the storage as
     * a non-owning borrow and owns only its Metal buffer.</p>
     *
     * @param storage non-null live accessible host storage; never closed here
     * @return a new non-null Metal buffer representation owned by the caller
     * @throws NullPointerException if {@code storage} is {@code null}
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
     * storage-layout transfer for any model data type.
     *
     * @param representation non-null candidate representation
     * @param descriptor non-null exact logical descriptor
     * @return whether representation type, context, layout, data type, and byte extent match
     */
    boolean acceptsStorageLayoutTransfer(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        Objects.requireNonNull(representation, "representation");
        Objects.requireNonNull(descriptor, "descriptor");
        if (!isStorageTransfer(descriptor)) {
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
     * Cold-binds one exact Metal destination to a physical storage-layout upload action.
     *
     * @param representation non-null candidate destination
     * @param descriptor non-null exact logical descriptor
     * @return non-null immutable action capturing the exact typed destination
     */
    Consumer<MemorySegment> bindStorageLayoutUpload(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        if (!(representation instanceof MetalBufferRepresentation metal)
                || !acceptsStorageLayoutTransfer(representation, descriptor)) {
            throw new IllegalArgumentException(
                    "Metal upload requires an exact writable storage representation");
        }
        long byteCount = transferByteCount(descriptor);
        return source -> {
            if (descriptor.dataType() == DataType.BOOL) {
                validateBooleanElements(source, descriptor);
            }
            metal.upload(0L, source, 0L, byteCount);
        };
    }

    /**
     * Cold-binds one exact Metal source to a physical storage-layout download action.
     *
     * <p>BOOL allocates one automatically managed private native staging span during this cold
     * bind. The reusable action serializes access to that span, validates every logical BOOL byte,
     * and commits only after complete validation. Its hot invocation allocates nothing.</p>
     *
     * @param representation non-null candidate source
     * @param descriptor non-null exact logical descriptor
     * @return non-null immutable action capturing the exact typed source
     */
    Consumer<MemorySegment> bindStorageLayoutDownload(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        if (!acceptsStorageLayoutTransfer(representation, descriptor)) {
            throw new IllegalArgumentException(
                    "Metal download requires an exact live storage representation");
        }
        MetalBufferRepresentation metal =
                MetalPreparedSplatResource.readableBuffer(representation);
        long byteCount = transferByteCount(descriptor);
        if (descriptor.dataType() != DataType.BOOL) {
            return destination -> metal.download(0L, destination, 0L, byteCount);
        }
        MemorySegment staging = Arena.ofAuto().allocate(byteCount, 1L);
        Object stagingLock = new Object();
        return destination -> {
            requireWritableNativeStorage(destination, byteCount);
            synchronized (stagingLock) {
                metal.download(0L, staging, 0L, byteCount);
                validateBooleanElements(staging, descriptor);
                MemorySegment.copy(staging, 0L, destination, 0L, byteCount);
            }
        };
    }

    /**
     * Downloads one canonical or authenticated storage-layout publication into row-major
     * big-endian bytes.
     *
     * <p>Canonical non-view rank-0..16 publications and authenticated SELECT/SLICE storage-layout
     * publications support all six model data types. BOOL logical elements must be zero or one;
     * prefix and gap bytes are not interpreted. Other authenticated affine publication remains
     * FLOAT32-only. The logical descriptor is validated but never rewritten.</p>
     *
     * @param representation non-null live representation owned by this context
     * @param descriptor non-null exact canonical or authenticated publication descriptor
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
                && layout.equals(LayoutDescriptor.contiguous(descriptor.shape()));
        boolean authenticatedPublication = descriptor.shape().isFullyStatic()
                && rank >= 1
                && rank <= 16
                && layout != null
                && metal.authenticatesPublication(descriptor);
        boolean storagePublication = authenticatedPublication
                && metal.authenticatedPublicationUsesStorageLayout(descriptor);
        if (!canonical && !authenticatedPublication) {
            throw new IllegalArgumentException(
                    "Metal materialization requires a canonical rank-0..16 descriptor"
                            + " or authenticated layout publication");
        }
        if (storagePublication && !isStorageTransfer(descriptor)) {
            throw new IllegalArgumentException(
                    "authenticated storage publication has unsupported layout geometry");
        }
        long elements = 1L;
        for (long dimension : descriptor.shape().toLongArray()) {
            if (dimension <= 0L) {
                throw new IllegalArgumentException(
                        "descriptor dimensions are outside the materialization domain");
            }
            elements = Math.multiplyExact(elements, dimension);
        }
        int width = dataType.byteWidth();
        long byteCount = Math.multiplyExact(elements, width);
        if (byteCount > maximumBytes) {
            throw new IllegalArgumentException("canonical payload exceeds maximumBytes");
        }
        long physicalByteCount = storagePublication
                ? transferByteCount(descriptor)
                : byteCount;
        if (metal.byteSize() < physicalByteCount) {
            throw new IllegalArgumentException("Metal representation is smaller than descriptor");
        }
        byte[] result = new byte[Math.toIntExact(byteCount)];
        long[] dimensions = descriptor.shape().toLongArray();
        long[] strides = layout.strides();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment staging = arena.allocate(physicalByteCount, width);
            metal.download(0L, staging, 0L, physicalByteCount);
            for (long ordinal = 0L; ordinal < elements; ordinal++) {
                long sourceElement = storagePublication
                        ? storageElementIndex(
                                ordinal, dimensions, strides, layout.storageOffset())
                        : ordinal;
                long sourceOffset = Math.multiplyExact(sourceElement, width);
                long destinationOffset = Math.multiplyExact(ordinal, width);
                if (dataType == DataType.BOOL) {
                    byte value = staging.get(BYTE, sourceOffset);
                    if (value != 0 && value != 1) {
                        throw new IllegalArgumentException(
                                "BOOL source contains a non-canonical byte");
                    }
                    result[Math.toIntExact(destinationOffset)] = value;
                } else if (ByteOrder.nativeOrder() == ByteOrder.BIG_ENDIAN || width == 1) {
                    for (int index = 0; index < width; index++) {
                        result[Math.toIntExact(destinationOffset + index)] =
                                staging.get(BYTE, sourceOffset + index);
                    }
                } else {
                    for (int index = 0; index < width; index++) {
                        result[Math.toIntExact(destinationOffset + index)] =
                                staging.get(BYTE, sourceOffset + width - 1L - index);
                    }
                }
            }
        }
        return result;
    }

    private static boolean isStorageTransfer(TensorDescriptor descriptor) {
        if (!descriptor.shape().isFullyStatic()
                || descriptor.shape().rank() > 16
                || descriptor.layout().isEmpty()) {
            return false;
        }
        long[] dimensions = descriptor.shape().toLongArray();
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        long[] strides = layout.strides();
        long[] activeStrides = new long[strides.length];
        long[] activeDimensions = new long[strides.length];
        int active = 0;
        try {
            for (int axis = 0; axis < dimensions.length; axis++) {
                if (dimensions[axis] <= 0L || strides[axis] <= 0L) {
                    return false;
                }
                if (dimensions[axis] > 1L) {
                    int insertion = active;
                    while (insertion > 0
                            && activeStrides[insertion - 1] > strides[axis]) {
                        activeStrides[insertion] = activeStrides[insertion - 1];
                        activeDimensions[insertion] = activeDimensions[insertion - 1];
                        insertion--;
                    }
                    activeStrides[insertion] = strides[axis];
                    activeDimensions[insertion] = dimensions[axis];
                    active++;
                }
            }
            long covered = 1L;
            for (int index = 0; index < active; index++) {
                if (activeStrides[index] < covered) {
                    return false;
                }
                covered = Math.addExact(
                        covered,
                        Math.multiplyExact(
                                activeDimensions[index] - 1L,
                                activeStrides[index]));
            }
            return layout.referencedElementSpan()
                    == Math.addExact(layout.storageOffset(), covered)
                    && Math.multiplyExact(
                            layout.referencedElementSpan(),
                            descriptor.dataType().byteWidth()) > 0L;
        } catch (ArithmeticException overflow) {
            return false;
        }
    }

    private static long transferByteCount(TensorDescriptor descriptor) {
        return Math.multiplyExact(
                descriptor.layout().orElseThrow().referencedElementSpan(),
                descriptor.dataType().byteWidth());
    }

    private static long storageElementIndex(
            long ordinal, long[] dimensions, long[] strides, long storageOffset) {
        long remaining = ordinal;
        long storage = storageOffset;
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            long coordinate = remaining % dimensions[axis];
            remaining /= dimensions[axis];
            storage = Math.addExact(storage, Math.multiplyExact(coordinate, strides[axis]));
        }
        return storage;
    }

    private static void validateBooleanElements(
            MemorySegment source, TensorDescriptor descriptor) {
        Objects.requireNonNull(source, "source");
        long byteCount = transferByteCount(descriptor);
        if (source.byteSize() < byteCount) {
            throw new IllegalArgumentException("BOOL source is smaller than its transfer extent");
        }
        if (!source.isAccessibleBy(Thread.currentThread())) {
            throw new IllegalStateException("BOOL source is not accessible");
        }
        long[] dimensions = descriptor.shape().toLongArray();
        long[] strides = descriptor.layout().orElseThrow().strides();
        long elements = descriptor.shape().knownElementCount().orElseThrow();
        long offset = descriptor.layout().orElseThrow().storageOffset();
        for (long ordinal = 0L; ordinal < elements; ordinal++) {
            byte value = source.get(
                    BYTE, storageElementIndex(ordinal, dimensions, strides, offset));
            if (value != 0 && value != 1) {
                throw new IllegalArgumentException("BOOL source contains a non-canonical byte");
            }
        }
    }

    private static void requireWritableNativeStorage(
            MemorySegment destination, long byteCount) {
        Objects.requireNonNull(destination, "destination");
        if (!destination.scope().isAlive()
                || !destination.isAccessibleBy(Thread.currentThread())) {
            throw new IllegalStateException(
                    "transfer destination is not alive and accessible");
        }
        if (!destination.isNative() || destination.isReadOnly()
                || destination.byteSize() < byteCount) {
            throw new IllegalArgumentException(
                    "transfer destination must be writable native storage of sufficient size");
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
