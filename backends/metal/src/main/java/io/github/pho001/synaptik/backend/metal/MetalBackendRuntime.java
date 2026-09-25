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
    private static final ValueLayout.OfInt NATIVE_INT =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder());

    private final MetalDeviceContext context;
    private final BackendAvailabilitySnapshot availabilitySnapshot;
    private final PartitionPreparation<?, ?> partitionPreparation;
    private final PreparedScheduleAssembler scheduleAssembler;

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
     * <p>The upload is complete before return. FLOAT32 and INT32 bytes are transferred without
     * conversion. The representation retains the storage as a non-owning borrow for the result
     * lifetime and owns only its Metal buffer.</p>
     *
     * @param storage non-null live accessible FLOAT32 or INT32 host storage; never closed here
     * @return a new non-null Metal buffer representation owned by the caller
     * @throws NullPointerException if {@code storage} is {@code null}
     * @throws IllegalArgumentException if the data type is neither FLOAT32 nor INT32
     * @throws IllegalStateException if storage or the Metal context is closed or inaccessible
     * @throws RuntimeException if native allocation or upload fails
     * @throws Error if allocation, upload, or rollback reports a fatal failure
     */
    BufferRepresentation borrow(HostTensorStorage storage) {
        Objects.requireNonNull(storage, "storage");
        if (storage.dataType() != DataType.FLOAT32 && storage.dataType() != DataType.INT32) {
            throw new IllegalArgumentException(
                    "Metal caller storage must have FLOAT32 or INT32 data type");
        }
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
     * contiguous FLOAT32 transfer.
     *
     * @param representation non-null candidate representation
     * @param descriptor non-null exact logical descriptor
     * @return whether representation type, context, layout, data type, and byte extent match
     */
    boolean acceptsContiguousFloat32Transfer(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        Objects.requireNonNull(representation, "representation");
        Objects.requireNonNull(descriptor, "descriptor");
        if (!isContiguousFloat32(descriptor)) {
            return false;
        }
        long byteCount = transferByteCount(descriptor);
        return representation instanceof MetalBufferRepresentation metal
                && metal.belongsTo(context)
                && metal.byteSize() == byteCount;
    }

    /**
     * Cold-binds one exact Metal destination to a direct upload action.
     *
     * @param representation non-null candidate destination
     * @param descriptor non-null exact logical descriptor
     * @return non-null immutable action capturing the exact typed destination
     */
    Consumer<MemorySegment> bindContiguousFloat32Upload(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        if (!acceptsContiguousFloat32Transfer(representation, descriptor)) {
            throw new IllegalArgumentException(
                    "Metal upload requires an exact live contiguous FLOAT32 representation");
        }
        MetalBufferRepresentation metal = (MetalBufferRepresentation) representation;
        long byteCount = transferByteCount(descriptor);
        return source -> metal.upload(0L, source, 0L, byteCount);
    }

    /**
     * Cold-binds one exact Metal source to a direct download action.
     *
     * @param representation non-null candidate source
     * @param descriptor non-null exact logical descriptor
     * @return non-null immutable action capturing the exact typed source
     */
    Consumer<MemorySegment> bindContiguousFloat32Download(
            BufferRepresentation representation, TensorDescriptor descriptor) {
        if (!acceptsContiguousFloat32Transfer(representation, descriptor)) {
            throw new IllegalArgumentException(
                    "Metal download requires an exact live contiguous FLOAT32 representation");
        }
        MetalBufferRepresentation metal = (MetalBufferRepresentation) representation;
        long byteCount = transferByteCount(descriptor);
        return destination -> metal.download(0L, destination, 0L, byteCount);
    }

    /**
     * Downloads one Metal FLOAT32 or BOOL publication into canonical row-major bytes.
     *
     * <p>FLOAT32 publication uses canonical big-endian element bytes. BOOL publication preserves
     * its exact one-byte zero-or-one elements. Canonical non-view FLOAT32 publications retain the
     * existing path and additionally admit one locally produced rank-zero reduction result as
     * exactly four detached bytes. A positive-rank logical affine view remains FLOAT32-only and is
     * accepted only when its representation carries exact finalized-route authentication for a
     * full dense represented-order target. The logical descriptor is validated but never
     * rewritten. This local BOOL publication does not add BOOL ingress or a general BOOL
     * consumer.</p>
     *
     * @param representation non-null live representation owned by this context
     * @param descriptor non-null exact canonical FLOAT32/BOOL or authenticated affine FLOAT32
     *     publication descriptor
     * @param maximumBytes non-negative caller byte ceiling
     * @return fresh non-null canonical bytes
     * @throws NullPointerException if an object argument is {@code null}
     * @throws IllegalArgumentException if type, layout, size, representation, or limit is invalid
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
        if (!(representation instanceof MetalBufferRepresentation metal)
                || !metal.belongsTo(context)) {
            throw new IllegalArgumentException(
                    "representation must be a live buffer owned by this Metal integration");
        }
        LayoutDescriptor layout = descriptor.layout().orElse(null);
        int rank = descriptor.shape().rank();
        DataType dataType = descriptor.dataType();
        boolean canonicalTypeAndRank = dataType == DataType.FLOAT32 && rank >= 0
                || dataType == DataType.BOOL && rank >= 1;
        boolean canonical = canonicalTypeAndRank
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
                    "Metal materialization requires canonical FLOAT32/BOOL"
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
            MemorySegment staging = arena.allocate(byteCount, dataType.byteWidth());
            metal.download(0L, staging, 0L, byteCount);
            if (dataType == DataType.BOOL) {
                MemorySegment.copy(staging, 0L, MemorySegment.ofArray(result), 0L, byteCount);
            } else {
                for (int offset = 0; offset < length; offset += Integer.BYTES) {
                    int bits = staging.get(NATIVE_INT, offset);
                    result[offset] = (byte) (bits >>> 24);
                    result[offset + 1] = (byte) (bits >>> 16);
                    result[offset + 2] = (byte) (bits >>> 8);
                    result[offset + 3] = (byte) bits;
                }
            }
        }
        return result;
    }

    private static boolean isContiguousFloat32(TensorDescriptor descriptor) {
        if (descriptor.dataType() != DataType.FLOAT32
                || !descriptor.shape().isFullyStatic()
                || descriptor.shape().rank() < 1
                || descriptor.shape().rank() > 16
                || descriptor.layout().isEmpty()
                || !descriptor.layout().orElseThrow().equals(
                        LayoutDescriptor.contiguous(descriptor.shape()))) {
            return false;
        }
        try {
            long elements = 1L;
            for (long dimension : descriptor.shape().toLongArray()) {
                if (dimension <= 0L) {
                    return false;
                }
                elements = Math.multiplyExact(elements, dimension);
            }
            Math.multiplyExact(elements, Float.BYTES);
            return true;
        } catch (IllegalArgumentException | ArithmeticException invalid) {
            return false;
        }
    }

    private static long transferByteCount(TensorDescriptor descriptor) {
        return Math.multiplyExact(
                descriptor.shape().knownElementCount().orElseThrow(), Float.BYTES);
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
