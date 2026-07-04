package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Public construction boundary for tensors with factory-assigned identity.
 *
 * <p>This non-instantiable static utility assigns each successful allocation a unique
 * non-negative {@link TensorId} within the current Java virtual machine (JVM), including when
 * callers create tensors concurrently. Identifier allocation is monotonic from zero through
 * {@link Long#MAX_VALUE}, but callers must treat values as opaque: semantic construction failures
 * consume identifiers, completion order may differ from numeric order, and neither adjacency nor
 * cross-process uniqueness is promised. After the final value is claimed, creation fails
 * permanently rather than wrapping or reusing an identifier.</p>
 *
 * <p>The factory retains no tensor, descriptor, storage, graph, backend, or service state. Its
 * atomics allocate model identity only; they are not a runtime service locator or registry. The
 * factory accepts completed descriptors and optional caller-supplied borrowed storage, delegates
 * label and storage semantics to {@link Tensor}, and can allocate exact-span Java primitive-array
 * storage for a resolved layout. It can also import a matching flat primitive array into a
 * resolved dense-contiguous tensor by copying its row-major logical values into newly allocated
 * heap storage. Numeric values and raw BFLOAT16 bits are copied unchanged; BOOL bytes are
 * normalized to zero or one. Heap allocation uses automatic-scope memory segments, so it
 * introduces no arena, close operation, external lifetime owner, or deterministic reclamation.
 * The factory does not build descriptors, resolve layouts, retain or expose source/backing arrays,
 * convert values, import nested arrays, provide typed tensor access, create provenance, allocate
 * native or backend memory, or provide compiler, runtime, or backend behavior.</p>
 */
public final class TensorFactory {
    /**
     * Holds the next ordinary non-negative candidate below {@link Long#MAX_VALUE}.
     *
     * <p>Successful compare-and-set advances establish unique ordinary allocations. At the upper
     * boundary this value remains {@code Long.MAX_VALUE}; the separate final-value flag records
     * whether that valid candidate has already been claimed.</p>
     */
    private static final AtomicLong NEXT_TENSOR_ID = new AtomicLong();

    /**
     * Records whether the valid final identifier value has been claimed.
     *
     * <p>This flag is false throughout ordinary allocation. Its one successful compare-and-set
     * linearizes the allocation of {@code Long.MAX_VALUE}; true then represents permanent
     * exhaustion without using a negative sentinel or wrapping the ordinary counter.</p>
     */
    private static final AtomicBoolean MAXIMUM_TENSOR_ID_CLAIMED = new AtomicBoolean();

    /**
     * Prevents instantiation because construction and identity allocation are JVM-wide static
     * operations and the factory has no instance state.
     */
    private TensorFactory() {
    }

    /**
     * Creates a fresh unlabeled tensor without host storage from a completed descriptor.
     *
     * <p>The exact descriptor reference is retained by the returned tensor. No layout or storage
     * is synthesized. A null descriptor is rejected before identifier allocation and therefore
     * does not consume an identifier.</p>
     *
     * @param descriptor non-null completed immutable descriptor to retain by exact reference;
     *     the factory does not inspect or alter its data type, shape, layout, or gradient request
     * @return a non-null fresh tensor with factory-assigned identity, the exact descriptor,
     *     no label, and no host storage
     * @throws NullPointerException if {@code descriptor} is {@code null}, with message
     *     {@code descriptor}; this failure does not consume an identifier
     * @throws IllegalStateException if every non-negative identifier has been allocated, with
     *     message {@code tensor identifier space exhausted}
     */
    public static Tensor create(TensorDescriptor descriptor) {
        return create(descriptor, Optional.empty(), Optional.empty());
    }

    /**
     * Creates a fresh tensor from a completed descriptor and optional caller-supplied metadata.
     *
     * <p>The descriptor and, when present, borrowed host-storage references are passed unchanged
     * to the package-private {@link Tensor} construction path. The label optional uses value
     * semantics; {@code Tensor} strips present text and rejects a blank result. Storage remains
     * caller-supplied and borrowed, may be read-only, and receives the existing Tensor data-type,
     * resolved-span, and point-in-time liveness validation. The factory neither accesses memory
     * nor extends its lifetime.</p>
     *
     * <p>Null descriptor, label-container, and storage-container failures are checked in that
     * order before allocation and consume no identifier. Allocation then precedes delegated
     * Tensor validation. Consequently, blank-label and incompatible or dead-storage failures
     * consume their allocated identifier without rollback or reuse. Exhaustion therefore wins
     * over a delegated semantic failure, while a null factory argument wins over exhaustion.</p>
     *
     * @param descriptor non-null completed immutable descriptor to retain by exact reference;
     *     the factory does not inspect or alter its contents
     * @param label non-null optional diagnostic label; empty means absent and present text is
     *     normalized and validated only by {@code Tensor}
     * @param hostStorage non-null optional caller-supplied borrowed host storage; empty means
     *     absent and a present object is retained by exact reference after delegated validation
     * @return a non-null fresh tensor with factory-assigned opaque identity and the exact supplied
     *     descriptor and compatible present storage references
     * @throws NullPointerException if {@code descriptor}, {@code label}, or {@code hostStorage}
     *     is {@code null}, checked in that order with the corresponding parameter name as the
     *     message; these failures do not consume an identifier
     * @throws IllegalArgumentException if {@code Tensor} rejects a present blank label, mismatched
     *     storage data type, or storage capacity smaller than a resolved layout span; the
     *     allocated identifier is consumed
     * @throws IllegalStateException if identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, or if {@code Tensor} rejects storage that is
     *     not alive at attachment time; a delegated storage failure consumes the allocated
     *     identifier
     */
    public static Tensor create(
            TensorDescriptor descriptor,
            Optional<String> label,
            Optional<HostTensorStorage> hostStorage) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(hostStorage, "hostStorage");
        return new Tensor(nextTensorId(), descriptor, label, hostStorage);
    }

    /**
     * Allocates JVM-managed heap storage for a completed resolved descriptor.
     *
     * <p>This convenience is equivalent to {@link #allocate(TensorDescriptor, Optional)} with an
     * empty label. The resolved layout's referenced element span is the exact physical capacity;
     * logical element count is not substituted. The data type selects {@code double[]},
     * {@code float[]}, {@code short[]}, {@code int[]}, {@code long[]}, or {@code byte[]} for
     * {@code FLOAT64}, {@code FLOAT32}, {@code BFLOAT16}, {@code INT32}, {@code INT64}, or
     * {@code BOOL}, respectively. Primitive-array contents begin with the JVM default all-zero raw
     * representation; this is an allocation fact, not a public fill or zeros operation.</p>
     *
     * <p>The array-backed segment has an automatic scope that keeps its heap base reachable and is
     * accessible from any thread. The returned tensor retains matching writable
     * {@link MemorySegmentStorage}; there is no arena, close operation, external owner, or
     * deterministic release. The tensor, storage, segment, and array have an ordinary
     * garbage-collected lifetime. Validation through storage wrapping completes before delegation
     * to the existing creation path and identifier allocation.</p>
     *
     * @param descriptor non-null completed immutable descriptor to retain by exact reference; it
     *     must contain a resolved layout whose referenced span fits a Java primitive array
     * @return a non-null fresh tensor with factory-assigned opaque identity, the exact descriptor,
     *     no label, and matching writable JVM-managed heap storage
     * @throws NullPointerException if {@code descriptor} is {@code null}, with message
     *     {@code descriptor}; this failure occurs before identifier allocation
     * @throws IllegalArgumentException if the descriptor layout is unresolved, with message
     *     {@code tensor allocation requires a resolved layout}, or its referenced span exceeds
     *     {@link Integer#MAX_VALUE}, with the documented Java-array-limit message; these failures
     *     occur before identifier allocation
     * @throws IllegalStateException if every non-negative identifier has been allocated, with
     *     message {@code tensor identifier space exhausted}; heap storage has already been
     *     allocated when this failure occurs
     * @throws OutOfMemoryError if the JVM cannot allocate the required primitive array; this
     *     propagates before identifier allocation
     */
    public static Tensor allocate(TensorDescriptor descriptor) {
        return allocate(descriptor, Optional.empty());
    }

    /**
     * Allocates JVM-managed heap storage for a completed resolved descriptor and optional label.
     *
     * <p>The descriptor and label are retained or normalized through the existing
     * {@link #create(TensorDescriptor, Optional, Optional)} path. Allocation requires a resolved
     * layout and uses exactly its referenced element span, including offset and strided geometry
     * and allowing broadcast geometry to require less storage than logical element count. Spans
     * above {@link Integer#MAX_VALUE} are rejected without a native or off-heap fallback.</p>
     *
     * <p>The exhaustive carrier mapping is {@code FLOAT64 -> double[]},
     * {@code FLOAT32 -> float[]}, {@code BFLOAT16 -> short[]}, {@code INT32 -> int[]},
     * {@code INT64 -> long[]}, and {@code BOOL -> byte[]}. The matching
     * {@link MemorySegment#ofArray(double[]) MemorySegment.ofArray} overload creates a writable
     * heap segment with an automatic scope that keeps the array reachable and permits access from
     * any thread. Raw contents start with the JVM default zero representation. No backing-array
     * API, typed access, conversion, normalization, fill, arena, close path, or deterministic
     * reclamation is added.</p>
     *
     * <p>Null, unresolved-layout, over-limit, JVM allocation, segment, and storage failures occur
     * before identifier allocation. Heap allocation and storage wrapping precede delegation to
     * {@code create}. A present blank label is therefore rejected only by {@code Tensor} after an
     * identifier is allocated and consumes that identifier; identifier exhaustion is likewise
     * observed after heap allocation. An {@link OutOfMemoryError} and any unexpected segment or
     * storage-construction failure propagate unchanged without consuming an identifier. No failed
     * identifier is rolled back or reused.</p>
     *
     * @param descriptor non-null completed immutable descriptor to retain by exact reference; it
     *     must contain resolved layout geometry whose referenced span fits a Java array
     * @param label non-null optional diagnostic label; empty means absent, while present text is
     *     normalized and validated by {@code Tensor} after heap allocation and ID allocation
     * @return a non-null fresh tensor with factory-assigned opaque identity, the exact descriptor,
     *     normalized optional label, and exact newly created writable heap-storage reference
     * @throws NullPointerException if {@code descriptor} or {@code label} is {@code null}, checked
     *     in that order with the corresponding parameter name as the message; these failures occur
     *     before allocation and do not consume an identifier
     * @throws IllegalArgumentException if the descriptor layout is unresolved, with message
     *     {@code tensor allocation requires a resolved layout}; if the referenced span exceeds the
     *     Java array limit, with message {@code tensor allocation span exceeds Java array limit:
     *     required=<required>, maximum=2147483647}; or if delegated Tensor validation rejects a
     *     blank label, with message {@code label must not be blank}. Preallocation failures consume
     *     no identifier, while the delegated label failure does
     * @throws IllegalStateException if every non-negative identifier has been allocated, with
     *     message {@code tensor identifier space exhausted}; heap storage has already been
     *     allocated when this failure occurs
     * @throws OutOfMemoryError if the JVM cannot allocate the required primitive array; this error
     *     propagates unchanged before identifier allocation
     */
    public static Tensor allocate(
            TensorDescriptor descriptor,
            Optional<String> label) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(label, "label");

        var layout = descriptor.layout();
        if (layout.isEmpty()) {
            throw new IllegalArgumentException("tensor allocation requires a resolved layout");
        }
        long requiredSpan = layout.orElseThrow().referencedElementSpan();
        if (requiredSpan > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "tensor allocation span exceeds Java array limit: required="
                            + requiredSpan
                            + ", maximum="
                            + Integer.MAX_VALUE);
        }

        int length = (int) requiredSpan;
        MemorySegment segment = switch (descriptor.dataType()) {
            case FLOAT64 -> MemorySegment.ofArray(new double[length]);
            case FLOAT32 -> MemorySegment.ofArray(new float[length]);
            case BFLOAT16 -> MemorySegment.ofArray(new short[length]);
            case INT32 -> MemorySegment.ofArray(new int[length]);
            case INT64 -> MemorySegment.ofArray(new long[length]);
            case BOOL -> MemorySegment.ofArray(new byte[length]);
        };
        MemorySegmentStorage storage =
                new MemorySegmentStorage(descriptor.dataType(), requiredSpan, segment);
        return create(descriptor, label, Optional.of(storage));
    }

    /**
     * Creates a dense-contiguous {@link DataType#FLOAT64} tensor by copying flat binary64 values.
     *
     * <p>The source order is the tensor's logical row-major order. The descriptor must have a
     * resolved dense-contiguous layout and its known logical element count must equal the source
     * length. The source array is not retained, and mutation after return cannot affect the
     * tensor. Allocation, optional-label normalization, identifier allocation, and their failure
     * side effects remain those of {@link #allocate(TensorDescriptor, Optional)}. Carrier, layout,
     * and length failures occur before destination or identifier allocation. A blank label fails
     * after destination and identifier allocation but before copying; exhaustion occurs after
     * destination allocation and before copying. Unexpected copy failures consume the identifier,
     * and no rollback is attempted.</p>
     *
     * @param descriptor non-null completed descriptor whose data type is {@code FLOAT64} and whose
     *     resolved layout is dense-contiguous
     * @param label non-null optional diagnostic label passed to allocation; present text is
     *     normalized and validated by {@link Tensor}
     * @param source non-null flat binary64 values to copy; the caller retains the array
     * @return a non-null fresh tensor containing an independent copy of all source values
     * @throws NullPointerException if {@code descriptor}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the descriptor data type is not {@code FLOAT64}, its
     *     layout is unresolved or not dense-contiguous, source length differs from logical element
     *     count, or delegated Tensor validation rejects a blank label
     * @throws IllegalStateException if resolved geometry unexpectedly has no known logical element
     *     count, or identifier space is exhausted with message
     *     {@code tensor identifier space exhausted}
     * @throws OutOfMemoryError if destination heap allocation fails before identifier allocation
     */
    public static Tensor fromFlatArray(
            TensorDescriptor descriptor, Optional<String> label, double[] source) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return importFlat(
                descriptor,
                label,
                DataType.FLOAT64,
                source.length,
                MemorySegment.ofArray(source));
    }

    /**
     * Creates a dense-contiguous {@link DataType#FLOAT32} tensor by copying flat binary32 values.
     *
     * <p>The source order is logical row-major order. A resolved dense-contiguous layout, exact
     * data-type match, and exact logical element count are required. The source array is not
     * retained. Allocation, label validation, and identifier side effects follow
     * {@link #allocate(TensorDescriptor, Optional)}. Carrier, layout, and length failures occur
     * before destination or identifier allocation. A blank label fails after destination and
     * identifier allocation but before copying; exhaustion occurs after destination allocation
     * and before copying. Unexpected copy failures consume the identifier without rollback.</p>
     *
     * @param descriptor non-null completed descriptor whose data type is {@code FLOAT32} and whose
     *     resolved layout is dense-contiguous
     * @param label non-null optional diagnostic label delegated to allocation
     * @param source non-null flat binary32 values to copy; the caller retains the array
     * @return a non-null fresh tensor containing an independent copy of all source values
     * @throws NullPointerException if {@code descriptor}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the descriptor data type is not {@code FLOAT32}, its
     *     layout is unresolved or not dense-contiguous, source length differs from logical element
     *     count, or delegated Tensor validation rejects a blank label
     * @throws IllegalStateException if resolved geometry unexpectedly has no known logical element
     *     count, or identifier space is exhausted with message
     *     {@code tensor identifier space exhausted}
     * @throws OutOfMemoryError if destination heap allocation fails before identifier allocation
     */
    public static Tensor fromFlatArray(
            TensorDescriptor descriptor, Optional<String> label, float[] source) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return importFlat(
                descriptor,
                label,
                DataType.FLOAT32,
                source.length,
                MemorySegment.ofArray(source));
    }

    /**
     * Creates a dense-contiguous {@link DataType#BFLOAT16} tensor by copying raw BFLOAT16 bits.
     *
     * <p>Each short is copied bit-for-bit without floating-point conversion. Source order is
     * logical row-major order. A resolved dense-contiguous layout and exact logical element count
     * are required. The source array is not retained; allocation, label validation, and identifier
     * side effects follow {@link #allocate(TensorDescriptor, Optional)}. Carrier, layout, and
     * length failures occur before destination or identifier allocation. A blank label fails after
     * destination and identifier allocation but before copying; exhaustion occurs after
     * destination allocation and before copying. Unexpected copy failures consume the identifier
     * without rollback.</p>
     *
     * @param descriptor non-null completed descriptor whose data type is {@code BFLOAT16} and whose
     *     resolved layout is dense-contiguous
     * @param label non-null optional diagnostic label delegated to allocation
     * @param source non-null raw BFLOAT16 bit patterns to copy; the caller retains the array
     * @return a non-null fresh tensor containing an independent bit-for-bit copy of the source
     * @throws NullPointerException if {@code descriptor}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the descriptor data type is not {@code BFLOAT16}, its
     *     layout is unresolved or not dense-contiguous, source length differs from logical element
     *     count, or delegated Tensor validation rejects a blank label
     * @throws IllegalStateException if resolved geometry unexpectedly has no known logical element
     *     count, or identifier space is exhausted with message
     *     {@code tensor identifier space exhausted}
     * @throws OutOfMemoryError if destination heap allocation fails before identifier allocation
     */
    public static Tensor fromFlatArray(
            TensorDescriptor descriptor, Optional<String> label, short[] source) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return importFlat(
                descriptor,
                label,
                DataType.BFLOAT16,
                source.length,
                MemorySegment.ofArray(source));
    }

    /**
     * Creates a dense-contiguous {@link DataType#INT32} tensor by copying flat integer values.
     *
     * <p>The source order is logical row-major order. A resolved dense-contiguous layout, exact
     * data-type match, and exact logical element count are required. The source array is not
     * retained; allocation, label validation, and identifier side effects follow
     * {@link #allocate(TensorDescriptor, Optional)}. Carrier, layout, and length failures occur
     * before destination or identifier allocation. A blank label fails after destination and
     * identifier allocation but before copying; exhaustion occurs after destination allocation
     * and before copying. Unexpected copy failures consume the identifier without rollback.</p>
     *
     * @param descriptor non-null completed descriptor whose data type is {@code INT32} and whose
     *     resolved layout is dense-contiguous
     * @param label non-null optional diagnostic label delegated to allocation
     * @param source non-null flat signed 32-bit values to copy; the caller retains the array
     * @return a non-null fresh tensor containing an independent copy of all source values
     * @throws NullPointerException if {@code descriptor}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the descriptor data type is not {@code INT32}, its
     *     layout is unresolved or not dense-contiguous, source length differs from logical element
     *     count, or delegated Tensor validation rejects a blank label
     * @throws IllegalStateException if resolved geometry unexpectedly has no known logical element
     *     count, or identifier space is exhausted with message
     *     {@code tensor identifier space exhausted}
     * @throws OutOfMemoryError if destination heap allocation fails before identifier allocation
     */
    public static Tensor fromFlatArray(
            TensorDescriptor descriptor, Optional<String> label, int[] source) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return importFlat(
                descriptor,
                label,
                DataType.INT32,
                source.length,
                MemorySegment.ofArray(source));
    }

    /**
     * Creates a dense-contiguous {@link DataType#INT64} tensor by copying flat integer values.
     *
     * <p>The source order is logical row-major order. A resolved dense-contiguous layout, exact
     * data-type match, and exact logical element count are required. The source array is not
     * retained; allocation, label validation, and identifier side effects follow
     * {@link #allocate(TensorDescriptor, Optional)}. Carrier, layout, and length failures occur
     * before destination or identifier allocation. A blank label fails after destination and
     * identifier allocation but before copying; exhaustion occurs after destination allocation
     * and before copying. Unexpected copy failures consume the identifier without rollback.</p>
     *
     * @param descriptor non-null completed descriptor whose data type is {@code INT64} and whose
     *     resolved layout is dense-contiguous
     * @param label non-null optional diagnostic label delegated to allocation
     * @param source non-null flat signed 64-bit values to copy; the caller retains the array
     * @return a non-null fresh tensor containing an independent copy of all source values
     * @throws NullPointerException if {@code descriptor}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the descriptor data type is not {@code INT64}, its
     *     layout is unresolved or not dense-contiguous, source length differs from logical element
     *     count, or delegated Tensor validation rejects a blank label
     * @throws IllegalStateException if resolved geometry unexpectedly has no known logical element
     *     count, or identifier space is exhausted with message
     *     {@code tensor identifier space exhausted}
     * @throws OutOfMemoryError if destination heap allocation fails before identifier allocation
     */
    public static Tensor fromFlatArray(
            TensorDescriptor descriptor, Optional<String> label, long[] source) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return importFlat(
                descriptor,
                label,
                DataType.INT64,
                source.length,
                MemorySegment.ofArray(source));
    }

    /**
     * Creates a dense-contiguous {@link DataType#BOOL} tensor from flat logical bytes.
     *
     * <p>Each zero byte is stored as canonical false byte {@code 0}; every non-zero byte is stored
     * as canonical true byte {@code 1}. Source order is logical row-major order. A resolved
     * dense-contiguous layout and exact logical element count are required. The source array is not
     * retained; allocation, label validation, and identifier side effects follow
     * {@link #allocate(TensorDescriptor, Optional)}. Carrier, layout, and length failures occur
     * before destination or identifier allocation. A blank label fails after destination and
     * identifier allocation but before normalization; exhaustion occurs after destination
     * allocation and before normalization. Unexpected normalization failures consume the
     * identifier without rollback.</p>
     *
     * @param descriptor non-null completed descriptor whose data type is {@code BOOL} and whose
     *     resolved layout is dense-contiguous
     * @param label non-null optional diagnostic label delegated to allocation
     * @param source non-null flat logical bytes to normalize and copy; the caller retains the array
     * @return a non-null fresh tensor containing canonical zero-or-one bytes independent of source
     * @throws NullPointerException if {@code descriptor}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the descriptor data type is not {@code BOOL}, its layout
     *     is unresolved or not dense-contiguous, source length differs from logical element count,
     *     or delegated Tensor validation rejects a blank label
     * @throws IllegalStateException if resolved geometry unexpectedly has no known logical element
     *     count, or identifier space is exhausted with message
     *     {@code tensor identifier space exhausted}
     * @throws OutOfMemoryError if destination heap allocation fails before identifier allocation
     */
    public static Tensor fromFlatArray(
            TensorDescriptor descriptor, Optional<String> label, byte[] source) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return importFlat(
                descriptor,
                label,
                DataType.BOOL,
                source.length,
                MemorySegment.ofArray(source));
    }

    /**
     * Validates and imports one already carrier-typed flat row-major source segment.
     *
     * <p>Logical import is limited to resolved dense-contiguous geometry so sequential source
     * positions map one-to-one to sequential destination positions. Exact carrier data type and
     * logical element count are validated before destination or identifier allocation. Numeric
     * carriers and raw BFLOAT16 bits use a byte-for-byte bulk copy; only BOOL is normalized because
     * its byte carrier accepts multiple non-zero encodings for the same logical true value. After
     * validation, the method calls {@link #allocate(TensorDescriptor, Optional)} exactly once,
     * obtains the attached destination segment, fully populates it, and only then returns the
     * tensor. The source segment and its array are never retained.</p>
     *
     * <p>A delegated blank-label failure occurs after destination and identifier allocation but
     * before population. Identifier exhaustion occurs inside allocation after destination storage
     * allocation and before population. An unexpected population failure occurs after identifier
     * creation, consumes that identifier, and is not rolled back.</p>
     *
     * @param descriptor non-null completed descriptor validated by the public overload
     * @param label non-null optional label delegated unchanged to allocation
     * @param sourceDataType exact non-null data type belonging to the public source carrier
     * @param sourceLength non-negative source element count
     * @param sourceSegment non-null temporary non-owning segment over the caller's source array
     * @return the exact newly allocated and fully populated tensor; never {@code null}
     * @throws IllegalArgumentException if the carrier data type differs from the descriptor, the
     *     layout is unresolved or is not {@link LayoutKind#DENSE_CONTIGUOUS}, source length differs
     *     from logical element count, or delegated label validation rejects blank text
     * @throws IllegalStateException if a resolved layout unexpectedly has no known logical element
     *     count, or identifier space is exhausted
     * @throws OutOfMemoryError if destination heap allocation fails
     */
    private static Tensor importFlat(
            TensorDescriptor descriptor,
            Optional<String> label,
            DataType sourceDataType,
            int sourceLength,
            MemorySegment sourceSegment) {
        if (descriptor.dataType() != sourceDataType) {
            throw new IllegalArgumentException(
                    "flat source data type must match descriptor: expected="
                            + sourceDataType
                            + ", actual="
                            + descriptor.dataType());
        }

        var layout = descriptor.layout();
        if (layout.isEmpty()) {
            throw new IllegalArgumentException("flat tensor import requires a resolved layout");
        }
        LayoutKind layoutKind = layout.orElseThrow().kind();
        if (layoutKind != LayoutKind.DENSE_CONTIGUOUS) {
            throw new IllegalArgumentException(
                    "flat tensor import requires dense-contiguous layout: actual=" + layoutKind);
        }

        var knownElementCount = descriptor.shape().knownElementCount();
        if (knownElementCount.isEmpty()) {
            throw new IllegalStateException("resolved tensor layout requires a fully static shape");
        }
        long required = knownElementCount.getAsLong();
        if (required != sourceLength) {
            throw new IllegalArgumentException(
                    "flat source length must equal logical element count: required="
                            + required
                            + ", actual="
                            + sourceLength);
        }

        Tensor tensor = allocate(descriptor, label);
        MemorySegment destination = tensor.hostStorage().orElseThrow().segment();
        if (sourceDataType == DataType.BOOL) {
            for (long index = 0; index < sourceLength; index++) {
                byte value = sourceSegment.get(ValueLayout.JAVA_BYTE, index);
                destination.set(ValueLayout.JAVA_BYTE, index, value == 0 ? (byte) 0 : (byte) 1);
            }
        } else {
            MemorySegment.copy(sourceSegment, 0, destination, 0, sourceSegment.byteSize());
        }
        return tensor;
    }

    /**
     * Allocates the next unique non-negative tensor identifier for this JVM.
     *
     * <p>Ordinary candidates from zero through {@code Long.MAX_VALUE - 1} are linearized by a
     * successful compare-and-set that advances the counter. At {@code Long.MAX_VALUE}, a separate
     * compare-and-set lets exactly one caller claim the valid final candidate. Every later call
     * fails permanently. Allocation never wraps, reserves a negative sentinel, rolls back, or
     * reuses a consumed value.</p>
     *
     * @return a non-null newly allocated tensor identifier unique among this factory's allocations
     *     in the current JVM
     * @throws IllegalStateException if {@code Long.MAX_VALUE} was already claimed, with message
     *     {@code tensor identifier space exhausted}
     */
    private static TensorId nextTensorId() {
        while (true) {
            long candidate = NEXT_TENSOR_ID.get();
            if (candidate < Long.MAX_VALUE) {
                if (NEXT_TENSOR_ID.compareAndSet(candidate, candidate + 1)) {
                    return new TensorId(candidate);
                }
            } else if (MAXIMUM_TENSOR_ID_CLAIMED.compareAndSet(false, true)) {
                return new TensorId(Long.MAX_VALUE);
            } else {
                throw new IllegalStateException("tensor identifier space exhausted");
            }
        }
    }
}
