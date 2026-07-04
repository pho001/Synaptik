package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;

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
 * heap storage. Nested primitive-array import additionally infers exact carrier type and fully
 * static dense shape after validating the complete rectangular structure. Numeric values and raw
 * BFLOAT16 bits are copied unchanged; BOOL bytes are normalized to zero or one. Heap allocation
 * uses automatic-scope memory segments, so it
 * introduces no arena, close operation, external lifetime owner, or deterministic reclamation.
 * Descriptor-based creation and flat import do not build descriptors or resolve layouts. Nested
 * import is the bounded exception: it synthesizes only the exact fully static shape and canonical
 * dense-contiguous descriptor proved by the source structure. Constant creation is the other
 * bounded descriptor-construction path: exact primitive scalar overloads create rank-zero values,
 * while zeros, ones, and their like-shaped variants create independent dense-contiguous tensors
 * for fully static shapes. The factory does not otherwise infer descriptors, retain or expose
 * source/backing arrays, convert values, accept boxed or generic nested values, provide typed
 * tensor access, create provenance, allocate native or backend memory, or provide compiler,
 * runtime, or backend behavior. Deterministic population additionally supports non-empty INT32
 * and INT64 ranges plus strict or cyclic prefixes of the six exact primitive carriers. Those
 * methods synthesize only canonical dense leaf descriptors, copy their complete logical result,
 * and reuse flat import for final allocation, normalization, and identity assignment. Normal and
 * bounded continuous uniform and bounded integral random creation likewise produce eager copied
 * leaf data, but require a transient caller-owned {@link RandomGenerator}; the factory never
 * selects, seeds, retains, replaces, synchronizes, splits, or closes a random source. Integral
 * primitive bounds infer INT32 or INT64, so those overloads need no data-type or gradient input.
 * No random package, source service, seed API, or distribution abstraction is introduced. The
 * distribution-specific methods remain cohesive factory operations beside their package-private
 * sampling helper; they do not create an independently useful random-domain type, and the
 * explicit caller-owned generator already supplies source and seeding policy without another
 * model service.</p>
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
     * Creates a dense-contiguous tensor by copying a rectangular multidimensional primitive array.
     *
     * <p>The {@code Object} parameter is necessary because Java assigns a distinct runtime class
     * to every primitive-array rank and no finite overload family can represent arbitrary rank.
     * Runtime class metadata must prove rank two or greater and an ultimate carrier of
     * {@code double}, {@code float}, {@code short}, {@code int}, {@code long}, or {@code byte};
     * those carriers infer {@link DataType#FLOAT64}, {@link DataType#FLOAT32}, raw
     * {@link DataType#BFLOAT16}, {@link DataType#INT32}, {@link DataType#INT64}, or logical
     * {@link DataType#BOOL}, respectively. Boxed, generic, rank-one, and other primitive carriers
     * are rejected without conversion or defaulting.</p>
     *
     * <p>Every reachable branch is validated in depth-first row-major order for rectangular
     * lengths and non-null subarrays before a flat carrier, destination storage, or tensor
     * identifier is allocated. Diagnostic paths use zero-based bracket notation: {@code []} is
     * the root, {@code [1]} its second child, and {@code [1][2]} that child's third child. A
     * zero-length final primitive axis is valid, but a zero-length earlier axis is rejected because
     * its trailing extents cannot be observed. The inferred shape is fully static and its layout
     * is canonical row-major dense-contiguous geometry. Values are flattened into a fresh matching
     * primitive array in row-major encounter order and delegated to exactly one matching flat
     * import overload. Numeric values and raw BFLOAT16 bits remain unchanged; BOOL bytes are
     * copied raw into the intermediate carrier, then flat import normalizes zero to zero and every
     * non-zero byte to one.</p>
     *
     * <p>No source array or intermediate flat carrier is retained or mutated. Later caller
     * mutation cannot affect the returned tensor. Inspection and flattening are not synchronized
     * with caller mutation, so callers must not mutate any source level concurrently and no atomic
     * deep-snapshot guarantee is provided. The descriptor enforces gradient eligibility after the
     * intermediate flat array has been allocated and populated. Structural, carrier,
     * element-count, and gradient-eligibility failures allocate no destination and consume no
     * identifier. A blank label reaches flat import after inference and flattening, allocates
     * destination storage and consumes an identifier before existing Tensor validation rejects
     * it. Identifier exhaustion likewise occurs after destination allocation and before the final
     * flat-to-storage copy. Neither failure exposes or retains the intermediate carrier.</p>
     *
     * @param source non-null runtime array with declared rank at least two, a supported ultimate
     *     primitive carrier, rectangular non-null subarrays, and no empty non-final axis; ownership
     *     remains with the caller
     * @param label non-null optional diagnostic label delegated to flat import; empty means absent,
     *     while present text is normalized and validated by {@link Tensor}
     * @param requiresGrad whether the inferred descriptor requests model-level gradient
     *     eligibility; true is valid only for the three inferred floating data types
     * @return a non-null fresh tensor with inferred exact data type, fully static shape,
     *     dense-contiguous layout, copied values, factory-assigned identity, and optional label
     * @throws NullPointerException if {@code source} or {@code label} is null, checked in that order
     *     with the parameter name as the message; these failures consume no identifier
     * @throws IllegalArgumentException if the source is not an array, with message
     *     {@code nested tensor source must be an array: actual=<runtimeClassName>}; has rank below
     *     two, with message {@code nested tensor source must have rank at least 2: actual=<rank>};
     *     has an unsupported ultimate carrier, with message
     *     {@code nested tensor source leaf carrier is unsupported: <componentTypeName>}; contains
     *     a null subarray, with message
     *     {@code nested tensor source contains null subarray at path <path>}; is ragged, with
     *     message {@code nested tensor source is ragged at axis <axis>, path <path>:
     *     expected=<expected>, actual=<actual>}; has an empty non-final axis, with message
     *     {@code nested tensor source cannot infer dimensions after empty axis <axis> at path
     *     <path>}; requires more than {@link Integer#MAX_VALUE} flat elements, with message
     *     {@code nested tensor element count exceeds Java array limit: required=<required>,
     *     maximum=2147483647}; requests gradients for a non-differentiable inferred type, with the
     *     {@link TensorDescriptor} eligibility message; or delegated {@link Tensor} validation
     *     rejects a blank label with message {@code label must not be blank}
     * @throws ArithmeticException if checked inferred element-count or dense-layout arithmetic
     *     overflows
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}
     * @throws OutOfMemoryError if the JVM cannot allocate the intermediate flat array or
     *     destination heap array
     */
    public static Tensor fromNestedArray(
            Object source,
            Optional<String> label,
            boolean requiresGrad) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(label, "label");
        return NestedTensorArray.importArray(source, label, requiresGrad);
    }

    /**
     * Creates an independent rank-zero {@link DataType#FLOAT64} tensor from one binary64 value.
     *
     * <p>The value is copied with its exact raw binary64 representation into newly allocated
     * dense-contiguous heap storage. The label is validated before descriptor, source carrier,
     * destination, or identifier allocation. Descriptor validation precedes source-carrier
     * allocation; a blank present label is rejected later by {@link Tensor}, after destination
     * and identifier allocation, and consumes that identifier.</p>
     *
     * <p>The descriptor uses the canonical shared {@link Shape#scalar()} value, but the returned
     * Tensor, descriptor, layout, storage wrapper, backing array, and identifier are new and are
     * not retained by the factory. The one-element source carrier exists before destination
     * allocation. Identifier exhaustion is therefore observed after both arrays exist and before
     * population; no identifier is rolled back. A source- or destination-array allocation failure
     * before identifier allocation propagates without consuming an identifier.</p>
     *
     * @param value binary64 value to store exactly, including signed zero, infinities, and NaNs
     * @param label non-null optional diagnostic label; present text is normalized by {@code Tensor}
     * @param requiresGrad whether the differentiable scalar requests model-level gradient eligibility
     * @return a non-null fresh rank-zero FLOAT64 tensor with independent dense storage and identity
     * @throws NullPointerException if {@code label} is null, with message {@code label}, before allocation
     * @throws IllegalArgumentException if delegated Tensor validation rejects a blank label, with
     *     message {@code label must not be blank}; the allocated identifier is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}; source and destination arrays have already
     *     been allocated
     * @throws ArithmeticException if checked scalar layout arithmetic unexpectedly overflows
     * @throws OutOfMemoryError if source or destination heap allocation fails
     */
    public static Tensor scalar(
            double value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(label, "label");
        return TensorConstants.scalar(value, label, requiresGrad);
    }

    /**
     * Creates an independent rank-zero {@link DataType#FLOAT32} tensor from one binary32 value.
     *
     * <p>The exact raw binary32 representation is copied into new dense-contiguous heap storage.
     * No widening, narrowing, or default data type is involved. Null-label and later allocation,
     * blank-label, identifier, memory, ownership, and non-retention semantics match
     * {@link #scalar(double, Optional, boolean)}.</p>
     *
     * @param value binary32 value to store exactly, including signed zero, infinities, and NaNs
     * @param label non-null optional diagnostic label; present text is normalized by {@code Tensor}
     * @param requiresGrad whether the differentiable scalar requests model-level gradient eligibility
     * @return a non-null fresh rank-zero FLOAT32 tensor with independent dense storage and identity
     * @throws NullPointerException if {@code label} is null, with message {@code label}, before allocation
     * @throws IllegalArgumentException if delegated Tensor validation rejects a blank label, with
     *     message {@code label must not be blank}; the allocated identifier is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after source and destination allocation
     * @throws ArithmeticException if checked scalar layout arithmetic unexpectedly overflows
     * @throws OutOfMemoryError if source or destination heap allocation fails
     */
    public static Tensor scalar(
            float value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(label, "label");
        return TensorConstants.scalar(value, label, requiresGrad);
    }

    /**
     * Creates an independent rank-zero {@link DataType#BFLOAT16} tensor by converting binary32 input.
     *
     * <p>{@code value} is converted explicitly by {@link BFloat16Bits#fromFloat(float)} with
     * round-to-nearest, ties-to-even BFLOAT16 semantics; signed zero and infinities are preserved
     * and NaN is canonicalized. The resulting raw {@code short} bits are copied into new dense
     * storage. Null-label and later allocation, blank-label, identifier, ownership, and memory side
     * effects match
     * {@link #scalar(double, Optional, boolean)}.</p>
     *
     * @param value binary32 semantic value to round to BFLOAT16
     * @param label non-null optional diagnostic label; present text is normalized by {@code Tensor}
     * @param requiresGrad whether the differentiable scalar requests model-level gradient eligibility
     * @return a non-null fresh rank-zero BFLOAT16 tensor with independent dense storage and identity
     * @throws NullPointerException if {@code label} is null, with message {@code label}, before allocation
     * @throws IllegalArgumentException if delegated Tensor validation rejects a blank label, with
     *     message {@code label must not be blank}; the allocated identifier is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after source and destination allocation
     * @throws ArithmeticException if checked scalar layout arithmetic unexpectedly overflows
     * @throws OutOfMemoryError if source or destination heap allocation fails
     */
    public static Tensor scalarBFloat16(
            float value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(label, "label");
        return TensorConstants.scalarBFloat16(value, label, requiresGrad);
    }

    /**
     * Creates an independent rank-zero {@link DataType#INT32} tensor from one exact signed value.
     *
     * <p>No conversion or rounding occurs. Because INT32 is not differentiable, a true gradient
     * request fails during descriptor validation before source-carrier, destination, or identifier
     * allocation. Other side effects match {@link #scalar(double, Optional, boolean)}.</p>
     *
     * @param value exact signed 32-bit value to store
     * @param label non-null optional diagnostic label; present text is normalized by {@code Tensor}
     * @param requiresGrad must be false because INT32 is not differentiable
     * @return a non-null fresh rank-zero INT32 tensor with independent dense storage and identity
     * @throws NullPointerException if {@code label} is null, with message {@code label}, before allocation
     * @throws IllegalArgumentException if gradients are requested, with the
     *     {@link TensorDescriptor} eligibility message before source or destination allocation, or
     *     if the label is blank after allocation; only the label failure consumes an identifier
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after source and destination allocation
     * @throws ArithmeticException if checked scalar layout arithmetic unexpectedly overflows
     * @throws OutOfMemoryError if source or destination heap allocation fails
     */
    public static Tensor scalar(
            int value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(label, "label");
        return TensorConstants.scalar(value, label, requiresGrad);
    }

    /**
     * Creates an independent rank-zero {@link DataType#INT64} tensor from one exact signed value.
     *
     * <p>No conversion or rounding occurs. A true gradient request fails before source-carrier,
     * destination, or identifier allocation because INT64 is not differentiable. Other side
     * effects match {@link #scalar(double, Optional, boolean)}.</p>
     *
     * @param value exact signed 64-bit value to store
     * @param label non-null optional diagnostic label; present text is normalized by {@code Tensor}
     * @param requiresGrad must be false because INT64 is not differentiable
     * @return a non-null fresh rank-zero INT64 tensor with independent dense storage and identity
     * @throws NullPointerException if {@code label} is null, with message {@code label}, before allocation
     * @throws IllegalArgumentException if gradients are requested, with the
     *     {@link TensorDescriptor} eligibility message before source or destination allocation, or
     *     if the label is blank after allocation; only the label failure consumes an identifier
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after source and destination allocation
     * @throws ArithmeticException if checked scalar layout arithmetic unexpectedly overflows
     * @throws OutOfMemoryError if source or destination heap allocation fails
     */
    public static Tensor scalar(
            long value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(label, "label");
        return TensorConstants.scalar(value, label, requiresGrad);
    }

    /**
     * Creates an independent rank-zero {@link DataType#BOOL} tensor from one semantic boolean.
     *
     * <p>{@code false} is stored as canonical byte zero and {@code true} as canonical byte one.
     * This method accepts no numeric truthiness conversion. A true gradient request fails before
     * source-carrier, destination, or identifier allocation because BOOL is not differentiable.
     * Other side effects match {@link #scalar(double, Optional, boolean)}.</p>
     *
     * @param value semantic boolean value to store canonically
     * @param label non-null optional diagnostic label; present text is normalized by {@code Tensor}
     * @param requiresGrad must be false because BOOL is not differentiable
     * @return a non-null fresh rank-zero BOOL tensor with independent dense storage and identity
     * @throws NullPointerException if {@code label} is null, with message {@code label}, before allocation
     * @throws IllegalArgumentException if gradients are requested, with the
     *     {@link TensorDescriptor} eligibility message before source or destination allocation, or
     *     if the label is blank after allocation; only the label failure consumes an identifier
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after source and destination allocation
     * @throws ArithmeticException if checked scalar layout arithmetic unexpectedly overflows
     * @throws OutOfMemoryError if source or destination heap allocation fails
     */
    public static Tensor scalar(
            boolean value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(label, "label");
        return TensorConstants.scalar(value, label, requiresGrad);
    }

    /**
     * Creates an independent dense-contiguous tensor filled with each data type's raw zero value.
     *
     * <p>The shape must be fully static; scalar and zero-sized shapes are supported. A canonical
     * layout and descriptor are synthesized from the supplied logical facts. Storage is created
     * directly through {@link #allocate(TensorDescriptor, Optional)}, relying on JVM primitive-array
     * zero initialization without a source array, fill loop, or copy. Shape/count/layout/gradient
     * validation precedes destination and identifier allocation.</p>
     *
     * <p>Each successful call returns a new Tensor, descriptor, layout, storage wrapper, backing
     * array, and identifier; no result aliases another factory result. A blank label is rejected
     * after destination and identifier allocation and consumes that identifier. Exhaustion is
     * observed after destination allocation. Because this path has no source carrier, JVM
     * destination-allocation failure occurs before identifier allocation and consumes no ID.</p>
     *
     * @param shape non-null fully static logical shape, retained by the new descriptor
     * @param dataType non-null exact element type; no default or conversion is applied
     * @param label non-null optional diagnostic label; present text is normalized by {@code Tensor}
     * @param requiresGrad whether model-level gradient eligibility is requested; valid only for a differentiable type
     * @return a non-null fresh tensor with a new dense descriptor, zeroed storage, and identity
     * @throws NullPointerException if {@code shape}, {@code dataType}, or {@code label} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if {@code shape} is dynamic, with message
     *     {@code constant tensor creation requires a fully static shape: <shape>}; logical count
     *     exceeds the Java array limit, with message {@code constant tensor element count exceeds
     *     Java array limit: required=<required>, maximum=2147483647}; gradients are requested for
     *     a non-differentiable type with the {@link TensorDescriptor} eligibility message; or the
     *     label is blank with message {@code label must not be blank}; only the label failure
     *     occurs after destination and ID allocation and consumes that ID
     * @throws ArithmeticException if checked element-count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after destination allocation
     * @throws OutOfMemoryError if destination heap allocation fails
     */
    public static Tensor zeros(
            Shape shape,
            DataType dataType,
            Optional<String> label,
            boolean requiresGrad) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(label, "label");
        return TensorConstants.zeros(shape, dataType, label, requiresGrad);
    }

    /**
     * Creates an independent dense-contiguous tensor filled with the exact typed value one.
     *
     * <p>The fully static shape may be scalar or zero-sized. After descriptor validation, one
     * matching primitive carrier is allocated and filled, then delegated to exactly one typed flat
     * import overload. BFLOAT16 one uses the converted raw representation {@code 0x3F80}; BOOL one
     * uses canonical byte one. FLOAT64, FLOAT32, INT32, and INT64 use {@code 1.0d}, {@code 1.0f},
     * {@code 1}, and {@code 1L}, respectively. The source carrier is not retained.</p>
     *
     * <p>Each successful call returns a new Tensor, descriptor, layout, storage wrapper, backing
     * array, and identifier. A blank label is rejected after source-carrier, destination, and ID
     * allocation and consumes that identifier. Exhaustion is observed after both arrays exist.
     * A source- or destination-array allocation failure before ID allocation consumes no ID.</p>
     *
     * @param shape non-null fully static logical shape, retained by the new descriptor
     * @param dataType non-null exact element type; no default or conversion is applied
     * @param label non-null optional diagnostic label; present text is normalized by {@code Tensor}
     * @param requiresGrad whether model-level gradient eligibility is requested; valid only for a differentiable type
     * @return a non-null fresh tensor with a new dense descriptor, one-filled storage, and identity
     * @throws NullPointerException if {@code shape}, {@code dataType}, or {@code label} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if {@code shape} is dynamic, with message
     *     {@code constant tensor creation requires a fully static shape: <shape>}; logical count
     *     exceeds the Java array limit, with message {@code constant tensor element count exceeds
     *     Java array limit: required=<required>, maximum=2147483647}; gradients are requested for
     *     a non-differentiable type with the {@link TensorDescriptor} eligibility message; or the
     *     label is blank with message {@code label must not be blank}; only the label failure
     *     occurs after source, destination, and ID allocation and consumes that ID
     * @throws ArithmeticException if checked element-count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after source and destination allocation
     * @throws OutOfMemoryError if source or destination heap allocation fails
     */
    public static Tensor ones(
            Shape shape,
            DataType dataType,
            Optional<String> label,
            boolean requiresGrad) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(label, "label");
        return TensorConstants.ones(shape, dataType, label, requiresGrad);
    }

    /**
     * Creates an independent zero tensor using only a template's shape and data type.
     *
     * <p>The template's label, identity, gradient request, layout, storage, liveness, and all other
     * state are neither copied nor retained. Its fully static shape and data type seed a new
     * canonical dense descriptor; the explicit label and gradient request govern the result.
     * Static unresolved, dense, offset, strided, and broadcast template layouts are accepted
     * because no template layout or storage is inspected.</p>
     *
     * <p>Successful creation returns new Tensor, descriptor, layout, storage wrapper, backing
     * array, and identifier objects; only the template's immutable shape and data-type values are
     * used. Validation and failure side effects are exactly those of
     * {@link #zeros(Shape, DataType, Optional, boolean)} after the template and label null checks.</p>
     *
     * @param template non-null tensor whose immutable shape and data type alone are reused
     * @param label non-null optional diagnostic label for the result, never inherited
     * @param requiresGrad explicit result gradient request, never inherited and valid only for floating types
     * @return a non-null fresh independent dense zero tensor with new descriptor, storage, and identity
     * @throws NullPointerException if {@code template} or {@code label} is null, checked in that order
     * @throws IllegalArgumentException if the template shape is dynamic, logical count exceeds the
     *     Java array limit, gradient request is ineligible, or result label is blank; only blank
     *     label failure occurs after destination and ID allocation and consumes that ID
     * @throws ArithmeticException if checked element-count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after destination allocation
     * @throws OutOfMemoryError if destination heap allocation fails
     */
    public static Tensor zerosLike(
            Tensor template, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(template, "template");
        Objects.requireNonNull(label, "label");
        TensorDescriptor templateDescriptor = template.descriptor();
        return TensorConstants.zeros(
                templateDescriptor.shape(), templateDescriptor.dataType(), label, requiresGrad);
    }

    /**
     * Creates an independent one tensor using only a template's shape and data type.
     *
     * <p>The template's label, identity, gradient request, layout, storage, liveness, and all other
     * state are neither copied nor retained. Its fully static shape and data type seed a new
     * canonical dense descriptor; one matching typed carrier is filled and copied, and the
     * explicit label and gradient request govern the result. Static unresolved, dense, offset,
     * strided, and broadcast template layouts are accepted because no template layout or storage
     * is inspected.</p>
     *
     * <p>Successful creation returns new Tensor, descriptor, layout, storage wrapper, backing
     * array, and identifier objects; only the template's immutable shape and data-type values are
     * used. Validation and failure side effects are exactly those of
     * {@link #ones(Shape, DataType, Optional, boolean)} after the template and label null checks.</p>
     *
     * @param template non-null tensor whose immutable shape and data type alone are reused
     * @param label non-null optional diagnostic label for the result, never inherited
     * @param requiresGrad explicit result gradient request, never inherited and valid only for floating types
     * @return a non-null fresh independent dense one tensor with new descriptor, storage, and identity
     * @throws NullPointerException if {@code template} or {@code label} is null, checked in that order
     * @throws IllegalArgumentException if the template shape is dynamic, logical count exceeds the
     *     Java array limit, gradient request is ineligible, or result label is blank; only blank
     *     label failure occurs after source, destination, and ID allocation and consumes that ID
     * @throws ArithmeticException if checked element-count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after source and destination allocation
     * @throws OutOfMemoryError if source or destination heap allocation fails
     */
    public static Tensor onesLike(
            Tensor template, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(template, "template");
        Objects.requireNonNull(label, "label");
        TensorDescriptor templateDescriptor = template.descriptor();
        return TensorConstants.ones(
                templateDescriptor.shape(), templateDescriptor.dataType(), label, requiresGrad);
    }

    /**
     * Creates an independent dense floating tensor from normally distributed samples.
     *
     * <p>The shape must be fully static and its checked logical element count must fit a Java
     * array. The exact data type must be {@link DataType#FLOAT64}, {@link DataType#FLOAT32}, or
     * {@link DataType#BFLOAT16}. Both distribution parameters must be finite, and
     * {@code standardDeviation} must be numerically non-negative; positive and negative zero are
     * accepted. The result uses a new canonical dense-contiguous descriptor with the supplied
     * gradient intent and optional label.</p>
     *
     * <p>For every logical element in row-major order, this method invokes
     * {@link RandomGenerator#nextGaussian()} exactly once and evaluates ordinary binary64
     * {@code mean + gaussian * standardDeviation}, multiplication before addition and without
     * fused multiply-add substitution. FLOAT64 stores that binary64 result directly. FLOAT32
     * narrows it once to binary32. BFLOAT16 first narrows it to binary32 and then applies
     * {@link BFloat16Bits#fromFloat(float)}. Generated overflow, underflow, signed zero, infinity,
     * or NaN is retained according to those conversions rather than post-validated. A
     * zero-element shape consumes no samples; a scalar consumes one.</p>
     *
     * <p>The caller creates, configures, seeds, owns, and advances the exact random generator.
     * Neither this factory nor the returned tensor retains or substitutes it, and the call does
     * not synchronize, reset, split, or close it. Equivalent output is bounded to equivalent
     * generator implementations and initial states, identical arguments, and no interfering
     * source use; no cross-algorithm, provider, Java-version, seed-expansion, concurrent-use, or
     * global reproducibility promise is made.</p>
     *
     * <p>Null, shape, count, type, parameter, layout, and descriptor failures occur before source
     * allocation, sampling, destination allocation, or identifier allocation. Source-carrier
     * allocation failure occurs before sampling. If the generator throws, preceding calls remain
     * consumed according to generator behavior, but no destination or identifier exists. After
     * all samples are produced, exactly one matching flat import allocates destination storage and
     * then an identifier. A blank label therefore consumes all samples, both carriers, and one
     * identifier before delegated Tensor validation fails. Identifier exhaustion consumes all
     * samples and both carrier allocations but performs no flat copy; no source or identifier
     * state is rolled back.</p>
     *
     * @param shape non-null fully static result shape; scalar and zero-element shapes are valid
     * @param dataType non-null exact floating output type, limited to FLOAT64, FLOAT32, or BFLOAT16
     * @param mean finite binary64 mean applied to every sampled Gaussian value
     * @param standardDeviation finite numerically non-negative binary64 standard deviation;
     *     either signed zero is accepted and still consumes one sample per element
     * @param randomGenerator non-null transient caller-owned source; it is never retained,
     *     substituted, seeded, reset, split, synchronized, or closed
     * @param label non-null optional diagnostic label; present text is normalized and validated by
     *     {@link Tensor} after sampling and destination/identifier allocation
     * @param requiresGrad whether model-level gradient eligibility is requested for the result
     * @return a non-null fresh dense tensor containing independently copied converted samples,
     *     with new storage and factory-assigned identity
     * @throws NullPointerException if {@code shape}, {@code dataType}, {@code randomGenerator}, or
     *     {@code label} is null, checked in that order with the parameter name as message; no
     *     sample or identifier is consumed
     * @throws IllegalArgumentException if the shape is dynamic; its logical count exceeds
     *     {@link Integer#MAX_VALUE}; the data type is not one of the three floating types; the
     *     mean is non-finite; the standard deviation is non-finite or numerically negative; or
     *     delegated Tensor validation rejects a blank label. Only the blank-label failure occurs
     *     after sampling, destination allocation, and identifier allocation
     * @throws ArithmeticException if checked logical-count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after sampling and both carrier allocations
     * @throws OutOfMemoryError if source or destination carrier allocation fails; source allocation
     *     failure consumes no samples or identifier, while destination failure follows sampling
     */
    public static Tensor randomNormal(
            Shape shape,
            DataType dataType,
            double mean,
            double standardDeviation,
            RandomGenerator randomGenerator,
            Optional<String> label,
            boolean requiresGrad) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomGenerator, "randomGenerator");
        Objects.requireNonNull(label, "label");
        return TensorRandoms.randomNormal(
                shape,
                dataType,
                mean,
                standardDeviation,
                randomGenerator,
                label,
                requiresGrad);
    }

    /**
     * Creates an independent dense floating tensor from bounded continuous uniform samples.
     *
     * <p>The shape must be fully static and its checked logical element count must fit a Java
     * array. The exact data type must be {@link DataType#FLOAT64}, {@link DataType#FLOAT32}, or
     * {@link DataType#BFLOAT16}. Both bounds must be finite and the binary64
     * {@code lowerBoundInclusive} must be strictly less than {@code upperBoundExclusive}. The
     * result uses a new canonical dense-contiguous descriptor with the supplied gradient intent
     * and optional label.</p>
     *
     * <p>For every logical element in row-major order, this method invokes
     * {@link RandomGenerator#nextDouble(double, double)} exactly once with the supplied bounds.
     * The conforming generator's binary64 result is in the half-open interval
     * {@code [lowerBoundInclusive, upperBoundExclusive)}. FLOAT64 stores that value directly.
     * FLOAT32 narrows it once to binary32. BFLOAT16 first narrows it to binary32 and then applies
     * {@link BFloat16Bits#fromFloat(float)}. Narrowing may round a stored FLOAT32 or BFLOAT16 value
     * to the corresponding narrowed upper bound or to a lower-rounded representable value; the
     * half-open promise applies to the generator's binary64 result, not to the narrowed carrier.
     * A custom non-conforming generator result is not post-validated. A zero-element shape makes
     * no source calls; a scalar makes one.</p>
     *
     * <p>The caller creates, configures, seeds, owns, and advances the exact random generator.
     * Neither this factory nor the returned tensor retains or substitutes it, and the call does
     * not synchronize, reset, split, or close it. Equivalent output is bounded to equivalent
     * generator implementations and initial states, identical arguments, and no interfering
     * source use; no cross-algorithm, provider, Java-version, seed-expansion, concurrent-use, or
     * global reproducibility promise is made.</p>
     *
     * <p>Validation after the public null checks is deterministic. Dynamic shape reports
     * {@code uniform random tensor creation requires a fully static shape: <shape>}; an
     * over-limit count reports {@code uniform random tensor element count exceeds Java array
     * limit: required=<required>, maximum=2147483647}; a non-floating type reports
     * {@code uniform random creation requires floating data type: <dataType>}; non-finite bounds
     * report {@code uniform random lower bound must be finite: <lower>} or
     * {@code uniform random upper bound must be finite: <upper>}; and unordered bounds report
     * {@code uniform random lower bound must be less than upper bound: lower=<lower>,
     * upper=<upper>}. Checked count or layout overflow remains an {@link ArithmeticException},
     * and descriptor construction remains authoritative for gradient eligibility.</p>
     *
     * <p>Null, shape, count, type, bound, layout, and descriptor failures occur before source
     * allocation, sampling, destination allocation, or identifier allocation. Source-carrier
     * allocation failure occurs before sampling. If the generator throws, preceding calls remain
     * consumed according to generator behavior, but no destination or identifier exists. After
     * all samples are produced, exactly one matching flat import allocates destination storage and
     * then an identifier. A blank label therefore consumes all samples, both carriers, and one
     * identifier before delegated Tensor validation fails. Identifier exhaustion consumes all
     * samples and both carrier allocations but performs no flat copy. Source advancement, array
     * allocation, storage construction, and identifier allocation are never rolled back.</p>
     *
     * @param shape non-null fully static result shape; scalar and zero-element shapes are valid
     * @param dataType non-null exact floating output type, limited to FLOAT64, FLOAT32, or BFLOAT16
     * @param lowerBoundInclusive finite inclusive binary64 lower bound
     * @param upperBoundExclusive finite exclusive binary64 upper bound, strictly greater than the
     *     lower bound
     * @param randomGenerator non-null transient caller-owned source; it is never retained,
     *     substituted, seeded, reset, split, synchronized, or closed
     * @param label non-null optional diagnostic label; present text is normalized and validated by
     *     {@link Tensor} after sampling and destination/identifier allocation
     * @param requiresGrad whether model-level gradient eligibility is requested for the result
     * @return a non-null fresh dense tensor containing independently copied converted samples,
     *     with new storage and factory-assigned identity
     * @throws NullPointerException if {@code shape}, {@code dataType}, {@code randomGenerator}, or
     *     {@code label} is null, checked in that order with the parameter name as message; no
     *     sample or identifier is consumed
     * @throws IllegalArgumentException if the shape is dynamic; its logical count exceeds
     *     {@link Integer#MAX_VALUE}; the data type is not one of the three floating types; either
     *     bound is non-finite; the lower bound is not strictly less than the upper bound; or
     *     delegated Tensor validation rejects a blank label. Only the blank-label failure occurs
     *     after sampling, destination allocation, and identifier allocation
     * @throws ArithmeticException if checked logical-count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after sampling and both carrier allocations
     * @throws OutOfMemoryError if source or destination carrier allocation fails; source allocation
     *     failure consumes no samples or identifier, while destination failure follows sampling
     */
    public static Tensor randomUniform(
            Shape shape,
            DataType dataType,
            double lowerBoundInclusive,
            double upperBoundExclusive,
            RandomGenerator randomGenerator,
            Optional<String> label,
            boolean requiresGrad) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomGenerator, "randomGenerator");
        Objects.requireNonNull(label, "label");
        return TensorRandoms.randomUniform(
                shape,
                dataType,
                lowerBoundInclusive,
                upperBoundExclusive,
                randomGenerator,
                label,
                requiresGrad);
    }

    /**
     * Creates an independent dense INT32 tensor from bounded integral samples.
     *
     * <p>The shape must be fully static and its checked logical element count must fit a Java
     * array. {@code originInclusive} must be strictly less than {@code boundExclusive}; the result
     * uses a new canonical dense-contiguous descriptor with gradients disabled and the supplied
     * optional label. The primitive bounds select {@link DataType#INT32} directly, so no data-type,
     * gradient, rounding, narrowing, widening, or conversion policy is involved.</p>
     *
     * <p>For every logical element in row-major order, this method invokes
     * {@link RandomGenerator#nextInt(int, int)} exactly once with the unchanged bounds and stores
     * that result directly in one {@code int[]} carrier. A conforming generator returns each value
     * in the half-open interval {@code [originInclusive, boundExclusive)} without project-owned
     * modulo arithmetic or bias. A custom non-conforming result is not post-validated. A
     * zero-element shape makes no source call, while a scalar makes one.</p>
     *
     * <p>The same-carrier exclusive bound cannot express the mathematical value one greater than
     * {@link Integer#MAX_VALUE}. Consequently, {@code Integer.MAX_VALUE} may be supplied as the
     * exclusive bound but is never emitted, and this overload does not provide an unbounded or
     * full-domain alternative. Negative and mixed-sign intervals remain valid when strictly
     * ordered.</p>
     *
     * <p>Validation after the public null checks is deterministic. Dynamic shape reports
     * {@code integral random tensor creation requires a fully static shape: <shape>}; an
     * over-limit count reports {@code integral random tensor element count exceeds Java array
     * limit: required=<required>, maximum=2147483647}; and unordered bounds report
     * {@code integral random origin must be less than bound: origin=<origin>, bound=<bound>}.
     * Checked count or layout overflow remains an {@link ArithmeticException}.</p>
     *
     * <p>The caller creates, configures, seeds, owns, and advances the exact random generator.
     * Neither this factory nor the returned tensor retains or substitutes it, and the call does
     * not synchronize, reset, split, or close it. Equivalent output is bounded to equivalent
     * generator implementations and initial states, identical arguments, and no interfering
     * source use; no cross-algorithm, provider, Java-version, seed-expansion, concurrent-use, or
     * global reproducibility promise is made. The one-call promise concerns the bounded generator
     * method, not the source's internal random-bit consumption.</p>
     *
     * <p>Null, shape, count, bound, layout, and descriptor failures occur before carrier
     * allocation, sampling, destination allocation, or identifier allocation. Source-carrier
     * allocation failure therefore consumes no source call or identifier. If the generator
     * throws, preceding calls remain consumed according to generator behavior, but no destination
     * or identifier exists. After all samples are produced, exactly one matching flat import
     * allocates destination storage and then an identifier. A destination-allocation failure
     * follows all source calls but precedes identifier allocation. A blank label consumes all
     * source calls, both carriers, and one identifier before delegated Tensor validation fails;
     * exhaustion likewise follows all calls and allocations. No source, allocation, or identifier
     * state is rolled back.</p>
     *
     * @param shape non-null fully static result shape; scalar and zero-element shapes are valid
     * @param originInclusive inclusive signed 32-bit lower bound
     * @param boundExclusive exclusive signed 32-bit upper bound, strictly greater than the origin
     * @param randomGenerator non-null transient caller-owned source; it is never retained,
     *     substituted, seeded, reset, split, synchronized, or closed
     * @param label non-null optional diagnostic label; present text is normalized and validated by
     *     {@link Tensor} after sampling and destination/identifier allocation
     * @return a non-null fresh dense INT32 tensor containing independently copied bounded samples,
     *     with gradients disabled, new storage, and factory-assigned identity
     * @throws NullPointerException if {@code shape}, {@code randomGenerator}, or {@code label} is
     *     null, checked in that order with the parameter name as message; no source call or
     *     identifier is consumed
     * @throws IllegalArgumentException if the shape is dynamic; its logical count exceeds
     *     {@link Integer#MAX_VALUE}; the origin is not strictly less than the bound; or delegated
     *     Tensor validation rejects a blank label. Only the blank-label failure occurs after
     *     sampling, destination allocation, and identifier allocation
     * @throws ArithmeticException if checked logical-count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after sampling and both carrier allocations
     * @throws OutOfMemoryError if source or destination carrier allocation fails; source allocation
     *     failure consumes no calls or identifier, while destination failure follows sampling
     */
    public static Tensor randomInt(
            Shape shape,
            int originInclusive,
            int boundExclusive,
            RandomGenerator randomGenerator,
            Optional<String> label) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(randomGenerator, "randomGenerator");
        Objects.requireNonNull(label, "label");
        return TensorRandoms.randomInt(
                shape, originInclusive, boundExclusive, randomGenerator, label);
    }

    /**
     * Creates an independent dense INT64 tensor from bounded integral samples.
     *
     * <p>This overload has the same static-shape, strict half-open interval, source ownership,
     * reproducibility, validation ordering, allocation, label, identifier, and no-rollback
     * contract as the INT32 overload. Primitive {@code long} bounds select
     * {@link DataType#INT64}; gradients are always disabled. Each row-major element is the direct
     * result of one {@link RandomGenerator#nextLong(long, long)} call stored in the sole
     * {@code long[]} source carrier, followed by exactly one matching flat import. No unbounded
     * draw, modulo reduction, floating arithmetic, conversion, stream, or alternate carrier is
     * used.</p>
     *
     * <p>The same-carrier exclusive bound cannot express the mathematical value one greater than
     * {@link Long#MAX_VALUE}. {@code Long.MAX_VALUE} may be the exclusive bound but is never
     * emitted; no unbounded or full-domain convenience is added. Negative and mixed-sign strictly
     * ordered intervals are supported. The generator is transient and never retained or managed,
     * and the one-call guarantee applies to bounded method invocations rather than internal source
     * bits.</p>
     *
     * <p>Validation uses the same public null order and the same shape, count, and bound messages
     * as the INT32 overload, with the supplied signed 64-bit bound values. Pre-sampling failures
     * and source-carrier allocation failure consume no calls or identifier. A source exception
     * leaves preceding calls consumed but creates no destination or identifier. Destination
     * allocation follows all calls and precedes identifier allocation; a blank label consumes all
     * calls and one identifier, while exhaustion follows all calls and both carrier allocations.
     * No source, allocation, or identifier state is rolled back.</p>
     *
     * @param shape non-null fully static result shape; scalar and zero-element shapes are valid
     * @param originInclusive inclusive signed 64-bit lower bound
     * @param boundExclusive exclusive signed 64-bit upper bound, strictly greater than the origin
     * @param randomGenerator non-null transient caller-owned source; it is never retained,
     *     substituted, seeded, reset, split, synchronized, or closed
     * @param label non-null optional diagnostic label; present text is normalized and validated by
     *     {@link Tensor} after sampling and destination/identifier allocation
     * @return a non-null fresh dense INT64 tensor containing independently copied bounded samples,
     *     with gradients disabled, new storage, and factory-assigned identity
     * @throws NullPointerException if {@code shape}, {@code randomGenerator}, or {@code label} is
     *     null, checked in that order with the parameter name as message; no source call or
     *     identifier is consumed
     * @throws IllegalArgumentException if the shape is dynamic; its logical count exceeds
     *     {@link Integer#MAX_VALUE}; the origin is not strictly less than the bound; or delegated
     *     Tensor validation rejects a blank label. Only the blank-label failure occurs after
     *     sampling, destination allocation, and identifier allocation
     * @throws ArithmeticException if checked logical-count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after sampling and both carrier allocations
     * @throws OutOfMemoryError if source or destination carrier allocation fails; source allocation
     *     failure consumes no calls or identifier, while destination failure follows sampling
     */
    public static Tensor randomInt(
            Shape shape,
            long originInclusive,
            long boundExclusive,
            RandomGenerator randomGenerator,
            Optional<String> label) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(randomGenerator, "randomGenerator");
        Objects.requireNonNull(label, "label");
        return TensorRandoms.randomInt(
                shape, originInclusive, boundExclusive, randomGenerator, label);
    }

    /**
     * Creates a non-empty dense INT32 range with inclusive start and exclusive end.
     *
     * <p>A positive step requires ascending bounds and a negative step requires descending bounds;
     * zero step, equal bounds, and a direction that cannot reach the end are rejected. The final
     * emitted value may be less than one step from {@code endExclusive}. Exact sizing prevents
     * primitive subtraction, absolute-value, and ceiling-division overflow. Population advances
     * only when another value remains, so a valid primitive-boundary range cannot fail from an
     * unused addition after its final value.</p>
     *
     * <p>The method creates one rank-one canonical dense descriptor with {@link DataType#INT32}
     * and {@code requiresGrad == false}, fills one exact {@code int[]} carrier, and delegates once
     * to matching flat import. The temporary carrier is not retained. A null label or invalid
     * range fails before carrier, destination, or identifier allocation. A blank present label
     * fails after carrier and destination allocation and consumes its allocated identifier without
     * copying. Identifier exhaustion is observed after both arrays exist and before copying; no
     * failed identifier is rolled back.</p>
     *
     * @param startInclusive first emitted signed 32-bit value
     * @param endExclusive exclusive signed 32-bit bound, which is never emitted
     * @param step non-zero signed increment that must advance toward {@code endExclusive}
     * @param label non-null optional diagnostic label; empty means absent and present text is
     *     normalized and validated by {@link Tensor}
     * @return a non-null fresh rank-one dense INT32 tensor containing the encounter-ordered range
     *     in independent writable heap storage
     * @throws NullPointerException if {@code label} is {@code null}, with message {@code label};
     *     this check runs before all range validation and consumes no identifier
     * @throws IllegalArgumentException if {@code step} is zero, the bounds are equal, the step
     *     direction cannot advance toward the end, the exact count exceeds
     *     {@link Integer#MAX_VALUE}, or delegated Tensor validation rejects a blank label; only the
     *     label failure occurs after allocation and consumes an identifier
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after carrier and destination allocation
     * @throws OutOfMemoryError if the source carrier or destination heap array cannot be allocated;
     *     an error before identifier allocation consumes no identifier
     */
    public static Tensor range(
            int startInclusive,
            int endExclusive,
            int step,
            Optional<String> label) {
        Objects.requireNonNull(label, "label");
        return TensorPopulations.range(startInclusive, endExclusive, step, label);
    }

    /**
     * Creates a non-empty dense INT64 range with inclusive start and exclusive end.
     *
     * <p>Positive and negative non-zero steps are supported only when their sign advances toward
     * the exclusive bound. Exact arbitrary-precision arithmetic is confined to count validation;
     * emitted values and storage remain signed 64-bit. The final emitted value may be less than one
     * step from the end, and population performs no unused post-final addition, including at
     * {@link Long#MIN_VALUE} and {@link Long#MAX_VALUE} boundaries.</p>
     *
     * <p>The result has a new rank-one canonical dense {@link DataType#INT64} descriptor with
     * gradients disabled. One exact {@code long[]} carrier is filled and delegated once to flat
     * import, which creates independent destination storage and identity; the carrier is not
     * retained. Validation, blank-label, exhaustion, allocation, and no-rollback side effects are
     * the same as the INT32 overload.</p>
     *
     * @param startInclusive first emitted signed 64-bit value
     * @param endExclusive exclusive signed 64-bit bound, which is never emitted
     * @param step non-zero signed increment that must advance toward {@code endExclusive}
     * @param label non-null optional diagnostic label; empty means absent and present text is
     *     normalized and validated by {@link Tensor}
     * @return a non-null fresh rank-one dense INT64 tensor containing the encounter-ordered range
     *     in independent writable heap storage
     * @throws NullPointerException if {@code label} is {@code null}, with message {@code label};
     *     this check runs before all range validation and consumes no identifier
     * @throws IllegalArgumentException if {@code step} is zero, the bounds are equal, the step
     *     direction cannot advance toward the end, the exact count exceeds
     *     {@link Integer#MAX_VALUE}, or delegated Tensor validation rejects a blank label; only the
     *     label failure occurs after allocation and consumes an identifier
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after carrier and destination allocation
     * @throws OutOfMemoryError if the source carrier or destination heap array cannot be allocated;
     *     an error before identifier allocation consumes no identifier
     */
    public static Tensor range(
            long startInclusive,
            long endExclusive,
            long step,
            Optional<String> label) {
        Objects.requireNonNull(label, "label");
        return TensorPopulations.range(startInclusive, endExclusive, step, label);
    }

    /**
     * Creates a dense FLOAT64 tensor from the requested strict prefix of binary64 values.
     *
     * <p>The shape must be fully static. Its checked logical element count must fit a Java array,
     * and {@code source} must contain at least that many values. Exactly that prefix is copied into
     * one fresh exact-length {@code double[]} carrier; any source tail is ignored, and even an
     * equal-length source is not retained. A zero-element shape accepts an empty source. The
     * result uses a new canonical dense-contiguous {@link DataType#FLOAT64} descriptor with the
     * explicit gradient request, label, storage, and factory identity.</p>
     *
     * <p>Null checks run in {@code shape}, {@code label}, {@code source} order. Dynamic shape,
     * checked-count, Java-array-limit, source-sufficiency, dense-layout, and gradient-eligibility
     * validation complete before output-carrier, destination, or identifier allocation. The fresh
     * carrier is delegated once to matching flat import. A blank label fails after the carrier,
     * destination, and identifier exist but before the flat copy, consuming the identifier.
     * Exhaustion is observed at the same point before copying, and failures are not rolled back.</p>
     *
     * @param shape non-null fully static result shape; the immutable shape is retained by the new
     *     descriptor and may be scalar or contain a zero-sized dimension
     * @param label non-null optional diagnostic label; empty means absent and present text is
     *     normalized and validated by {@link Tensor}
     * @param requiresGrad whether model-level gradient eligibility is requested for FLOAT64
     * @param source non-null caller-owned binary64 values; it must contain at least the shape's
     *     logical element count and is never retained or mutated
     * @return a non-null fresh dense FLOAT64 tensor containing an independent copy of exactly the
     *     requested prefix
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is
     *     {@code null}, checked in that order with the parameter name as message
     * @throws IllegalArgumentException if the shape is dynamic, the count exceeds
     *     {@link Integer#MAX_VALUE}, the source is shorter than the count, descriptor construction
     *     rejects the gradient request, or delegated Tensor validation rejects a blank label; only
     *     the label failure consumes an identifier
     * @throws ArithmeticException if checked logical-count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}, after carrier and destination allocation
     * @throws OutOfMemoryError if the copied carrier or destination heap array cannot be allocated;
     *     an error before identifier allocation consumes no identifier
     */
    public static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, double[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromStrictFlatPrefix(shape, label, requiresGrad, source);
    }

    /**
     * Creates a dense FLOAT32 tensor from a copied strict prefix of binary32 values.
     *
     * <p>This overload has the same static-shape, exact-prefix, source non-retention, validation
     * order, allocation, label, identity, and no-rollback contract as the FLOAT64 overload. The
     * exact {@code float[]} carrier selects {@link DataType#FLOAT32}; no numeric conversion occurs.
     * A source tail is ignored, and a zero-element result requires no source values.</p>
     *
     * @param shape non-null fully static result shape retained by the new dense descriptor
     * @param label non-null optional diagnostic label normalized and validated by {@link Tensor}
     * @param requiresGrad whether model-level gradient eligibility is requested for FLOAT32
     * @param source non-null caller-owned binary32 source with at least the logical result count;
     *     the source is never retained or mutated
     * @return a non-null fresh dense FLOAT32 tensor containing an independent exact-prefix copy
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if static-shape, Java-array-limit, source-sufficiency,
     *     descriptor-gradient, or delegated label validation fails
     * @throws ArithmeticException if checked count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after allocation
     * @throws OutOfMemoryError if copied-carrier or destination allocation fails
     */
    public static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, float[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromStrictFlatPrefix(shape, label, requiresGrad, source);
    }

    /**
     * Creates a dense BFLOAT16 tensor from a copied strict prefix of raw bit patterns.
     *
     * <p>This overload follows the FLOAT64 overload's static-shape, prefix, ownership, validation,
     * allocation, and ID-side-effect contract. Each {@code short} is copied as raw
     * {@link DataType#BFLOAT16} bits without conversion or canonicalization; a source tail is
     * ignored and the source is never retained.</p>
     *
     * @param shape non-null fully static result shape retained by the new dense descriptor
     * @param label non-null optional diagnostic label normalized and validated by {@link Tensor}
     * @param requiresGrad whether model-level gradient eligibility is requested for BFLOAT16
     * @param source non-null caller-owned raw BFLOAT16 source with at least the logical result
     *     count; it is never retained or mutated
     * @return a non-null fresh dense BFLOAT16 tensor preserving an independent raw-bit prefix
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if static-shape, Java-array-limit, source-sufficiency,
     *     descriptor-gradient, or delegated label validation fails
     * @throws ArithmeticException if checked count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after allocation
     * @throws OutOfMemoryError if copied-carrier or destination allocation fails
     */
    public static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, short[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromStrictFlatPrefix(shape, label, requiresGrad, source);
    }

    /**
     * Creates a dense INT32 tensor from a copied strict prefix of signed values.
     *
     * <p>This overload follows the FLOAT64 overload's static-shape, prefix, ownership, validation,
     * allocation, and ID-side-effect contract. The exact {@code int[]} carrier selects
     * {@link DataType#INT32}; values are copied unchanged, and gradients must be disabled.</p>
     *
     * @param shape non-null fully static result shape retained by the new dense descriptor
     * @param label non-null optional diagnostic label normalized and validated by {@link Tensor}
     * @param requiresGrad must be false because INT32 is not differentiable
     * @param source non-null caller-owned signed source with at least the logical result count;
     *     it is never retained or mutated
     * @return a non-null fresh dense INT32 tensor containing an independent exact-prefix copy
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if static-shape, Java-array-limit, source-sufficiency,
     *     descriptor-gradient, or delegated label validation fails
     * @throws ArithmeticException if checked count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after allocation
     * @throws OutOfMemoryError if copied-carrier or destination allocation fails
     */
    public static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, int[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromStrictFlatPrefix(shape, label, requiresGrad, source);
    }

    /**
     * Creates a dense INT64 tensor from a copied strict prefix of signed values.
     *
     * <p>This overload follows the FLOAT64 overload's static-shape, prefix, ownership, validation,
     * allocation, and ID-side-effect contract. The exact {@code long[]} carrier selects
     * {@link DataType#INT64}; values are copied unchanged, and gradients must be disabled.</p>
     *
     * @param shape non-null fully static result shape retained by the new dense descriptor
     * @param label non-null optional diagnostic label normalized and validated by {@link Tensor}
     * @param requiresGrad must be false because INT64 is not differentiable
     * @param source non-null caller-owned signed source with at least the logical result count;
     *     it is never retained or mutated
     * @return a non-null fresh dense INT64 tensor containing an independent exact-prefix copy
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if static-shape, Java-array-limit, source-sufficiency,
     *     descriptor-gradient, or delegated label validation fails
     * @throws ArithmeticException if checked count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after allocation
     * @throws OutOfMemoryError if copied-carrier or destination allocation fails
     */
    public static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, long[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromStrictFlatPrefix(shape, label, requiresGrad, source);
    }

    /**
     * Creates a dense BOOL tensor from a copied strict prefix of logical bytes.
     *
     * <p>This overload follows the FLOAT64 overload's static-shape, prefix, ownership, validation,
     * allocation, and ID-side-effect contract. The exact {@code byte[]} carrier selects
     * {@link DataType#BOOL}. The prefix carrier preserves its bytes until flat import maps zero to
     * canonical {@code 0} and every non-zero value to canonical {@code 1}; gradients must be
     * disabled.</p>
     *
     * @param shape non-null fully static result shape retained by the new dense descriptor
     * @param label non-null optional diagnostic label normalized and validated by {@link Tensor}
     * @param requiresGrad must be false because BOOL is not differentiable
     * @param source non-null caller-owned logical-byte source with at least the logical result
     *     count; it is never retained or mutated
     * @return a non-null fresh dense BOOL tensor containing an independent canonical prefix
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if static-shape, Java-array-limit, source-sufficiency,
     *     descriptor-gradient, or delegated label validation fails
     * @throws ArithmeticException if checked count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after allocation
     * @throws OutOfMemoryError if copied-carrier or destination allocation fails
     */
    public static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, byte[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromStrictFlatPrefix(shape, label, requiresGrad, source);
    }

    /**
     * Creates a dense FLOAT64 tensor by cyclically repeating binary64 source values.
     *
     * <p>The shape must be fully static and its checked logical element count must fit a Java
     * array. For output index {@code i}, the copied value is {@code source[i % source.length]}.
     * A non-empty result therefore requires a non-empty source; a zero-element result accepts an
     * empty source and performs no modulo operation. One fresh exact-length {@code double[]}
     * carrier holds the complete result, so neither the source nor a source subrange is retained.
     * A source at least as long as the result contributes exactly its requested prefix.</p>
     *
     * <p>The result uses a new canonical dense-contiguous {@link DataType#FLOAT64} descriptor with
     * explicit gradient intent. Null checks run in {@code shape}, {@code label}, {@code source}
     * order. Dynamic shape, checked-count, array-limit, cyclic-source, layout, and gradient
     * validation complete before output-carrier, destination, or ID allocation. The carrier is
     * delegated once to flat import. Blank-label failure occurs after carrier, destination, and ID
     * allocation but before copying and consumes the ID. Exhaustion occurs after both arrays exist;
     * neither path rolls back state.</p>
     *
     * @param shape non-null fully static result shape retained by the new dense descriptor; scalar
     *     and zero-element shapes are supported
     * @param label non-null optional diagnostic label normalized and validated by {@link Tensor}
     * @param requiresGrad whether model-level gradient eligibility is requested for FLOAT64
     * @param source non-null caller-owned binary64 cycle; it may be empty only for a zero-element
     *     result and is never retained or mutated
     * @return a non-null fresh dense FLOAT64 tensor containing an independent cyclic population
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if the shape is dynamic, count exceeds
     *     {@link Integer#MAX_VALUE}, source is empty for non-empty output, descriptor construction
     *     rejects the gradient request, or delegated Tensor validation rejects a blank label; only
     *     the label failure consumes an identifier
     * @throws ArithmeticException if checked logical-count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after carrier and
     *     destination allocation
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    public static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, double[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromCyclicFlatPrefix(shape, label, requiresGrad, source);
    }

    /**
     * Creates a dense FLOAT32 tensor by cyclically repeating binary32 source values.
     *
     * <p>This overload shares the FLOAT64 overload's modulo-order repetition, empty-result rule,
     * source non-retention, validation order, allocation, label, identity, and no-rollback
     * semantics. The exact {@code float[]} carrier selects {@link DataType#FLOAT32}; values are
     * copied unchanged without numeric conversion.</p>
     *
     * @param shape non-null fully static result shape retained by the new dense descriptor
     * @param label non-null optional diagnostic label normalized and validated by {@link Tensor}
     * @param requiresGrad whether model-level gradient eligibility is requested for FLOAT32
     * @param source non-null caller-owned binary32 cycle, empty only for empty output and never
     *     retained or mutated
     * @return a non-null fresh dense FLOAT32 tensor containing an independent cyclic population
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if static-shape, Java-array-limit, cyclic-source,
     *     descriptor-gradient, or delegated label validation fails
     * @throws ArithmeticException if checked count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after allocation
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    public static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, float[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromCyclicFlatPrefix(shape, label, requiresGrad, source);
    }

    /**
     * Creates a dense BFLOAT16 tensor by cyclically repeating raw bit patterns.
     *
     * <p>This overload shares the FLOAT64 overload's repetition, empty-result, ownership,
     * validation, allocation, and ID-side-effect contract. Each {@code short} remains raw
     * {@link DataType#BFLOAT16} bits; repetition performs no conversion or canonicalization.</p>
     *
     * @param shape non-null fully static result shape retained by the new dense descriptor
     * @param label non-null optional diagnostic label normalized and validated by {@link Tensor}
     * @param requiresGrad whether model-level gradient eligibility is requested for BFLOAT16
     * @param source non-null caller-owned raw-bit cycle, empty only for empty output and never
     *     retained or mutated
     * @return a non-null fresh dense BFLOAT16 tensor preserving the repeated raw bits independently
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if static-shape, Java-array-limit, cyclic-source,
     *     descriptor-gradient, or delegated label validation fails
     * @throws ArithmeticException if checked count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after allocation
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    public static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, short[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromCyclicFlatPrefix(shape, label, requiresGrad, source);
    }

    /**
     * Creates a dense INT32 tensor by cyclically repeating signed values.
     *
     * <p>This overload shares the FLOAT64 overload's repetition, empty-result, ownership,
     * validation, allocation, and ID-side-effect contract. The exact {@code int[]} carrier selects
     * {@link DataType#INT32}; values are copied unchanged and gradients must be disabled.</p>
     *
     * @param shape non-null fully static result shape retained by the new dense descriptor
     * @param label non-null optional diagnostic label normalized and validated by {@link Tensor}
     * @param requiresGrad must be false because INT32 is not differentiable
     * @param source non-null caller-owned signed cycle, empty only for empty output and never
     *     retained or mutated
     * @return a non-null fresh dense INT32 tensor containing an independent cyclic population
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if static-shape, Java-array-limit, cyclic-source,
     *     descriptor-gradient, or delegated label validation fails
     * @throws ArithmeticException if checked count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after allocation
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    public static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, int[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromCyclicFlatPrefix(shape, label, requiresGrad, source);
    }

    /**
     * Creates a dense INT64 tensor by cyclically repeating signed values.
     *
     * <p>This overload shares the FLOAT64 overload's repetition, empty-result, ownership,
     * validation, allocation, and ID-side-effect contract. The exact {@code long[]} carrier selects
     * {@link DataType#INT64}; values are copied unchanged and gradients must be disabled.</p>
     *
     * @param shape non-null fully static result shape retained by the new dense descriptor
     * @param label non-null optional diagnostic label normalized and validated by {@link Tensor}
     * @param requiresGrad must be false because INT64 is not differentiable
     * @param source non-null caller-owned signed cycle, empty only for empty output and never
     *     retained or mutated
     * @return a non-null fresh dense INT64 tensor containing an independent cyclic population
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if static-shape, Java-array-limit, cyclic-source,
     *     descriptor-gradient, or delegated label validation fails
     * @throws ArithmeticException if checked count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after allocation
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    public static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, long[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromCyclicFlatPrefix(shape, label, requiresGrad, source);
    }

    /**
     * Creates a dense BOOL tensor by cyclically repeating logical bytes.
     *
     * <p>This overload shares the FLOAT64 overload's repetition, empty-result, ownership,
     * validation, allocation, and ID-side-effect contract. The exact {@code byte[]} carrier selects
     * {@link DataType#BOOL}. Repetition preserves raw cycle bytes until flat import maps zero to
     * canonical {@code 0} and every non-zero value to canonical {@code 1}; gradients must be
     * disabled.</p>
     *
     * @param shape non-null fully static result shape retained by the new dense descriptor
     * @param label non-null optional diagnostic label normalized and validated by {@link Tensor}
     * @param requiresGrad must be false because BOOL is not differentiable
     * @param source non-null caller-owned zero/non-zero cycle, empty only for empty output and never
     *     retained or mutated
     * @return a non-null fresh dense BOOL tensor containing independent canonical repeated bytes
     * @throws NullPointerException if {@code shape}, {@code label}, or {@code source} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if static-shape, Java-array-limit, cyclic-source,
     *     descriptor-gradient, or delegated label validation fails
     * @throws ArithmeticException if checked count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after allocation
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    public static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, byte[] source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
        return TensorPopulations.fromCyclicFlatPrefix(shape, label, requiresGrad, source);
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
