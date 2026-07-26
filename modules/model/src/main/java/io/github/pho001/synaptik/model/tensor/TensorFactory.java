package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Constructs eager tensors and assigns their model identities.
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
 * import is the bounded exception: it synthesizes the exact fully static shape and canonical
 * dense-contiguous descriptor proved by the source structure. Exact primitive scalar overloads,
 * zeros, ones, type-safe full-value tensors, rectangular identity matrices, like-shaped zero/one
 * variants, and integer ranges create independent dense-contiguous eager leaf tensors.
 * Full-value methods infer the exact type from one primitive value, with only the explicitly named
 * BFLOAT16 method performing conversion. Rectangular identity creation supports all six current
 * data types, writes typed one on the main diagonal, and leaves typed zero elsewhere. The
 * {@link #eye(long, long, DataType, Optional, boolean)} method is only an unchanged-argument
 * convenience delegation to canonical identity-matrix creation. These are eager leaf-data
 * constructors, not general mutable fill, identity operations, or expression semantics. The
 * factory does not otherwise infer descriptors, retain or expose source/backing arrays, convert
 * values, accept boxed or generic nested values, provide typed
 * tensor access, expose caller-supplied provenance, allocate native or backend memory, or provide
 * compiler, runtime, backend, or random-source behavior. Every public creation method produces a
 * provenance-free leaf. Package-private derived-construction seams create one validated
 * {@link TensorProducer} together with every canonical output wrapper and indexed provenance.
 * They assign the producer's final output snapshot before returning any wrapper, then return
 * either canonical output zero or an immutable ordered list assembled from canonical indexed
 * outputs. This intentionally creates an immutable
 * {@code Tensor -> TensorProvenance -> TensorProducer -> outputs -> Tensor} cycle. Retaining one
 * result may retain sibling results; the cycle owns no external resource and remains eligible for
 * ordinary garbage collection when unreachable. These seams perform no graph capture, traversal,
 * gradient construction, compiler work, inference, evaluation, or execution. Integer ranges
 * synthesize canonical dense INT32 or INT64 leaf data and reuse flat import for final allocation
 * and identity assignment. Explicit-source random leaf initialization belongs to
 * {@link TensorRandoms}; test-fixture prefix preparation is not a production capability.</p>
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
     * Creates a fresh unlabeled provenance-free leaf tensor without host storage from a completed
     * descriptor.
     *
     * <p>The exact descriptor reference is retained by the returned tensor. No layout or storage
     * is synthesized. A null descriptor is rejected before identifier allocation and therefore
     * does not consume an identifier.</p>
     *
     * @param descriptor non-null completed immutable descriptor to retain by exact reference;
     *     the factory does not inspect or alter its data type, shape, layout, or gradient request
     * @return a non-null fresh provenance-free leaf tensor with factory-assigned identity, the
     *     exact descriptor, no label, and no host storage
     * @throws NullPointerException if {@code descriptor} is {@code null}, with message
     *     {@code descriptor}; this failure does not consume an identifier
     * @throws IllegalStateException if every non-negative identifier has been allocated, with
     *     message {@code tensor identifier space exhausted}
     */
    public static Tensor create(TensorDescriptor descriptor) {
        return create(descriptor, Optional.empty(), Optional.empty());
    }

    /**
     * Creates a fresh provenance-free leaf tensor from a completed descriptor and optional
     * caller-supplied metadata.
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
     * @return a non-null fresh provenance-free leaf tensor with factory-assigned opaque identity
     *     and the exact supplied descriptor and compatible present storage references
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
        return new Tensor(nextTensorId(), descriptor, label, Optional.empty(), hostStorage);
    }

    /**
     * Creates a storage-free derived tensor with immutable expression-origin metadata.
     *
     * <p>Factory arguments are null-checked in declaration order before a producer or identifier
     * is created. One producer snapshots the ordered input references and the exact descriptor as
     * its sole output, then validates the resulting counts through the selected operation
     * signature. Producer validation failures consume no identifier. The result uses producer
     * output index zero, retains that same descriptor reference, and is the exact object returned
     * by its producer's {@link TensorProducer#output(int) output(0)} accessor.</p>
     *
     * <p>After producer validation, exactly one identifier is allocated. A blank label rejected by
     * the package-private Tensor constructor consumes that identifier; exhaustion reports before
     * Tensor construction and is not rolled back. The producer's final output snapshot is assigned
     * before the canonical result is returned, so successful publication cannot expose a partial
     * occurrence. The fresh result has no host storage. This seam performs no family-specific
     * operand validation, descriptor inference, graph capture or traversal, gradient inference,
     * storage allocation, eager evaluation, or execution.</p>
     *
     * @param descriptor non-null completed immutable descriptor to retain by exact reference
     * @param label non-null value-based optional diagnostic label; Tensor normalizes and validates
     *     present text after identifier allocation
     * @param operation exact non-null immutable operation reference retained by the new producer
     * @param inputs non-null ordered input tensor references snapshotted by the new producer;
     *     elements must be non-null, while repeated positions are retained when the operation
     *     signature permits them
     * @return a non-null fresh derived tensor with one factory-assigned ID, producer output index
     *     zero, the exact descriptor, and no host storage
     * @throws NullPointerException if {@code descriptor}, {@code label}, {@code operation}, or
     *     {@code inputs} is null, checked in that order, or if an input element is null; these
     *     failures consume no ID
     * @throws IllegalArgumentException if the operation signature rejects the supplied input count
     *     or the single output count before allocation, or if Tensor rejects a present blank label
     *     after allocation; only the latter failure consumes an ID
     * @throws IllegalStateException if tensor identifier space is exhausted before construction,
     *     with message {@code tensor identifier space exhausted}
     */
    static Tensor createDerived(
            TensorDescriptor descriptor,
            Optional<String> label,
            Operation operation,
            List<Tensor> inputs) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        TensorProducer producer = new TensorProducer(
                operation,
                inputs,
                List.of(descriptor),
                label,
                TensorFactory::nextTensorId);
        return producer.output(0);
    }

    /**
     * Creates every storage-free, unlabeled output tensor of one validated producer occurrence.
     *
     * <p>All argument containers and producer-owned elements are validated before identifier
     * allocation. Exactly one producer snapshots the ordered inputs and output descriptors,
     * validates their counts, and is then shared by every returned tensor. Each tensor retains the
     * exact descriptor reference from its zero-based producer position, receives an independent
     * ID, and has no label or host storage. The returned list is an immutable ordered snapshot
     * assembled only through {@link TensorProducer#output(int)}; every element is therefore the
     * exact producer-retained wrapper at the same index.</p>
     *
     * <p>If identifier exhaustion occurs after earlier output IDs were allocated, those IDs remain
     * consumed and no partial result list is returned. The producer owns the canonical result
     * tensors and their indexed provenance intentionally points back to that producer. Its final
     * output snapshot is assigned before the list becomes caller-visible, so successful
     * publication exposes only a complete occurrence.</p>
     *
     * @param operation exact non-null operation reference retained by the producer
     * @param inputs non-null ordered input tensor references snapshotted by the producer
     * @param outputDescriptors non-null, non-empty ordered output descriptors snapshotted by the
     *     producer
     * @return an immutable ordered list of fresh output tensors that share the exact producer
     * @throws NullPointerException if {@code operation}, {@code inputs}, or
     *     {@code outputDescriptors} is null, checked in that order, or if an input or output
     *     descriptor element is null; these failures consume no ID
     * @throws IllegalArgumentException if output descriptors are empty or occurrence counts are
     *     rejected by the operation signature; these failures consume no ID
     * @throws IllegalStateException if tensor identifier space is exhausted; IDs allocated for
     *     earlier positions remain consumed, and no partial list is returned
     */
    static List<Tensor> createDerivedOutputs(
            Operation operation,
            List<Tensor> inputs,
            List<TensorDescriptor> outputDescriptors) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputDescriptors, "outputDescriptors");
        TensorProducer producer = new TensorProducer(
                operation,
                inputs,
                outputDescriptors,
                Optional.empty(),
                TensorFactory::nextTensorId);
        List<Tensor> outputs = new ArrayList<>(producer.outputCount());
        for (int outputIndex = 0; outputIndex < producer.outputCount(); outputIndex++) {
            outputs.add(producer.output(outputIndex));
        }
        return List.copyOf(outputs);
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
     * Creates a fully static dense FLOAT64 tensor filled with one exact binary64 value.
     *
     * <p>Scalar and zero-element shapes are valid. The value is repeated through one filled
     * {@code double[]} source and one exact flat import, preserving signed-zero and NaN payload
     * bits. Every result has new descriptor, layout, storage, backing array, identity, and empty
     * provenance. The caller supplies label and gradient intent explicitly; present label text is
     * normalized by {@link Tensor}.</p>
     *
     * <p>Validation checks {@code shape} and {@code label} for null in that order, then requires a
     * fully static shape, obtains its checked logical count, enforces the Java-array limit,
     * constructs canonical dense geometry, and validates gradient eligibility. Those failures
     * precede allocation and ID consumption. Source allocation and fill precede destination and
     * ID allocation. A blank label fails after both arrays and the ID exist and consumes that ID;
     * exhaustion occurs after both arrays exist, and an unexpected flat-copy failure occurs after
     * ID allocation. No failure rolls an ID back.</p>
     *
     * @param shape non-null fully static result shape
     * @param value exact binary64 fill value
     * @param label non-null optional diagnostic label
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent dense FLOAT64 leaf tensor
     * @throws NullPointerException if {@code shape} or {@code label} is null, checked in that order
     * @throws IllegalArgumentException if {@code shape} is dynamic, its count exceeds
     *     {@link Integer#MAX_VALUE}, {@code requiresGrad} is ineligible, or a present label is blank
     * @throws ArithmeticException if checked non-zero element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    public static Tensor full(
            Shape shape, double value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        return TensorConstants.full(shape, value, label, requiresGrad);
    }

    /**
     * Creates a fully static dense FLOAT32 tensor filled with one exact binary32 value.
     *
     * <p>One {@code float[]} source is filled exactly, including signed-zero or NaN payload bits,
     * and copied once through the matching flat import. Scalar shapes contain one value and static
     * shapes with a zero extent create empty source and destination arrays. The result has fresh
     * descriptor, layout, storage, backing array, identity, and empty provenance; present label
     * text is normalized by {@link Tensor}.</p>
     *
     * <p>Shape then label null checks precede static-shape, checked-count, Java-array-limit,
     * dense-layout, and gradient validation. Those failures consume no ID. A blank label fails
     * after source, destination, and ID allocation; exhaustion occurs after both arrays exist; and
     * an unexpected copy failure occurs after ID allocation. Identifiers are never rolled back.</p>
     *
     * @param shape non-null fully static result shape; scalar and empty shapes are valid
     * @param value exact binary32 fill value, preserving signed-zero and NaN payload bits
     * @param label non-null optional diagnostic label
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent provenance-free FLOAT32 leaf with copied storage
     * @throws NullPointerException if {@code shape} or {@code label} is null, checked in that order
     * @throws IllegalArgumentException if {@code shape} is dynamic, its count exceeds
     *     {@link Integer#MAX_VALUE}, {@code requiresGrad} is ineligible, or a present label is blank
     * @throws ArithmeticException if checked non-zero element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    public static Tensor full(
            Shape shape, float value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        return TensorConstants.full(shape, value, label, requiresGrad);
    }

    /**
     * Creates a fully static dense BFLOAT16 tensor by converting and repeating one binary32 value.
     *
     * <p>After descriptor validation, {@link BFloat16Bits#fromFloat(float)} converts the semantic
     * input once using round-to-nearest with ties to even and canonical NaN handling. One
     * {@code short[]} source is filled with the converted raw bits and copied once. Scalar and
     * empty static shapes are valid. The result owns fresh metadata and storage, has empty
     * provenance, and retains neither the source nor any conversion state. Empty label means
     * absent; present text is stripped and validated by {@link Tensor}.</p>
     *
     * <p>Shape then label null checks precede static-shape, checked-count, Java-array-limit,
     * dense-layout, and gradient validation. Those failures consume no ID. A blank label fails
     * after source, destination, and ID allocation; exhaustion occurs after both arrays exist; and
     * an unexpected copy failure occurs after ID allocation. Identifiers are never rolled back.</p>
     *
     * @param shape non-null fully static result shape; scalar and empty shapes are valid
     * @param value binary32 semantic value converted with {@link BFloat16Bits#fromFloat(float)}
     * @param label non-null optional diagnostic label
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent provenance-free BFLOAT16 leaf with copied storage
     * @throws NullPointerException if {@code shape} or {@code label} is null, checked in that order
     * @throws IllegalArgumentException if {@code shape} is dynamic, its count exceeds
     *     {@link Integer#MAX_VALUE}, {@code requiresGrad} is ineligible, or a present label is blank
     * @throws ArithmeticException if checked non-zero element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    public static Tensor fullBFloat16(
            Shape shape, float value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        return TensorConstants.fullBFloat16(shape, value, label, requiresGrad);
    }

    /**
     * Creates a fully static dense INT32 tensor filled with one exact signed value.
     *
     * <p>One {@code int[]} source is filled exactly and copied once through the matching flat
     * import. Scalar and empty static shapes are valid. The result has new descriptor, layout,
     * source and destination carriers, storage, Tensor identity, and empty provenance; it retains
     * no source carrier. Empty label means absent; present text is stripped and validated by
     * {@link Tensor}. INT32 is not differentiable, so {@code requiresGrad} must be false.</p>
     *
     * <p>Shape then label null checks precede static-shape, checked-count, Java-array-limit,
     * dense-layout, and gradient validation. Those failures consume no ID. A blank label fails
     * after source, destination, and ID allocation; exhaustion occurs after both arrays exist; and
     * an unexpected copy failure occurs after ID allocation. Identifiers are never rolled back.</p>
     *
     * @param shape non-null fully static result shape; scalar and empty shapes are valid
     * @param value exact signed 32-bit fill value
     * @param label non-null optional diagnostic label
     * @param requiresGrad explicit request, which must be false for INT32
     * @return a non-null fresh independent provenance-free INT32 leaf with copied storage
     * @throws NullPointerException if {@code shape} or {@code label} is null, checked in that order
     * @throws IllegalArgumentException if {@code shape} is dynamic, its count exceeds
     *     {@link Integer#MAX_VALUE}, gradients are requested, or a present label is blank
     * @throws ArithmeticException if checked non-zero element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    public static Tensor full(
            Shape shape, int value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        return TensorConstants.full(shape, value, label, requiresGrad);
    }

    /**
     * Creates a fully static dense INT64 tensor filled with one exact signed value.
     *
     * <p>One {@code long[]} source is filled exactly and copied once through the matching flat
     * import. Scalar and empty static shapes are valid. The result has new descriptor, layout,
     * source and destination carriers, storage, Tensor identity, and empty provenance; it retains
     * no source carrier. Empty label means absent; present text is stripped and validated by
     * {@link Tensor}. INT64 is not differentiable, so {@code requiresGrad} must be false.</p>
     *
     * <p>Shape then label null checks precede static-shape, checked-count, Java-array-limit,
     * dense-layout, and gradient validation. Those failures consume no ID. A blank label fails
     * after source, destination, and ID allocation; exhaustion occurs after both arrays exist; and
     * an unexpected copy failure occurs after ID allocation. Identifiers are never rolled back.</p>
     *
     * @param shape non-null fully static result shape; scalar and empty shapes are valid
     * @param value exact signed 64-bit fill value
     * @param label non-null optional diagnostic label
     * @param requiresGrad explicit request, which must be false for INT64
     * @return a non-null fresh independent provenance-free INT64 leaf with copied storage
     * @throws NullPointerException if {@code shape} or {@code label} is null, checked in that order
     * @throws IllegalArgumentException if {@code shape} is dynamic, its count exceeds
     *     {@link Integer#MAX_VALUE}, gradients are requested, or a present label is blank
     * @throws ArithmeticException if checked non-zero element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    public static Tensor full(
            Shape shape, long value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        return TensorConstants.full(shape, value, label, requiresGrad);
    }

    /**
     * Creates a fully static dense BOOL tensor filled with one semantic boolean value.
     *
     * <p>The semantic value maps directly to canonical byte zero or one; no numeric truthiness or
     * conversion is accepted. One {@code byte[]} source is filled and copied once through BOOL
     * flat import. Scalar and empty static shapes are valid. The result has fresh independent
     * metadata, storage, identity, and empty provenance, and retains no source carrier. BOOL is
     * not differentiable, so {@code requiresGrad} must be false. Empty label means absent;
     * present text is stripped and validated by {@link Tensor}.</p>
     *
     * <p>Shape then label null checks precede static-shape, checked-count, Java-array-limit,
     * dense-layout, and gradient validation. Those failures consume no ID. A blank label fails
     * after source, destination, and ID allocation; exhaustion occurs after both arrays exist; and
     * an unexpected copy failure occurs after ID allocation. Identifiers are never rolled back.</p>
     *
     * @param shape non-null fully static result shape; scalar and empty shapes are valid
     * @param value semantic value stored only as canonical byte zero or one
     * @param label non-null optional diagnostic label
     * @param requiresGrad explicit request, which must be false for BOOL
     * @return a non-null fresh independent provenance-free BOOL leaf with copied storage
     * @throws NullPointerException if {@code shape} or {@code label} is null, checked in that order
     * @throws IllegalArgumentException if {@code shape} is dynamic, its count exceeds
     *     {@link Integer#MAX_VALUE}, gradients are requested, or a present label is blank
     * @throws ArithmeticException if checked non-zero element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    public static Tensor full(
            Shape shape, boolean value, Optional<String> label, boolean requiresGrad) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        return TensorConstants.full(shape, value, label, requiresGrad);
    }

    /**
     * Creates a dense rectangular matrix with typed one on the main diagonal and zero elsewhere.
     *
     * <p>All six current data types are supported. Rows and columns may be unequal or zero, and
     * the result always has shape {@code [rows, columns]}. One default-zero matching carrier is
     * populated only at coordinates {@code (i, i)} for {@code 0 <= i < min(rows, columns)}, then
     * copied through one flat import. Diagonal values are {@code 1.0d}, {@code 1.0f}, converted
     * BFLOAT16 one bits, {@code 1}, {@code 1L}, or canonical BOOL byte {@code 1}; off-diagonal
     * values retain the corresponding JVM default-zero representation. Square, wide, tall,
     * zero-row, and zero-column matrices are valid. The result has new descriptor, dense layout,
     * source and destination carriers, storage, identity, and empty provenance; no carrier is
     * shared or retained outside its storage. Empty label means absent; present text is stripped
     * and validated by {@link Tensor}.</p>
     *
     * <p>Validation checks {@code dataType} and {@code label} for null in that order, then rejects
     * negative rows before negative columns. Rank-two shape construction is followed by checked
     * element count, Java-array limit, canonical dense geometry, and gradient eligibility before
     * source, destination, or ID allocation. A blank label fails after both arrays and the ID
     * exist and consumes that ID. Exhaustion is observed after both arrays exist; an unexpected
     * copy failure occurs after ID allocation. No ID is rolled back.</p>
     *
     * @param rows non-negative row count
     * @param columns non-negative column count
     * @param dataType non-null exact matrix element type
     * @param label non-null optional diagnostic label
     * @param requiresGrad explicit model-level gradient request, valid only for floating types
     * @return a non-null fresh independent dense rectangular identity-matrix leaf
     * @throws NullPointerException if {@code dataType} or {@code label} is null, checked in that order
     * @throws IllegalArgumentException if {@code rows} or {@code columns} is negative, the element
     *     count exceeds {@link Integer#MAX_VALUE}, {@code requiresGrad} is ineligible for
     *     {@code dataType}, or a present label is blank
     * @throws ArithmeticException if checked positive element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    public static Tensor identityMatrix(
            long rows,
            long columns,
            DataType dataType,
            Optional<String> label,
            boolean requiresGrad) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(label, "label");
        return TensorConstants.identityMatrix(rows, columns, dataType, label, requiresGrad);
    }

    /**
     * Delegates unchanged to {@link #identityMatrix(long, long, DataType, Optional, boolean)}.
     *
     * <p>This alias performs no validation or allocation of its own. Equal arguments yield equal
     * descriptor/value behavior, while separate successful calls still return distinct Tensor,
     * descriptor, layout, storage, backing-array, and identifier objects. It therefore has the
     * same rectangular, data-type, label, gradient, provenance, validation-order, allocation, and
     * failure-side-effect contract as the canonical method.</p>
     *
     * @param rows non-negative row count passed unchanged to {@code identityMatrix}
     * @param columns non-negative column count passed unchanged to {@code identityMatrix}
     * @param dataType non-null exact element type passed unchanged to {@code identityMatrix}
     * @param label non-null optional label passed unchanged to {@code identityMatrix}
     * @param requiresGrad explicit gradient request passed unchanged to {@code identityMatrix}
     * @return the exact fresh tensor returned by the single canonical method invocation
     * @throws NullPointerException if {@code dataType} or {@code label} is null, checked by the
     *     canonical method in that order
     * @throws IllegalArgumentException if canonical row, column, count, gradient, or label validation fails
     * @throws ArithmeticException if canonical checked positive count or dense geometry overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after canonical destination allocation
     * @throws OutOfMemoryError if canonical source or destination allocation fails before ID allocation
     */
    public static Tensor eye(
            long rows,
            long columns,
            DataType dataType,
            Optional<String> label,
            boolean requiresGrad) {
        return identityMatrix(rows, columns, dataType, label, requiresGrad);
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
        return TensorRanges.range(startInclusive, endExclusive, step, label);
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
        return TensorRanges.range(startInclusive, endExclusive, step, label);
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
     * <p>This package-owned, non-public seam is shared by ordinary factory methods and
     * {@link TensorProducer}'s existing package-private three-argument construction path. That
     * path now creates canonical wrappers and must use the same global identity sequence. Package
     * visibility is the minimal collaboration needed to preserve uniqueness without another
     * allocator type, duplicate state, reflection, or a redundant forwarding abstraction; it
     * does not broaden the public API.</p>
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
    static TensorId nextTensorId() {
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
