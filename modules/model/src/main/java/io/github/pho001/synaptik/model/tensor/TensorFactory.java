package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.storage.HostTensorStorage;
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
 * label and
 * storage semantics to {@link Tensor}, and does not allocate host memory, own storage lifetime,
 * build descriptors, resolve layouts, create provenance, or provide compiler, runtime, or backend
 * behavior.</p>
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
