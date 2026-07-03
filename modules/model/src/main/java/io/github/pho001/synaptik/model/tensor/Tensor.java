package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import java.util.Objects;
import java.util.Optional;

/**
 * Public mutable state for one tensor, identified independently of graph-local values and nodes.
 *
 * <p>The tensor retains one immutable {@link TensorId}, one immutable {@link TensorDescriptor},
 * and one normalized optional diagnostic label. Its sole mutable state in this contract is an
 * optional borrowed {@link HostTensorStorage} association. The synchronized storage methods make
 * reference replacement and clearing atomic and visible with respect to one another; they do not
 * synchronize access to the underlying memory or prevent its caller-owned scope from closing.</p>
 *
 * <p>Construction remains package-private, and {@link TensorFactory} is the supported public
 * creation surface. The factory assigns identifiers unique among its allocations in the current
 * Java virtual machine, while this class still accepts any validated identifier through its
 * internal construction path and does not independently enforce uniqueness. It uses ordinary
 * object identity for inherited equality and hashing, so equal identifier values do not make two
 * tensor objects equal.</p>
 *
 * <p>A tensor is distinct from its immutable descriptor, graph values and nodes, operation
 * provenance, publication bindings and plans, device buffers, runtime residency, and prepared
 * execution. It owns none of those cross-layer states and neither allocates nor closes storage.</p>
 */
public final class Tensor {
    private final TensorId id;
    private final TensorDescriptor descriptor;
    private final Optional<String> label;
    private HostTensorStorage hostStorage;

    /**
     * Creates tensor state from stable metadata and an optional borrowed host-storage association.
     *
     * <p>Validation proceeds in parameter order: {@code id}, {@code descriptor}, {@code label},
     * and {@code hostStorage} optionals must be non-null; a present label is stripped and must
     * remain non-blank; then present storage is checked for matching data type, sufficient capacity
     * when layout geometry is resolved, and point-in-time liveness. A static or dynamic unresolved
     * layout performs no capacity check because this class does not invent row-major geometry.
     * Resolved capacity uses the complete referenced element span, including offset and striding;
     * scalar span is one and zero-sized span is zero.</p>
     *
     * <p>The exact immutable identifier and descriptor references are retained. Label uses optional
     * value semantics and is stored normalized. A present storage reference is borrowed and
     * retained exactly, whether writable or read-only. The caller owns its lifetime and may close
     * its scope immediately after construction; synchronization does not make raw memory access
     * thread-safe or extend JDK scope accessibility.</p>
     *
     * @param id non-null immutable tensor identity reference to retain exactly
     * @param descriptor non-null immutable logical descriptor reference to retain exactly
     * @param label non-null optional diagnostic label; present text is stripped and must contain a
     *     non-whitespace character, while empty represents absence
     * @param hostStorage non-null optional borrowed host storage to retain exactly when present;
     *     read-only storage is accepted
     * @throws NullPointerException if {@code id}, {@code descriptor}, {@code label}, or
     *     {@code hostStorage} is {@code null}, with the corresponding parameter name as the message
     * @throws IllegalArgumentException if a present label is blank, with message
     *     {@code label must not be blank}; if storage data type differs from the descriptor, with
     *     message {@code hostStorage data type must match descriptor data type:
     *     expected=<expected>, actual=<actual>}; or if resolved layout span exceeds storage
     *     capacity, with message {@code hostStorage element capacity is smaller than resolved
     *     layout span: required=<required>, actual=<actual>}
     * @throws IllegalStateException if present storage is not alive at the attachment check, with
     *     message {@code hostStorage must be alive when attached}
     */
    Tensor(
            TensorId id,
            TensorDescriptor descriptor,
            Optional<String> label,
            Optional<HostTensorStorage> hostStorage) {
        this.id = Objects.requireNonNull(id, "id");
        this.descriptor = Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(hostStorage, "hostStorage");
        this.label = normalizeLabel(label);

        if (hostStorage.isPresent()) {
            HostTensorStorage suppliedStorage = hostStorage.orElseThrow();
            validateHostStorage(suppliedStorage);
            this.hostStorage = suppliedStorage;
        }
    }

    /**
     * Returns this tensor's stable identity metadata.
     *
     * @return the exact non-null immutable identifier reference supplied at construction
     */
    public TensorId id() {
        return id;
    }

    /**
     * Returns this tensor's stable logical descriptor.
     *
     * @return the exact non-null immutable descriptor reference supplied at construction
     */
    public TensorDescriptor descriptor() {
        return descriptor;
    }

    /**
     * Returns the normalized immutable diagnostic label value.
     *
     * @return a non-null value-based optional containing stripped non-blank text, or empty when no
     *     label was supplied; optional-container identity is not part of the contract
     */
    public Optional<String> label() {
        return label;
    }

    /**
     * Returns a synchronized snapshot of the current borrowed host-storage association.
     *
     * <p>A present result contains the exact attached identity-bearing storage object. Storage is
     * borrowed rather than owned and may be read-only. It is not hidden when its caller-owned
     * scope dies after attachment, and the liveness observation available through the returned
     * object may become stale immediately. The optional is a snapshot of reference state; later
     * replacement or clearing does not mutate it. Synchronization covers only this association,
     * not segment contents, liveness, scope closure, or thread accessibility.</p>
     *
     * @return a non-null optional containing the exact current storage reference, or empty when no
     *     storage is associated
     */
    public synchronized Optional<HostTensorStorage> hostStorage() {
        return Optional.ofNullable(hostStorage);
    }

    /**
     * Validates and atomically replaces the borrowed host-storage association.
     *
     * <p>Validation completes before assignment, so failure preserves the previous exact
     * association. The proposed storage must match descriptor data type. A resolved layout
     * requires capacity at least its referenced element span; unresolved layout skips capacity
     * validation. Read-only storage is accepted, while storage already dead at the point-in-time
     * liveness check is rejected. Successful attachment does not transfer ownership, retain or
     * close a scope, guarantee future liveness, or synchronize underlying memory access. The
     * synchronized method makes validation and reference replacement atomic with respect to the
     * other synchronized storage methods only.</p>
     *
     * @param hostStorage non-null live borrowed storage to retain by exact reference
     * @return a non-null optional containing the exact previous reference, or empty when there was
     *     no previous association; the result is a snapshot
     * @throws NullPointerException if {@code hostStorage} is {@code null}, with message
     *     {@code hostStorage}
     * @throws IllegalArgumentException if data type differs from the descriptor, with message
     *     {@code hostStorage data type must match descriptor data type: expected=<expected>,
     *     actual=<actual>}, or resolved layout span exceeds capacity, with message
     *     {@code hostStorage element capacity is smaller than resolved layout span:
     *     required=<required>, actual=<actual>}
     * @throws IllegalStateException if {@code hostStorage} is not alive at the attachment check,
     *     with message {@code hostStorage must be alive when attached}
     */
    public synchronized Optional<HostTensorStorage> replaceHostStorage(
            HostTensorStorage hostStorage) {
        Objects.requireNonNull(hostStorage, "hostStorage");
        validateHostStorage(hostStorage);
        Optional<HostTensorStorage> previous = Optional.ofNullable(this.hostStorage);
        this.hostStorage = hostStorage;
        return previous;
    }

    /**
     * Atomically clears the borrowed host-storage association.
     *
     * <p>Clearing is valid for live or dead storage and never closes, releases, copies, or mutates
     * the borrowed storage, whether writable or read-only. The returned optional is a snapshot
     * containing the exact previous reference; later association changes do not mutate it.
     * Clearing performs no liveness check, so a caller-owned scope may close before or during the
     * call. Synchronization makes this reference transition atomic with respect to the other two
     * storage methods only; it does not coordinate other tensors that may share the same storage,
     * raw-memory access, scope closure, or thread accessibility.</p>
     *
     * @return a non-null optional containing the exact removed storage reference, or empty when
     *     the tensor was already storage-free
     */
    public synchronized Optional<HostTensorStorage> clearHostStorage() {
        Optional<HostTensorStorage> previous = Optional.ofNullable(hostStorage);
        hostStorage = null;
        return previous;
    }

    /**
     * Returns stable metadata-only diagnostic text.
     *
     * <p>The text contains the tensor identity, descriptor, and normalized label. It deliberately
     * omits storage presence, implementation identity, addresses, contents, liveness, graph state,
     * and runtime facts, so storage transitions do not change it. The format is not serialization.</p>
     *
     * @return non-null stable diagnostic text for this tensor's immutable metadata
     */
    @Override
    public String toString() {
        return "Tensor["
                + "id=" + id
                + ", descriptor=" + descriptor
                + ", label=" + label
                + ']';
    }

    /**
     * Normalizes optional diagnostic text without changing absence semantics.
     *
     * @param label non-null optional whose present text is stripped
     * @return non-null empty optional or an optional containing normalized non-blank text
     * @throws IllegalArgumentException if present text is blank after stripping, with message
     *     {@code label must not be blank}
     */
    private static Optional<String> normalizeLabel(Optional<String> label) {
        if (label.isEmpty()) {
            return Optional.empty();
        }
        String normalized = label.orElseThrow().strip();
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException("label must not be blank");
        }
        return Optional.of(normalized);
    }

    /**
     * Validates a proposed borrowed storage association in deterministic compatibility order.
     *
     * <p>Data type is compared first by enum identity. For resolved geometry, capacity is then
     * compared with the complete referenced element span, covering scalar, zero-sized, offset,
     * strided, and broadcast layouts. Capacity may exceed the span for shared views. Static and
     * dynamic unresolved layouts skip capacity comparison because their physical geometry is
     * unknown. Point-in-time liveness is checked last and cannot guarantee later access.</p>
     *
     * @param hostStorage non-null proposed storage; ownership remains with its caller
     * @throws IllegalArgumentException if data type differs from the descriptor, with message
     *     {@code hostStorage data type must match descriptor data type: expected=<expected>,
     *     actual=<actual>}, or resolved layout span exceeds capacity, with message
     *     {@code hostStorage element capacity is smaller than resolved layout span:
     *     required=<required>, actual=<actual>}
     * @throws IllegalStateException if storage is not alive, with message
     *     {@code hostStorage must be alive when attached}
     */
    private void validateHostStorage(HostTensorStorage hostStorage) {
        if (hostStorage.dataType() != descriptor.dataType()) {
            throw new IllegalArgumentException(
                    "hostStorage data type must match descriptor data type: expected="
                            + descriptor.dataType()
                            + ", actual="
                            + hostStorage.dataType());
        }

        descriptor.layout().ifPresent(layout -> {
            long required = layout.referencedElementSpan();
            long actual = hostStorage.elementCapacity();
            if (actual < required) {
                throw new IllegalArgumentException(
                        "hostStorage element capacity is smaller than resolved layout span: required="
                                + required
                                + ", actual="
                                + actual);
            }
        });

        if (!hostStorage.isAlive()) {
            throw new IllegalStateException("hostStorage must be alive when attached");
        }
    }
}
