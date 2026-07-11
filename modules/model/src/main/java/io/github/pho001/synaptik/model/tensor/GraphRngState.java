package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Opaque public value for one explicit RNG state occurrence in a Tensor expression graph.
 *
 * <p>The state privately retains one storage-free Tensor whose two logical {@code INT64} lanes
 * carry raw unsigned key and counter words. The Tensor supplies expression identity and
 * provenance only: this type exposes no numerical Tensor, storage mutation, random source,
 * execution state, or hidden process-global or thread-local generator.</p>
 *
 * <p>Instances are shallowly immutable and inherit ordinary object-identity equality and hashing.
 * Equal key/counter inputs request equivalent abstract stream positions but create distinct state,
 * Tensor, producer, and identifier occurrences. Branching the same state into multiple dropout
 * consumers intentionally requests interval reuse; sequential callers thread each operation's
 * {@link DropoutResult#nextState() next state}. The immutable wrapper may be shared, but it does
 * not synchronize consuming execution.</p>
 *
 * <p>No random algorithm or portable bitstream is selected. Replay is therefore bounded to a
 * conforming prepared implementation and configuration that defines the consuming operation.
 * Future graph serialization must preserve the raw words losslessly, but this type defines no
 * byte encoding, parser, schema, or stable token. Construction records semantics only and
 * performs no sampling or execution.</p>
 */
public final class GraphRngState {
    private final Tensor tensor;

    /**
     * Creates an opaque wrapper around a validated state Tensor occurrence.
     *
     * <p>Validation checks, in order, non-nullity, exact {@code INT64} type, exact structural
     * {@code Shape.of(2)}, unresolved layout, false gradient eligibility, absent label, absent
     * host storage, and present provenance. The exact Tensor reference is retained.</p>
     *
     * @param tensor non-null state Tensor occurrence to validate and retain exactly
     * @throws NullPointerException if {@code tensor} is null, with message {@code tensor}
     * @throws IllegalArgumentException if the first violated state invariant is data type, shape,
     *     layout, gradient eligibility, label, host storage, or provenance
     */
    GraphRngState(Tensor tensor) {
        Tensor validatedTensor = Objects.requireNonNull(tensor, "tensor");
        TensorDescriptor descriptor = validatedTensor.descriptor();
        if (descriptor.dataType() != DataType.INT64) {
            throw new IllegalArgumentException("state tensor data type must be INT64");
        }
        if (!descriptor.shape().equals(Shape.of(2))) {
            throw new IllegalArgumentException("state tensor shape must be Shape[2]");
        }
        if (descriptor.layout().isPresent()) {
            throw new IllegalArgumentException("state tensor layout must be unresolved");
        }
        if (descriptor.requiresGrad()) {
            throw new IllegalArgumentException("state tensor must not require gradients");
        }
        if (validatedTensor.label().isPresent()) {
            throw new IllegalArgumentException("state tensor label must be absent");
        }
        if (validatedTensor.hostStorage().isPresent()) {
            throw new IllegalArgumentException("state tensor host storage must be absent");
        }
        if (validatedTensor.provenance().isEmpty()) {
            throw new IllegalArgumentException("state tensor provenance must be present");
        }
        this.tensor = validatedTensor;
    }

    /**
     * Creates one explicit graph RNG state-expression occurrence.
     *
     * <p>Both arguments accept every Java {@code long} bit pattern and are interpreted as unsigned
     * 64-bit words. The result privately wraps one fresh, unlabeled, storage-free derived Tensor
     * with exact descriptor {@code INT64}, {@code Shape.of(2)}, unresolved layout, and false
     * gradient eligibility. Its zero-input {@link GraphRngKind#INITIAL_STATE} producer records
     * {@link GraphRngStateAttrs}, and provenance selects output index zero.</p>
     *
     * <p>Construction allocates exactly one Tensor identifier after operation-signature
     * validation. It allocates no random source or state storage, performs no draw, and makes no
     * cross-backend bitstream promise. A separately constructed result with equal arguments is a
     * replay-equivalent abstract position, not the same expression occurrence.</p>
     *
     * @param key caller-selected stream/domain identity as an unsigned 64-bit bit pattern
     * @param counter next abstract logical sample position as an unsigned 64-bit bit pattern
     * @return a non-null fresh state occurrence with identity equality
     * @throws IllegalStateException if Tensor identifier space is exhausted, with message
     *     {@code tensor identifier space exhausted}
     */
    public static GraphRngState initial(long key, long counter) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.INT64, Shape.of(2), Optional.empty(), false);
        Operation operation = new Operation(
                GraphRngKind.INITIAL_STATE, new GraphRngStateAttrs(key, counter));
        Tensor stateTensor = TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of());
        return new GraphRngState(stateTensor);
    }

    /**
     * Returns the exact private Tensor occurrence for same-package state-consuming construction.
     *
     * @return the exact non-null Tensor reference retained by this state
     */
    Tensor tensor() {
        return tensor;
    }
}
