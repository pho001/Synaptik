package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;

/**
 * Owns one ordinary publication lease returned by an {@link Engine} run.
 *
 * <p>Every publication occurrence remains distinct even when occurrences select the same inward
 * representation. Closing the result releases the Runtime lease and Engine-created non-owning
 * input wrappers but never caller storage. Immutable metadata remains readable after closure.
 * While both result and Engine are open, an exact occurrence may be copied explicitly into a
 * detached {@link HostTensorValue}; host storage, physical alias identity, and backend
 * representations are never exposed. Caller storage captured for this run must remain usable
 * until this result closes.</p>
 */
public final class RunResult implements AutoCloseable {
    private final AdvancedRunResult owner;
    private final List<Publication> publications;

    /**
     * Creates a metadata view over one registered lifecycle owner.
     *
     * @param owner non-null registered inward lifecycle owner
     * @param specifications non-null immutable publication specifications with non-null elements
     * @throws NullPointerException if an argument or specification is {@code null}
     */
    RunResult(AdvancedRunResult owner, List<CompiledGraph.PublicationSpec> specifications) {
        this.owner = Objects.requireNonNull(owner, "owner");
        Objects.requireNonNull(specifications, "specifications");
        var snapshot = new ArrayList<Publication>(specifications.size());
        for (int index = 0; index < specifications.size(); index++) {
            CompiledGraph.PublicationSpec spec = Objects.requireNonNull(
                    specifications.get(index), "specifications[" + index + "]");
            snapshot.add(new Publication(this, index, spec));
        }
        publications = List.copyOf(snapshot);
    }

    /**
     * Returns the complete ordered publication-occurrence count, including aliases.
     *
     * @return the immutable non-negative publication count
     */
    public int resultCount() {
        return publications.size();
    }

    /**
     * Returns every forward-then-gradient publication occurrence in result order.
     *
     * @return the same non-null immutable list and occurrence objects on every call
     */
    public List<Publication> publications() {
        return publications;
    }

    /**
     * Materializes one exact publication occurrence into an immutable detached host value.
     *
     * <p>The selector must be the exact {@link Publication} object created by this result; equal
     * metadata, a matching index, or an occurrence from another run is not sufficient. The
     * synchronous call is valid only while both this result and its Engine remain open. Calls on
     * one result serialize with each other and with result closure; Engine closure waits for an
     * admitted copy. Every success performs a new CPU copy and returns an independent value that
     * remains readable after result or Engine closure. No result is cached or deduplicated, even
     * when publication occurrences alias the same inward representation.</p>
     *
     * <p>Current support is the fixed CPU-only standard composition with a fully static Shape and
     * resolved final layout. This operation performs no cross-backend transfer, route search,
     * data-type conversion, Tensor construction, or implicit materialization.</p>
     *
     * @param publication non-null exact occurrence object from {@link #publications()}
     * @param maximumBytes non-negative upper bound for the canonical payload in bytes
     * @return a fresh non-null detached immutable host value
     * @throws NullPointerException if {@code publication} or backend bytes are null
     * @throws IllegalArgumentException if the occurrence is foreign, the limit is negative, the
     *     Shape is not fully static, the layout is unresolved, or the payload exceeds the caller
     *     limit or JVM array ceiling
     * @throws IllegalStateException if Engine or result closure has begun, or backend byte length
     *     differs from the descriptor-derived canonical count
     * @throws ArithmeticException if checked logical count or byte-count arithmetic overflows
     * @throws RuntimeException if Runtime or CPU validation or copying reports another unchecked
     *     failure
     * @throws Error if copying reports a fatal failure
     */
    public HostTensorValue materialize(Publication publication, long maximumBytes) {
        return owner.owner().materializeOrdinary(owner, this, publication, maximumBytes);
    }

    /** Performs ordered outward validation and one-hop copying while the owner monitor is held. */
    HostTensorValue materializeOpen(
            Publication publication,
            long maximumBytes,
            io.github.pho001.synaptik.runtime.run.RunResult delegate,
            EngineBackendComposition composition) {
        Objects.requireNonNull(publication, "publication");
        if (publication.result != this) {
            throw new IllegalArgumentException(
                    "publication does not belong to this run result");
        }
        if (maximumBytes < 0) {
            throw new IllegalArgumentException(
                    "maximumBytes must be non-negative: " + maximumBytes);
        }
        TensorDescriptor descriptor = publication.descriptor;
        if (!descriptor.shape().isFullyStatic()) {
            throw new IllegalArgumentException(
                    "host snapshot requires a fully static shape: " + descriptor.shape());
        }
        if (descriptor.layout().isEmpty()) {
            throw new IllegalArgumentException("host snapshot requires a resolved layout");
        }
        long elementCount = descriptor.shape().knownElementCount().orElseThrow();
        long byteCount = Math.multiplyExact(elementCount, descriptor.dataType().byteWidth());
        if (byteCount > maximumBytes) {
            throw new IllegalArgumentException(
                    "canonical byte count exceeds maximumBytes: required=" + byteCount
                            + ", maximum=" + maximumBytes);
        }
        if (byteCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "canonical byte count exceeds JVM byte[] limit: " + byteCount);
        }
        byte[] bytes = composition.copyToCanonicalHostBytes(
                delegate.publicationRepresentation(publication.index), descriptor, maximumBytes);
        Objects.requireNonNull(bytes, "canonicalBytes");
        if (bytes.length != byteCount) {
            throw new IllegalStateException(
                    "backend canonical byte count does not match descriptor: expected="
                            + byteCount + ", actual=" + bytes.length);
        }
        return new HostTensorValue(descriptor.dataType(), descriptor.shape(), bytes);
    }

    /**
     * Reports whether cleanup of this result has begun.
     *
     * @return {@code true} during or after the first close call; otherwise {@code false}
     */
    public boolean isClosed() {
        return owner.isClosed();
    }

    /**
     * Closes the registered Runtime lease and its Engine-owned borrow wrappers exactly once.
     *
     * @throws RuntimeException if cleanup reports an unchecked failure
     * @throws Error if cleanup reports a fatal failure
     */
    @Override
    public void close() {
        owner.close();
    }

    /** Describes whether one publication is a requested forward value or target gradient. */
    public enum Role {
        /** A requested forward-output occurrence. */
        FORWARD,
        /** A first-order gradient occurrence associated with one requested target. */
        GRADIENT
    }

    /**
     * Immutable identity and derivative-role metadata for one result occurrence.
     *
     * <p>This identity-bearing object is fresh per run and retains its originating result lease
     * privately. A forward occurrence identifies the requested output Tensor. A gradient
     * occurrence identifies the requested target rather than a generated gradient Tensor, and
     * retains the target-list position even when another occurrence selects the same inward
     * representation. It has no independent close or value access.</p>
     */
    public static final class Publication {
        private final RunResult result;
        private final int index;
        private final TensorId tensorId;
        private final TensorDescriptor descriptor;
        private final Role role;
        private final OptionalInt derivativeOrder;
        private final OptionalInt targetIndex;

        private Publication(
                RunResult result, int index, CompiledGraph.PublicationSpec specification) {
            this.result = result;
            this.index = index;
            tensorId = specification.tensorId;
            descriptor = specification.descriptor;
            role = specification.role;
            derivativeOrder = role == Role.GRADIENT
                    ? OptionalInt.of(specification.derivativeOrder) : OptionalInt.empty();
            targetIndex = role == Role.GRADIENT
                    ? OptionalInt.of(specification.targetIndex) : OptionalInt.empty();
        }

        /**
         * Returns this occurrence's stable position in its originating result.
         *
         * @return the dense zero-based occurrence position in the complete result
         */
        public int index() {
            return index;
        }

        /**
         * Returns the logical Tensor identity selected by this publication role.
         *
         * @return the requested forward Tensor identity or differentiation-target identity
         */
        public TensorId tensorId() {
            return tensorId;
        }

        /**
         * Returns the final descriptor after compilation, including resolved layout state.
         *
         * @return the exact immutable descriptor of the final published logical value
         */
        public TensorDescriptor descriptor() {
            return descriptor;
        }

        /**
         * Classifies this occurrence without exposing its inward representation.
         *
         * @return the non-null forward or gradient role
         */
        public Role role() {
            return role;
        }

        /**
         * Reports the derivative order only where the role defines one.
         *
         * @return one for a gradient occurrence, or empty for a forward occurrence
         */
        public OptionalInt derivativeOrder() {
            return derivativeOrder;
        }

        /**
         * Reports which explicit target produced a gradient role.
         *
         * @return the explicit target-list index for a gradient, or empty for a forward
         */
        public OptionalInt targetIndex() {
            return targetIndex;
        }

        /**
         * Observes the shared lifecycle of the originating result lease.
         *
         * @return whether closure of the originating result lease has begun
         */
        public boolean isClosed() {
            return result.isClosed();
        }
    }
}
