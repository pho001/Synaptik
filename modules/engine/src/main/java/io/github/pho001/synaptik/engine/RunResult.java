package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;

/**
 * Owns one ordinary metadata-only publication lease returned by an {@link Engine} run.
 *
 * <p>Every publication occurrence remains distinct even when occurrences select the same inward
 * representation. Closing the result releases the Runtime lease and Engine-created non-owning
 * input wrappers but never caller storage. Immutable metadata remains readable after closure;
 * numerical values, host storage, physical alias identity, and backend representations are
 * deliberately not exposed. Caller storage captured for this run must remain usable until this
 * result closes.</p>
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
