package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Describes one bounded functional reverse-mode differentiation request.
 *
 * <p>A request contains either one reverse-mode stage or exactly two stages. In the two-stage
 * form, the second stage differentiates exact gradient results produced for selected targets of
 * the first stage. Construction validates and snapshots only the public structural shape of the
 * request. Tensor membership, descriptor, provenance, route, and seed checks remain compiler
 * preflight work because they depend on the supplied forward boundary.</p>
 *
 * <p>The request is immutable compiler input. It neither mutates a Tensor nor establishes a
 * recording scope, persistent derivative chain, runtime tape, or executable computation.</p>
 *
 * @param stages non-null one- or two-element ordered stage list; membership is snapshotted
 */
public record FunctionalGradientRequest(
        List<FunctionalGradientRequest.Stage> stages) {

    /**
     * Selects the result for a valid target that has no differentiable route from the selected
     * stage outputs.
     */
    public enum DisconnectedPolicy {
        /** Reject the first disconnected target during stage preflight. */
        ERROR,
        /** Return an ordinary exact typed zero expression for each disconnected target role. */
        ZERO
    }

    /**
     * Identifies one output whose cotangent seeds a reverse-mode stage.
     */
    public sealed interface OutputReference
            permits ForwardTensorReference, FirstStageGradientReference {}

    /**
     * Refers by exact object identity to a Tensor in the requested forward-output boundary.
     *
     * @param tensor non-null exact forward Tensor reference
     */
    public record ForwardTensorReference(Tensor tensor)
            implements OutputReference {
        /**
         * Validates one forward reference.
         *
         * @param tensor non-null exact forward Tensor reference
         * @throws NullPointerException if {@code tensor} is {@code null}
         */
        public ForwardTensorReference {
            Objects.requireNonNull(tensor, "tensor");
        }
    }

    /**
     * Refers to the exact generated first-stage gradient for one target-list position.
     *
     * @param targetIndex zero-based index in the first-stage target list
     */
    public record FirstStageGradientReference(int targetIndex)
            implements OutputReference {
        /**
         * Validates one first-stage result reference.
         *
         * @param targetIndex zero-based index in the first-stage target list
         * @throws IllegalArgumentException if {@code targetIndex} is negative
         */
        public FirstStageGradientReference {
            if (targetIndex < 0) {
                throw new IllegalArgumentException("targetIndex must not be negative");
            }
        }
    }

    /**
     * Describes one ordered reverse-mode stage.
     *
     * @param outputs non-null, non-empty ordered output references; snapshotted
     * @param cotangentSeeds non-null output-aligned optional explicit seeds; snapshotted
     * @param targets non-null, non-empty exact-object-identity-unique target list; snapshotted
     * @param createGraph whether the generated formula is retained for the immediately following
     *     stage; valid only for stage one of a two-stage request
     * @param disconnectedPolicy non-null policy for valid targets with no differentiable route
     */
    public record Stage(
            List<OutputReference> outputs,
            List<Optional<Tensor>> cotangentSeeds,
            List<Tensor> targets,
            boolean createGraph,
            DisconnectedPolicy disconnectedPolicy) {
        /**
         * Validates and snapshots one stage's local structure.
         *
         * @param outputs non-null, non-empty ordered output references
         * @param cotangentSeeds non-null output-aligned optional explicit seeds
         * @param targets non-null, non-empty exact-object-identity-unique target list
         * @param createGraph compile-local immediate-next-stage retention permission
         * @param disconnectedPolicy non-null disconnected-target policy
         * @throws NullPointerException if a component, element, optional, present seed, target,
         *     or policy is {@code null}
         * @throws IllegalArgumentException if an output or target list is empty, seed count does
         *     not match output count, or an exact target reference repeats
         */
        public Stage {
            Objects.requireNonNull(outputs, "outputs");
            Objects.requireNonNull(cotangentSeeds, "cotangentSeeds");
            Objects.requireNonNull(targets, "targets");
            Objects.requireNonNull(disconnectedPolicy, "disconnectedPolicy");
            if (outputs.isEmpty()) {
                throw new IllegalArgumentException("outputs must not be empty");
            }
            if (cotangentSeeds.size() != outputs.size()) {
                throw new IllegalArgumentException(
                        "cotangentSeeds size must equal outputs size");
            }
            if (targets.isEmpty()) {
                throw new IllegalArgumentException("targets must not be empty");
            }
            for (int index = 0; index < outputs.size(); index++) {
                Objects.requireNonNull(outputs.get(index), "outputs[" + index + "]");
                Optional<Tensor> seed = Objects.requireNonNull(
                        cotangentSeeds.get(index), "cotangentSeeds[" + index + "]");
                if (seed.isPresent()) {
                    Objects.requireNonNull(
                            seed.orElseThrow(), "cotangentSeeds[" + index + "].value");
                }
            }
            IdentityHashMap<Tensor, Integer> positions = new IdentityHashMap<>();
            for (int index = 0; index < targets.size(); index++) {
                Tensor target = Objects.requireNonNull(targets.get(index), "targets[" + index + "]");
                Integer first = positions.putIfAbsent(target, index);
                if (first != null) {
                    throw new IllegalArgumentException(
                            "targets[" + index + "] duplicates targets[" + first + "]");
                }
            }
            outputs = List.copyOf(outputs);
            cotangentSeeds = List.copyOf(cotangentSeeds);
            targets = List.copyOf(targets);
        }
    }

    /**
     * Validates the bounded one- or two-stage request matrix and snapshots stage membership.
     *
     * @param stages non-null ordered stage list containing exactly one or two stages
     * @throws NullPointerException if {@code stages} or a stage element is {@code null}
     * @throws IllegalArgumentException if the stage count, reference direction, reference index,
     *     or {@code createGraph} combination is outside the bounded contract
     */
    public FunctionalGradientRequest {
        Objects.requireNonNull(stages, "stages");
        if (stages.size() < 1 || stages.size() > 2) {
            throw new IllegalArgumentException("stages must contain exactly one or two stages");
        }
        for (int index = 0; index < stages.size(); index++) {
            Objects.requireNonNull(stages.get(index), "stages[" + index + "]");
        }
        Stage first = stages.getFirst();
        for (int index = 0; index < first.outputs().size(); index++) {
            if (!(first.outputs().get(index) instanceof ForwardTensorReference)) {
                throw new IllegalArgumentException(
                        "stages[0].outputs[" + index
                                + "] must be a ForwardTensorReference");
            }
        }
        if (stages.size() == 1) {
            if (first.createGraph()) {
                throw new IllegalArgumentException(
                        "stages[0].createGraph must be false for a one-stage request");
            }
        } else {
            if (!first.createGraph()) {
                throw new IllegalArgumentException(
                        "stages[0].createGraph must be true for a two-stage request");
            }
            Stage second = stages.get(1);
            if (second.createGraph()) {
                throw new IllegalArgumentException(
                        "stages[1].createGraph must be false");
            }
            for (int index = 0; index < second.outputs().size(); index++) {
                OutputReference reference = second.outputs().get(index);
                if (!(reference instanceof FirstStageGradientReference gradientReference)) {
                    throw new IllegalArgumentException(
                            "stages[1].outputs[" + index
                                    + "] must be a FirstStageGradientReference");
                }
                if (gradientReference.targetIndex() >= first.targets().size()) {
                    throw new IllegalArgumentException(
                            "stages[1].outputs[" + index + "].targetIndex must be smaller than "
                                    + "stages[0].targets size");
                }
            }
        }
        stages = List.copyOf(stages);
    }
}
