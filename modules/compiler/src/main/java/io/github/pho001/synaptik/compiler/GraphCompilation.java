package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.ValueId;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable mode-neutral result of compiler graph construction and exact optimization.
 *
 * <p>The validated graph is the sole graph-stage result: forward-only for
 * {@link CompileMode#FORWARD_ONLY}, and potentially combined forward/backward for
 * {@link CompileMode#FORWARD_AND_BACKWARD} or {@link CompileMode#TRAINING_STEP}. Forward values
 * form the graph-output prefix. Gradient publication bindings retain request order and may share
 * one final {@link ValueId}; the graph boundary lists that value only at its first occurrence. No
 * Tensor, producer, provenance, mutable request map, or runtime/backend object is retained.</p>
 *
 * <p>This package-private record is not {@code CompileArtifacts}. It contains no publication,
 * partition, logical-memory, diagnostic, preparation, schedule, physical storage, executable, or
 * optimizer state.</p>
 *
 * @param mode non-null exact requested compile mode
 * @param validatedGraph non-null final validated graph, constraints, constant sidecar, bindable
 *     inputs, and per-node phase facts
 * @param forwardOutputs non-null ordered identity-unique forward graph-output prefix; snapshotted
 * @param gradientResults non-null ordered requested target-to-gradient publication bindings;
 *     snapshotted and empty exactly for forward-only mode
 * @param derivatives non-null derivative-order metadata for the exact validated graph
 */
record GraphCompilation(
        CompileMode mode,
        ValidatedGraph validatedGraph,
        List<ValueId> forwardOutputs,
        List<GradientPublicationBinding> gradientResults,
        DerivativeGraphMetadata derivatives) {
    /**
     * Validates and snapshots one mode-selected graph-stage result.
     *
     * @param mode non-null exact requested compile mode
     * @param validatedGraph non-null final validated graph and sidecars
     * @param forwardOutputs non-null ordered identity-unique forward graph-output prefix
     * @param gradientResults non-null ordered requested target-to-gradient publication bindings
     * @param derivatives non-null derivative-order metadata for the exact validated graph
     * @throws NullPointerException if a required component or list element is {@code null}
     * @throws IllegalArgumentException if a boundary value is absent, forward values repeat, mode
     *     and backward state disagree, or the graph output boundary is not the stable forward
     *     prefix followed by first-occurrence-distinct gradient values
     */
    GraphCompilation {
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(validatedGraph, "validatedGraph");
        forwardOutputs = List.copyOf(Objects.requireNonNull(forwardOutputs, "forwardOutputs"));
        gradientResults =
                List.copyOf(Objects.requireNonNull(gradientResults, "gradientResults"));
        Objects.requireNonNull(derivatives, "derivatives");
        if (derivatives.graph() != validatedGraph.graph()
                || validatedGraph.derivatives() != derivatives) {
            throw new IllegalArgumentException(
                    "derivatives must be the exact validated-graph sidecar");
        }

        Set<ValueId> values = new HashSet<>();
        validatedGraph.graph().values().forEach(value -> values.add(value.id()));
        Set<ValueId> graphOutputs = new HashSet<>(validatedGraph.graph().outputs());
        Set<ValueId> seenForward = new HashSet<>();
        for (int index = 0; index < forwardOutputs.size(); index++) {
            ValueId output = Objects.requireNonNull(
                    forwardOutputs.get(index), "forwardOutputs[" + index + "]");
            if (!values.contains(output) || !graphOutputs.contains(output)) {
                throw new IllegalArgumentException(
                        "forwardOutputs[" + index + "] is not a graph output");
            }
            if (!seenForward.add(output)) {
                throw new IllegalArgumentException(
                        "forwardOutputs[" + index + "] duplicates an earlier forward output");
            }
        }
        for (int index = 0; index < gradientResults.size(); index++) {
            GradientPublicationBinding role = Objects.requireNonNull(
                    gradientResults.get(index), "gradientResults[" + index + "]");
            if (!values.contains(role.valueId()) || !graphOutputs.contains(role.valueId())) {
                throw new IllegalArgumentException(
                        "gradientResults[" + index + "] is not a graph output");
            }
        }

        int currentOrder = 1;
        int expectedTargetIndex = 0;
        Set<io.github.pho001.synaptik.model.tensor.TensorId> targetIds = new HashSet<>();
        for (int index = 0; index < gradientResults.size(); index++) {
            GradientPublicationBinding role = gradientResults.get(index);
            if (role.derivativeOrder() != currentOrder) {
                if (currentOrder != 1 || role.derivativeOrder() != 2) {
                    throw new IllegalArgumentException(
                            "gradientResults must be ordered by derivative order");
                }
                currentOrder = 2;
                expectedTargetIndex = 0;
                targetIds.clear();
            }
            if (role.targetIndex() != expectedTargetIndex++) {
                throw new IllegalArgumentException(
                        "gradientResults[" + index
                                + "] targetIndex does not match stage-local order");
            }
            if (!targetIds.add(role.target())) {
                throw new IllegalArgumentException(
                        "gradientResults[" + index
                                + "] duplicates a target within derivative order "
                                + currentOrder);
            }
        }

        if (mode == CompileMode.FORWARD_ONLY) {
            if (!gradientResults.isEmpty()) {
                throw new IllegalArgumentException(
                        "FORWARD_ONLY must not contain gradient results");
            }
            if (validatedGraph.graph().nodePhases().containsValue(GraphPhase.BACKWARD)) {
                throw new IllegalArgumentException(
                        "FORWARD_ONLY must not contain BACKWARD nodes");
            }
        } else if (gradientResults.isEmpty()) {
            throw new IllegalArgumentException(
                    mode + " must contain at least one gradient result");
        }

        List<ValueId> expectedBoundary =
                new ArrayList<>(forwardOutputs.size() + gradientResults.size());
        expectedBoundary.addAll(forwardOutputs);
        Set<ValueId> seenBoundary = new HashSet<>(forwardOutputs);
        for (GradientPublicationBinding role : gradientResults) {
            if (seenBoundary.add(role.valueId())) {
                expectedBoundary.add(role.valueId());
            }
        }
        if (!validatedGraph.graph().outputs().equals(expectedBoundary)) {
            throw new IllegalArgumentException(
                    "graph output boundary does not match forward and gradient bindings");
        }
    }

}
