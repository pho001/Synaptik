package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorId;
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
 * form the graph-output prefix. Gradient roles retain request order and may share one final
 * {@link ValueId}; the graph boundary lists that value only at its first occurrence. No Tensor,
 * producer, provenance, mutable request map, or runtime/backend object is retained.</p>
 *
 * <p>This package-private record is not {@code CompileArtifacts}. It contains no publication,
 * partition, logical-memory, diagnostic, preparation, schedule, physical storage, executable, or
 * optimizer state.</p>
 *
 * @param mode non-null exact requested compile mode
 * @param validatedGraph non-null final validated graph, constraints, constant sidecar, bindable
 *     inputs, and per-node phase facts
 * @param forwardOutputs non-null ordered identity-unique forward graph-output prefix; snapshotted
 * @param gradientResults non-null ordered requested target-to-gradient roles; snapshotted and
 *     empty exactly for forward-only mode
 */
record GraphCompilation(
        CompileMode mode,
        ValidatedGraph validatedGraph,
        List<ValueId> forwardOutputs,
        List<GraphCompilation.GradientResultRole> gradientResults) {
    /**
     * Validates and snapshots one mode-selected graph-stage result.
     *
     * @param mode non-null exact requested compile mode
     * @param validatedGraph non-null final validated graph and sidecars
     * @param forwardOutputs non-null ordered identity-unique forward graph-output prefix
     * @param gradientResults non-null ordered requested target-to-gradient roles
     * @throws NullPointerException if a required component or list element is {@code null}
     * @throws IllegalArgumentException if a boundary value is absent, forward values repeat, mode
     *     and backward state disagree, or the graph output boundary is not the stable forward
     *     prefix followed by first-occurrence-distinct gradient values
     */GraphCompilation {
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(validatedGraph, "validatedGraph");
        forwardOutputs = List.copyOf(Objects.requireNonNull(forwardOutputs, "forwardOutputs"));
        gradientResults =
                List.copyOf(Objects.requireNonNull(gradientResults, "gradientResults"));

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
            GradientResultRole role = Objects.requireNonNull(
                    gradientResults.get(index), "gradientResults[" + index + "]");
            if (!values.contains(role.gradient()) || !graphOutputs.contains(role.gradient())) {
                throw new IllegalArgumentException(
                        "gradientResults[" + index + "] is not a graph output");
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
        for (GradientResultRole role : gradientResults) {
            if (seenBoundary.add(role.gradient())) {
                expectedBoundary.add(role.gradient());
            }
        }
        if (!validatedGraph.graph().outputs().equals(expectedBoundary)) {
            throw new IllegalArgumentException(
                    "graph output boundary does not match forward and gradient roles");
        }
    }

    /**
     * Immutable requested-target identity and final gradient graph-value role.
     *
     * @param target non-null {@link TensorId} of the exact requested target Tensor
     * @param gradient non-null final gradient {@link ValueId}, which is present at the graph
     *     output boundary and may be shared by several target roles
     */
    record GradientResultRole(TensorId target, ValueId gradient) {
        /**
         * Validates one target-identity-to-gradient-value role.
         *
         * @param target non-null exact requested target identity
         * @param gradient non-null final gradient graph value
         * @throws NullPointerException if either component is {@code null}
         */GradientResultRole {
            Objects.requireNonNull(target, "target");
            Objects.requireNonNull(gradient, "gradient");
        }
    }
}
