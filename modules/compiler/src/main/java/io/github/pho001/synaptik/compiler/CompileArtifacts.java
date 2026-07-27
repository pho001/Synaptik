package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlan;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Collects the immutable non-executable result of complete graph compilation.
 *
 * <p>The aggregate retains the exact final graph and output-only plans, snapshots partition-list
 * membership while retaining exact partition references, and cross-validates mode, graph
 * boundary, maximal partitioning, logical memory, input-source roles, and diagnostics. It is a
 * compile-time recipe, not a public compile facade or prepared execution.</p>
 *
 * <p>No component retains a live provider, availability request, selected device, route, kernel,
 * physical buffer, byte count, transfer, executable, schedule, runtime residency, or mutable run
 * state.</p>
 *
 * @param mode non-null exact graph-scope mode
 * @param graph non-null exact final immutable graph
 * @param partitions non-null exact maximal graph-order partition recipes; membership snapshotted
 * @param memory non-null logical-memory plan derived from this graph and partition list
 * @param publication non-null ordered publication roles owning this exact graph reference
 * @param constants non-null complete graph-input source-role classification
 * @param diagnostics non-null successful-compile deferred diagnostics
 */
public record CompileArtifacts(
        CompileMode mode,
        CompiledGraphModel graph,
        List<PlannedPartition> partitions,
        LogicalMemoryPlan memory,
        PublicationPlan publication,
        CompileConstantPlan constants,
        CompileDiagnostics diagnostics) {
    /**
     * Validates and snapshots one complete immutable compile recipe.
     *
     * @param mode non-null exact graph-scope mode
     * @param graph non-null exact final immutable graph
     * @param partitions non-null ordered exact maximal graph partitions
     * @param memory non-null logical-memory plan for the supplied graph and partitions
     * @param publication non-null publication plan retaining the exact graph reference
     * @param constants non-null complete graph-input source-role classification
     * @param diagnostics non-null successful-compile deferred diagnostics
     * @throws NullPointerException if a component or partition element is {@code null}
     * @throws IllegalArgumentException if components disagree about graph identity, mode,
     *     partitioning, logical memory, source roles, descriptor types, gradient eligibility, or
     *     diagnostic node membership
     */
    public CompileArtifacts {
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(partitions, "partitions");
        Objects.requireNonNull(memory, "memory");
        Objects.requireNonNull(publication, "publication");
        Objects.requireNonNull(constants, "constants");
        Objects.requireNonNull(diagnostics, "diagnostics");

        for (int index = 0; index < partitions.size(); index++) {
            Objects.requireNonNull(partitions.get(index), "partitions[" + index + "]");
        }
        partitions = List.copyOf(partitions);

        if (publication.graph() != graph) {
            throw new IllegalArgumentException(
                    "publication graph must be the exact graph reference");
        }
        if (mode == CompileMode.FORWARD_ONLY) {
            if (!publication.gradientResults().isEmpty()) {
                throw new IllegalArgumentException(
                        "FORWARD_ONLY must not contain gradient results");
            }
            if (graph.nodePhases().containsValue(GraphPhase.BACKWARD)) {
                throw new IllegalArgumentException(
                        "FORWARD_ONLY must not contain BACKWARD nodes");
            }
        } else if (publication.gradientResults().isEmpty()) {
            throw new IllegalArgumentException(
                    mode + " must contain at least one gradient result");
        }

        LogicalMemoryPlan derivedMemory = LogicalMemoryPlanning.plan(graph, partitions);
        if (!memory.equals(derivedMemory)) {
            throw new IllegalArgumentException(
                    "memory does not match graph and partitions");
        }

        validateConstants(graph, constants);

        Set<NodeId> nodeIds = new HashSet<>();
        graph.nodes().forEach(node -> nodeIds.add(node.id()));
        for (int index = 0; index < diagnostics.deferredConstraints().size(); index++) {
            if (!nodeIds.contains(diagnostics.deferredConstraints().get(index).nodeId())) {
                throw new IllegalArgumentException(
                        "diagnostics.deferredConstraints[" + index
                                + "] references unknown "
                                + diagnostics.deferredConstraints().get(index).nodeId());
            }
        }
    }

    private static void validateConstants(
            CompiledGraphModel graph,
            CompileConstantPlan constants) {
        Map<ValueId, TensorDescriptor> descriptors = new HashMap<>();
        for (GraphValue value : graph.values()) {
            descriptors.put(value.id(), value.descriptor());
        }

        int bindableIndex = 0;
        int constantIndex = 0;
        for (int inputIndex = 0; inputIndex < graph.inputs().size(); inputIndex++) {
            ValueId input = graph.inputs().get(inputIndex);
            if (bindableIndex < constants.bindableInputs().size()
                    && constants.bindableInputs().get(bindableIndex).equals(input)) {
                bindableIndex++;
                continue;
            }
            if (constantIndex < constants.constantSources().size()
                    && constants.constantSources().get(constantIndex).valueId().equals(input)) {
                CompileConstantPlan.ConstantSource source =
                        constants.constantSources().get(constantIndex++);
                TensorDescriptor descriptor = descriptors.get(input);
                if (source.value().dataType() != descriptor.dataType()) {
                    throw new IllegalArgumentException(
                            "constantSources[" + (constantIndex - 1) + "] data type "
                                    + source.value().dataType()
                                    + " does not match graph input descriptor "
                                    + descriptor.dataType());
                }
                if (descriptor.requiresGrad()) {
                    throw new IllegalArgumentException(
                            "constantSources[" + (constantIndex - 1)
                                    + "] fixes a gradient-eligible graph input");
                }
                continue;
            }
            throw new IllegalArgumentException(
                    "constants do not classify graph.inputs[" + inputIndex + "] " + input
                            + " in graph-input order");
        }
        if (bindableIndex != constants.bindableInputs().size()) {
            throw new IllegalArgumentException(
                    "bindableInputs contains a value outside the graph-input classification");
        }
        if (constantIndex != constants.constantSources().size()) {
            throw new IllegalArgumentException(
                    "constantSources contains a value outside the graph-input classification");
        }
    }
}
