package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.PublicationBinding;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.BackendOwnerPlanning;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlan;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning;
import io.github.pho001.synaptik.planning.partition.MaximalSameOwnerPartitioning;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Owns package-private mode routing for capture, first-order expansion, and exact optimization.
 *
 * <p>{@link CompileMode#FORWARD_ONLY} captures and compiles only the requested forward boundary.
 * The two backward-capable modes preflight one scalar first-order request, construct gradients
 * with public Tensor operations, and capture forward outputs and gradient roots together once.
 * Every mode then passes its single immutable graph through inference, mandatory
 * canonicalization, bounded exact optional optimization, and final validation.</p>
 *
 * <p>The original direct entry returns internal graph-stage state. A package-private complete
 * overload additionally derives publication roles, logical constants and diagnostics, selects
 * backend ownership once per final node, and invokes Planning's partition and logical-memory
 * operations to return immutable compile artifacts. Neither entry allocates storage, lowers a
 * backend, prepares or executes work, performs optimizer updates or higher-order differentiation,
 * or exposes a public compiler facade.</p>
 */
final class GraphCompiler {
    private GraphCompiler() {}

    /**
     * Compiles one forward-only or combined first-order Tensor expression graph.
     *
     * <p>Top-level arguments are checked in declaration order before graph construction. Known
     * unsupported first-order facts fail during complete preflight before a seed, derivative
     * constant, or formula Tensor is created. Failures after expansion begins may consume
     * temporary opaque Tensor IDs; identifiers are never rolled back or reused.</p>
     *
     * @param mode non-null graph-scope mode
     * @param forwardOutputs non-null, non-empty ordered forward boundary; exact Tensor references
     *     and resolved logical values must be unique, and the list is not mutated
     * @param firstOrderRequest non-null optional scalar-objective request, absent exactly for
     *     {@link CompileMode#FORWARD_ONLY} and present for both backward-capable modes
     * @param forwardConstants non-null ordered explicit logical-splat bindings for reachable
     *     forward leaves
     * @param optimizationConfig non-null permission controlling the optional exact pass sequence;
     *     inference, canonicalization, and validation remain mandatory
     * @return a non-null immutable mode-neutral graph compilation with ordered forward values and
     *     target-specific gradient roles
     * @throws NullPointerException if a top-level argument or required list element is
     *     {@code null}, checked in declaration order
     * @throws IllegalArgumentException if the forward boundary is empty or duplicates a Tensor or
     *     logical value, the mode/request matrix is invalid, preflight rejects the request,
     *     ingress is invalid, or capture, inference, validation, optimization, or final boundary
     *     validation fails
     */
    static GraphCompilation compile(
            CompileMode mode,
            List<Tensor> forwardOutputs,
            Optional<AutogradPreflight.FirstOrderRequest> firstOrderRequest,
            CompileTimeConstantGraph.Ingress forwardConstants,
            GraphOptimizationConfig optimizationConfig) {
        Objects.requireNonNull(mode, "mode");
        validateForwardOutputs(forwardOutputs);
        Objects.requireNonNull(firstOrderRequest, "firstOrderRequest");
        Objects.requireNonNull(forwardConstants, "forwardConstants");
        Objects.requireNonNull(optimizationConfig, "optimizationConfig");

        if (mode == CompileMode.FORWARD_ONLY) {
            if (firstOrderRequest.isPresent()) {
                throw new IllegalArgumentException(
                        "FORWARD_ONLY must not include a first-order request");
            }
            CompileTimeConstantGraph captured =
                    GraphCapture.capture(forwardOutputs, forwardConstants);
            ValidatedGraph inferred = CapturedGraphInference.inferAndValidate(captured);
            ValidatedGraph optimized =
                    ForwardGraphOptimization.optimize(inferred, optimizationConfig);
            List<ValueId> finalForward = List.copyOf(optimized.graph().outputs());
            return new GraphCompilation(mode, optimized, finalForward, List.of());
        }
        if (firstOrderRequest.isEmpty()) {
            throw new IllegalArgumentException(mode + " requires a first-order request");
        }

        AutogradPreflight.Plan plan =
                AutogradPreflight.preflight(
                        mode,
                        forwardOutputs,
                        firstOrderRequest.orElseThrow(),
                        forwardConstants);
        FirstOrderAutograd.Expansion expansion =
                FirstOrderAutograd.expand(plan, forwardConstants);
        GraphCapture.CombinedCapture captured = GraphCapture.captureCombined(
                forwardOutputs,
                expansion.targetGradients(),
                expansion.originalProducers(),
                expansion.ingress());
        ValidatedGraph inferred =
                CapturedGraphInference.inferAndValidate(captured.constantGraph());
        ValidatedGraph optimized =
                ForwardGraphOptimization.optimize(inferred, optimizationConfig);

        List<ValueId> finalForward = List.copyOf(
                optimized.graph().outputs().subList(0, captured.forwardOutputCount()));
        List<GraphCompilation.GradientResultRole> gradientResults =
                new ArrayList<>(expansion.targetGradients().size());
        for (int index = 0; index < expansion.targetGradients().size(); index++) {
            int ordinal = captured.gradientOutputOrdinals().get(index);
            gradientResults.add(new GraphCompilation.GradientResultRole(
                    expansion.targetGradients().get(index).target().id(),
                    optimized.graph().outputs().get(ordinal)));
        }
        return new GraphCompilation(mode, optimized, finalForward, gradientResults);
    }

    /**
     * Compiles one graph and derives its immutable publication and backend-neutral planning
     * artifacts.
     *
     * <p>All nine top-level arguments are validated in declaration order before graph
     * construction. The existing graph-stage compile entry is invoked exactly once. Publication,
     * constant, and diagnostic snapshots are then built from its final graph before each node is
     * queried in stored order, followed by maximal partitioning, logical-memory derivation, and
     * final aggregate cross-validation.</p>
     *
     * @param mode non-null graph-scope mode
     * @param forwardOutputs non-null, non-empty ordered requested forward boundary
     * @param firstOrderRequest non-null optional first-order request with mode-compatible presence
     * @param forwardConstants non-null explicit logical-splat ingress
     * @param optimizationConfig non-null exact graph-optimization permission
     * @param backendIntent non-null optional hard backend target
     * @param partitionScoringConfig non-null optional coarse device-class preference
     * @param capabilityProviders non-null ordered capability providers; list elements are first
     *     inspected only when a graph node is planned
     * @param availabilitySnapshots non-null availability snapshots; list elements are first
     *     inspected only when a graph node is planned
     * @return non-null immutable complete compile artifacts retaining no provider or snapshot
     * @throws NullPointerException if a required argument or nested value is {@code null}
     * @throws IllegalArgumentException if graph compilation, publication, planning composition,
     *     partitioning, memory derivation, or artifact cross-validation rejects the request
     * @throws IllegalStateException if a graph node has no hard-eligible backend; the message adds
     *     node occurrence context and the Planning failure is retained as its cause
     * @throws RuntimeException if a capability provider throws; the same failure propagates
     *     unchanged
     */
    static CompileArtifacts compile(
            CompileMode mode,
            List<Tensor> forwardOutputs,
            Optional<AutogradPreflight.FirstOrderRequest> firstOrderRequest,
            CompileTimeConstantGraph.Ingress forwardConstants,
            GraphOptimizationConfig optimizationConfig,
            BackendIntent backendIntent,
            PartitionScoringConfig partitionScoringConfig,
            List<BackendCapabilityProvider> capabilityProviders,
            List<BackendAvailabilitySnapshot> availabilitySnapshots) {
        Objects.requireNonNull(mode, "mode");
        validateForwardOutputs(forwardOutputs);
        Objects.requireNonNull(firstOrderRequest, "firstOrderRequest");
        Objects.requireNonNull(forwardConstants, "forwardConstants");
        Objects.requireNonNull(optimizationConfig, "optimizationConfig");
        Objects.requireNonNull(backendIntent, "backendIntent");
        Objects.requireNonNull(partitionScoringConfig, "partitionScoringConfig");
        Objects.requireNonNull(capabilityProviders, "capabilityProviders");
        Objects.requireNonNull(availabilitySnapshots, "availabilitySnapshots");

        GraphCompilation compilation = compile(
                mode,
                forwardOutputs,
                firstOrderRequest,
                forwardConstants,
                optimizationConfig);
        ValidatedGraph validated = compilation.validatedGraph();
        CompiledGraphModel graph = validated.graph();

        List<PublicationBinding> forwardBindings =
                new ArrayList<>(compilation.forwardOutputs().size());
        for (int index = 0; index < compilation.forwardOutputs().size(); index++) {
            forwardBindings.add(new PublicationBinding(
                    forwardOutputs.get(index).id(),
                    compilation.forwardOutputs().get(index)));
        }
        List<PublicationBinding> gradientBindings =
                new ArrayList<>(compilation.gradientResults().size());
        for (GraphCompilation.GradientResultRole role : compilation.gradientResults()) {
            gradientBindings.add(new PublicationBinding(role.target(), role.gradient()));
        }
        PublicationPlan publication =
                new PublicationPlan(graph, forwardBindings, gradientBindings);

        List<ValueId> bindableInputs = new ArrayList<>();
        List<CompileConstantPlan.ConstantSource> constantSources = new ArrayList<>();
        for (ValueId input : graph.inputs()) {
            CompileTimeConstantGraph.Splat splat = validated.constants().get(input);
            if (splat == null) {
                bindableInputs.add(input);
            } else {
                constantSources.add(
                        new CompileConstantPlan.ConstantSource(input, splat.value()));
            }
        }
        CompileConstantPlan constants =
                new CompileConstantPlan(bindableInputs, constantSources);
        CompileDiagnostics diagnostics = new CompileDiagnostics(validated.constraints());

        Map<ValueId, TensorDescriptor> descriptors = new HashMap<>();
        for (GraphValue value : graph.values()) {
            descriptors.put(value.id(), value.descriptor());
        }
        Map<NodeId, BackendId> ownershipByNodeId = new LinkedHashMap<>();
        for (int nodeIndex = 0; nodeIndex < graph.nodes().size(); nodeIndex++) {
            CompiledNode node = graph.nodes().get(nodeIndex);
            List<TensorDescriptor> inputs = new ArrayList<>(node.inputs().size());
            for (ValueId input : node.inputs()) {
                inputs.add(descriptors.get(input));
            }
            List<TensorDescriptor> outputs = new ArrayList<>(node.outputs().size());
            for (ValueId output : node.outputs()) {
                outputs.add(descriptors.get(output));
            }
            OperationCapabilityQuery query =
                    new OperationCapabilityQuery(node.operation(), inputs, outputs);
            BackendId owner;
            try {
                owner = BackendOwnerPlanning.selectOwner(
                        query,
                        backendIntent,
                        capabilityProviders,
                        availabilitySnapshots,
                        partitionScoringConfig);
            } catch (IllegalStateException failure) {
                if (!"no hard-eligible backend is available for ownership selection"
                                .equals(failure.getMessage())
                        || failure.getCause() == null
                        || !"no hard-eligible backend is available for ownership selection"
                                .equals(failure.getCause().getMessage())) {
                    throw failure;
                }
                throw new IllegalStateException(
                        "nodes[" + nodeIndex + "] " + node.id() + " "
                                + node.operation().kind().getClass().getName() + "."
                                + node.operation().kind().name() + ": "
                                + failure.getMessage(),
                        failure.getCause());
            }
            ownershipByNodeId.put(node.id(), owner);
        }

        List<PlannedPartition> partitions =
                MaximalSameOwnerPartitioning.partition(graph, ownershipByNodeId);
        LogicalMemoryPlan memory = LogicalMemoryPlanning.plan(graph, partitions);
        return new CompileArtifacts(
                mode,
                graph,
                partitions,
                memory,
                publication,
                constants,
                diagnostics);
    }

    private static void validateForwardOutputs(List<Tensor> forwardOutputs) {
        Objects.requireNonNull(forwardOutputs, "forwardOutputs");
        if (forwardOutputs.isEmpty()) {
            throw new IllegalArgumentException("forwardOutputs must not be empty");
        }
        IdentityHashMap<Tensor, Integer> positions = new IdentityHashMap<>();
        IdentityHashMap<TensorProducer, int[]> producerOutputPositions = new IdentityHashMap<>();
        for (int index = 0; index < forwardOutputs.size(); index++) {
            Tensor output = Objects.requireNonNull(
                    forwardOutputs.get(index), "forwardOutputs[" + index + "]");
            Integer first = positions.putIfAbsent(output, index);
            if (first != null) {
                throw new IllegalArgumentException(
                        "forwardOutputs[" + index + "] duplicates forwardOutputs[" + first + "]");
            }
            TensorProvenance provenance = output.provenance().orElse(null);
            if (provenance != null) {
                int[] outputPositions = producerOutputPositions.computeIfAbsent(
                        provenance.producer(),
                        producer -> {
                            int[] created = new int[producer.outputCount()];
                            Arrays.fill(created, -1);
                            return created;
                        });
                int firstLogical = outputPositions[provenance.outputIndex()];
                if (firstLogical >= 0) {
                    throw new IllegalArgumentException(
                            "forwardOutputs[" + index + "] duplicates forwardOutputs["
                                    + firstLogical + "] logical producer output");
                }
                outputPositions[provenance.outputIndex()] = index;
            }
        }
    }
}
