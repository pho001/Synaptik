package graph.compile.session;

import backend.partition.BackendPartitionDescriptorRegistry;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.CompiledProgram;
import graph.SemanticForwardCanonicalizer;
import graph.compile.CompileArtifacts;
import graph.compile.GraphStructureContract;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.compile.intent.BackendIntentPropagator;
import graph.compile.planning.BackendPlanningRequest;
import graph.compile.planning.BackendPlanningResult;
import graph.compile.planning.BackendPlanningService;
import graph.compile.planning.memory.MemoryPlan;
import graph.compile.planning.memory.MemoryPlanner;
import graph.compile.planning.memory.MemoryPlannerPolicy;
import graph.compile.planning.memory.MemoryPlanningInput;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.PlannedPartition;
import graph.compile.planning.region.DefaultRegionOptimizer;
import graph.compile.planning.region.OptimizedRegion;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.compile.publication.PublicationPlan;
import graph.execution.trace.PartitionCompileTrace;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.state.OptimizerState;
import graph.optimizer.state.OptimizerTrace;
import tensor.AutogradCompilationScope;
import tensor.CompileMode;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Owns one graph compile workflow from semantic forward capture to immutable compile artifacts.
 */
public final class CompileSession {
    private record ForwardGraph(
            List<Tensor> graph,
            Tensor forwardOutput,
            Map<Tensor, Tensor> publicationTensors
    ) {
        private ForwardGraph {
            graph = List.copyOf(graph == null ? List.of() : graph);
            forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
            publicationTensors = identityTensorCopy(publicationTensors);
        }
    }

    private record OptimizedGraph(
            List<Tensor> graph,
            Tensor forwardOutput,
            Map<Tensor, Tensor> publicationTensors,
            BackendIntentPlan backendIntentPlan,
            OptimizerState optimizerState
    ) {
        private OptimizedGraph {
            graph = List.copyOf(graph == null ? List.of() : graph);
            forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
            publicationTensors = identityTensorCopy(publicationTensors);
            backendIntentPlan = backendIntentPlan == null ? BackendIntentPlan.empty() : backendIntentPlan;
            optimizerState = Objects.requireNonNull(optimizerState, "optimizerState cannot be null");
        }
    }

    private record CompiledSnapshot(
            List<Tensor> graph,
            List<CompiledNode> compiledNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            Map<Tensor, CompiledNode> compiledNodeByTensor,
            CompiledNode forwardOutput,
            int forwardBoundaryNodeId,
            BackendIntentPlan backendIntentPlan
    ) {
        private CompiledSnapshot {
            graph = List.copyOf(graph == null ? List.of() : graph);
            compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
            descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
            compiledNodeByTensor = identityCompiledNodeCopy(compiledNodeByTensor);
            forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
            if (forwardBoundaryNodeId < 0) {
                throw new IllegalArgumentException("forwardBoundaryNodeId must be >= 0");
            }
            backendIntentPlan = backendIntentPlan == null ? BackendIntentPlan.empty() : backendIntentPlan;
        }
    }

    private record BackendOwnershipPlan(
            List<PlannedPartition> plannedPartitions,
            PartitionCompileTrace trace
    ) {
        private BackendOwnershipPlan {
            plannedPartitions = List.copyOf(plannedPartitions == null ? List.of() : plannedPartitions);
            trace = trace == null ? PartitionCompileTrace.empty() : trace;
        }
    }

    private record PlannedRegionsAndMemory(
            List<OptimizedRegion> optimizedRegions,
            OptimizerState optimizerState,
            MemoryPlan memoryPlan
    ) {
        private PlannedRegionsAndMemory {
            optimizedRegions = List.copyOf(optimizedRegions == null ? List.of() : optimizedRegions);
            optimizerState = Objects.requireNonNull(optimizerState, "optimizerState cannot be null");
        }
    }

    private final Tensor rootTensor;
    private final SemanticForwardCanonicalizer forwardCanonicalizer;
    private final GraphOptimizer optimizer;
    private final CompileConfig compileConfig;
    private final CompileMode compileMode;
    private final BackendPartitionDescriptorRegistry backendPartitionDescriptors;
    private final BackendPlanningService backendPlanningService;
    private final BackendIntentPlan backendIntentPlan;

    private int forwardGraphSize;
    private OptimizerTrace optimizerTrace = OptimizerTrace.empty();
    private PartitionCompileTrace partitionPlanningTrace = PartitionCompileTrace.empty();

    public CompileSession(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer optimizer,
            CompileConfig compileConfig,
            CompileMode compileMode,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors,
            BackendPlanningService backendPlanningService,
            BackendIntentPlan backendIntentPlan
    ) {
        this.rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        this.forwardCanonicalizer = forwardCanonicalizer;
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer cannot be null");
        this.compileConfig = Objects.requireNonNull(compileConfig, "compileConfig cannot be null");
        this.compileMode = Objects.requireNonNull(compileMode, "compileMode cannot be null");
        this.backendPartitionDescriptors = Objects.requireNonNull(
                backendPartitionDescriptors,
                "backendPartitionDescriptors cannot be null"
        );
        this.backendPlanningService = Objects.requireNonNull(
                backendPlanningService,
                "backendPlanningService cannot be null"
        );
        this.backendIntentPlan = backendIntentPlan == null ? BackendIntentPlan.empty() : backendIntentPlan;
    }

    public CompileArtifacts compile() {
        ForwardGraph forward = captureForwardGraph();
        forwardGraphSize = forward.graph().size();

        try (AutogradCompilationScope ignored = AutogradCompilationScope.open()) {
            BackwardGraphCompiler.Result backward = BackwardGraphCompiler.compile(
                    rootTensor,
                    forward.graph(),
                    forward.forwardOutput(),
                    compileMode
            );
            OptimizedGraph optimized = optimizeGraph(
                    backward.workingGraph(),
                    forward.forwardOutput(),
                    forward.publicationTensors(),
                    backward.supportsBackward()
            );
            optimizerTrace = optimized.optimizerState().trace();

            String graphDescription = backward.supportsBackward() ? "finalGraph" : "inference finalGraph";
            CompiledSnapshot snapshot = snapshotProgram(
                    optimized.graph(),
                    optimized.forwardOutput(),
                    graphDescription,
                    optimized.backendIntentPlan()
            );
            Map<Tensor, CompiledGradientBinding> compiledGradients = backward.supportsBackward()
                    ? GradientBindingCollector.captureCompiledGradients(
                            snapshot.graph(),
                            optimized.publicationTensors(),
                            snapshot.compiledNodeByTensor()
                    )
                    : Map.of();
            CompiledGradientBinding forwardSeedGradient = GradientBindingCollector.captureForwardSeedGradient(
                    BackwardGraphCompiler.requireForwardRoot(optimized.forwardOutput()),
                    snapshot.compiledNodeByTensor()
            );
            BackendOwnershipPlan backendPlanning = planBackendOwnership(
                    backward.supportsBackward(),
                    snapshot.compiledNodes(),
                    snapshot.descriptorIndex(),
                    snapshot.forwardOutput(),
                    compiledGradients
            );
            partitionPlanningTrace = backendPlanning.trace();

            PlannedRegionsAndMemory planning = planRegionsAndMemory(
                    backendPlanning.plannedPartitions(),
                    snapshot.compiledNodes(),
                    optimized.optimizerState(),
                    snapshot.graph(),
                    optimized.forwardOutput(),
                    backward.supportsBackward(),
                    snapshot.forwardBoundaryNodeId()
            );
            optimizerTrace = planning.optimizerState().trace();

            GraphStructureContract graphContract = GraphStructureContract.capture(rootTensor);
            PublicationPlan publicationPlan = buildPublicationPlan(
                    graphContract,
                    snapshot.graph(),
                    snapshot.compiledNodeByTensor(),
                    compiledGradients,
                    forwardSeedGradient,
                    snapshot.forwardOutput(),
                    optimized.forwardOutput(),
                    snapshot.forwardBoundaryNodeId(),
                    optimized.publicationTensors()
            );
            return new CompileArtifacts(
                    new CompiledProgram(
                            snapshot.compiledNodes(),
                            snapshot.descriptorIndex(),
                            snapshot.forwardOutput().id(),
                            snapshot.forwardBoundaryNodeId(),
                            backward.supportsBackward(),
                            backendPlanning.plannedPartitions(),
                            planning.optimizedRegions(),
                            planning.memoryPlan()
                    ),
                    publicationPlan
            );
        }
    }

    private ForwardGraph captureForwardGraph() {
        Tensor semanticForwardOutput = rootTensor.forwardOutput();
        if (forwardCanonicalizer == null) {
            return new ForwardGraph(
                    semanticForwardOutput.topologicalSort(),
                    semanticForwardOutput,
                    new IdentityHashMap<>()
            );
        }
        SemanticForwardCanonicalizer.Result canonicalized = forwardCanonicalizer.canonicalize(
                semanticForwardOutput.topologicalSort(),
                semanticForwardOutput,
                rootTensor
        );
        return new ForwardGraph(
                canonicalized.graph(),
                canonicalized.forwardOutput(),
                canonicalized.publicationTensors()
        );
    }

    private OptimizedGraph optimizeGraph(
            List<Tensor> workingGraph,
            Tensor forwardOutput,
            Map<Tensor, Tensor> publicationTensors,
            boolean supportsBackward
    ) {
        List<Tensor> graph = List.copyOf(workingGraph == null ? List.of() : workingGraph);
        BackendIntentPlan propagatedIntent = BackendIntentPropagator.propagateBackwardClosure(graph, backendIntentPlan);
        OptimizerGraphSnapshot snapshot = OptimizerGraphSnapshot.capture(graph, forwardOutput, propagatedIntent);
        OptimizerState optimizedState = optimizer.optimize(
                OptimizerState.ofGraph(
                        snapshot.graph(),
                        snapshot.forwardOutput()
                ).withExecutionMetadata(
                        supportsBackward ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD,
                        supportsBackward,
                        snapshot.graph().indexOf(snapshot.forwardOutput())
                )
        );
        return new OptimizedGraph(
                optimizedState.graph(),
                optimizedState.forwardOutput(),
                composePublicationTensors(snapshot, publicationTensors),
                snapshot.backendIntentPlan().remapThrough(optimizedState.rewriteMap()),
                optimizedState
        );
    }

    private CompiledSnapshot snapshotProgram(
            List<Tensor> graph,
            Tensor forwardOutput,
            String graphDescription,
            BackendIntentPlan backendIntentPlan
    ) {
        List<Tensor> finalGraph = List.copyOf(graph == null ? List.of() : graph);
        int forwardBoundaryNodeId = finalGraph.indexOf(forwardOutput);
        if (forwardBoundaryNodeId == -1) {
            String description = graphDescription == null ? "finalGraph" : graphDescription;
            throw new IllegalStateException("Forward output node not found in " + description + ".");
        }

        BackendIntentPlan propagatedIntent = BackendIntentPropagator.propagateBackwardClosure(finalGraph, backendIntentPlan);
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(finalGraph, propagatedIntent);
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(compiledNodes);

        IdentityHashMap<Tensor, CompiledNode> index = new IdentityHashMap<>();
        for (int i = 0; i < compiledNodes.size(); i++) {
            index.put(finalGraph.get(i), compiledNodes.get(i));
        }
        CompiledNode compiledForwardOutput = index.get(forwardOutput);
        if (compiledForwardOutput == null) {
            throw new IllegalStateException("Forward output compiled node snapshot is missing.");
        }

        return new CompiledSnapshot(
                finalGraph,
                compiledNodes,
                descriptorIndex,
                index,
                compiledForwardOutput,
                forwardBoundaryNodeId,
                propagatedIntent
        );
    }

    private BackendOwnershipPlan planBackendOwnership(
            boolean supportsBackward,
            List<CompiledNode> compiledNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode forwardOutput,
            Map<Tensor, CompiledGradientBinding> compiledGradients
    ) {
        BackendPlanningResult planning = backendPlanningService.plan(new BackendPlanningRequest(
                compileConfig.backendPlanning(),
                supportsBackward,
                compiledNodes,
                descriptorIndex,
                forwardOutput,
                compiledGradients,
                backendPartitionDescriptors
        ));
        return new BackendOwnershipPlan(planning.plannedPartitions(), planning.trace());
    }

    private PlannedRegionsAndMemory planRegionsAndMemory(
            List<PlannedPartition> plannedPartitions,
            List<CompiledNode> compiledNodes,
            OptimizerState optimizerState,
            List<Tensor> graph,
            Tensor forwardOutput,
            boolean supportsBackward,
            int forwardBoundaryNodeId
    ) {
        List<PlannedPartition> partitions = List.copyOf(plannedPartitions == null ? List.of() : plannedPartitions);
        List<CompiledNode> nodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        List<OptimizedRegion> optimizedRegions = optimizedRegions(partitions, nodes);
        OptimizerState base = optimizerState == null
                ? OptimizerState.ofGraph(graph, forwardOutput)
                : optimizerState;
        ExecutionMode executionMode = supportsBackward ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD;
        OptimizerState withMetadata = base.withExecutionMetadata(
                executionMode,
                supportsBackward,
                forwardBoundaryNodeId
        );
        Map<String, PartitionPlan> partitionPlansById = planByPartitionId(partitions);
        boolean memoryRequired = !partitions.isEmpty();
        MemoryPlan memoryPlan = (compileConfig.memoryPlanning().enabled() || memoryRequired)
                ? MemoryPlanner.plan(
                        new MemoryPlanningInput(
                                nodes,
                                optimizedRegions,
                                partitionPlansById,
                                executionMode,
                                supportsBackward,
                                forwardBoundaryNodeId
                        ),
                        MemoryPlannerPolicy.fromConfig(compileConfig.memoryPlanning().memory())
                )
                : null;
        return new PlannedRegionsAndMemory(optimizedRegions, withMetadata, memoryPlan);
    }

    private List<OptimizedRegion> optimizedRegions(
            List<PlannedPartition> plannedPartitions,
            List<CompiledNode> compiledNodes
    ) {
        if (!compileConfig.regionOptimization().enabled()) {
            return List.of();
        }
        DefaultRegionOptimizer optimizer = new DefaultRegionOptimizer();
        RegionOptimizationContext context = new RegionOptimizationContext(
                compiledNodes,
                compileConfig.regionOptimization().fuse(),
                compileConfig.regionOptimization().cpuFusion()
        );
        return plannedPartitions.stream()
                .map(PlannedPartition::partition)
                .map(partition -> optimizer.optimize(partition, context))
                .toList();
    }

    private PublicationPlan buildPublicationPlan(
            GraphStructureContract graphContract,
            List<Tensor> graph,
            Map<Tensor, CompiledNode> compiledNodeByTensor,
            Map<Tensor, CompiledGradientBinding> compiledGradients,
            CompiledGradientBinding forwardSeedGradient,
            CompiledNode compiledForwardOutput,
            Tensor forwardOutput,
            int forwardBoundaryNodeId,
            Map<Tensor, Tensor> publicationTensors
    ) {
        Tensor actualForwardRoot = BackwardGraphCompiler.requireForwardRoot(forwardOutput);
        IdentityHashMap<Tensor, Tensor> sources = new IdentityHashMap<>();
        if (publicationTensors != null) {
            sources.putAll(publicationTensors);
        }
        mapComputedForwardRootForPublish(actualForwardRoot, sources);

        ArrayList<PublicationPlan.RuntimeInputBinding> runtimeInputs = new ArrayList<>();
        ArrayList<PublicationPlan.ForwardPublicationBinding> forwardPublications = new ArrayList<>();
        ArrayList<PublicationPlan.GradientPublicationBinding> gradientPublications = new ArrayList<>();
        ArrayList<PublicationPlan.TrainableParameterBinding> trainableParameters = new ArrayList<>();
        IdentityHashMap<Tensor, Boolean> gradientClearTargets = new IdentityHashMap<>();

        Map<Tensor, CompiledNode> compiledNodes = compiledNodeByTensor == null ? Map.of() : compiledNodeByTensor;
        Map<Tensor, CompiledGradientBinding> gradients = compiledGradients == null ? Map.of() : compiledGradients;
        for (Tensor tensor : List.copyOf(graph == null ? List.of() : graph)) {
            CompiledNode node = compiledNodes.get(tensor);
            if (node == null) {
                continue;
            }
            Tensor publicationTarget = sources.getOrDefault(tensor, tensor);
            if (node.leaf()) {
                runtimeInputs.add(new PublicationPlan.RuntimeInputBinding(
                        node.id(),
                        publicationTarget,
                        runtimeInputKind(node, forwardBoundaryNodeId)
                ));
            }
            if (node.backwardNode()) {
                continue;
            }
            forwardPublications.add(new PublicationPlan.ForwardPublicationBinding(
                    publicationTarget,
                    node.id(),
                    PublicationPlan.PublicationKind.FORWARD_VALUE,
                    PublicationPlan.aliasRepairChainFor(publicationTarget)
            ));
            gradientClearTargets.put(publicationTarget, Boolean.TRUE);
            CompiledGradientBinding gradientBinding = gradients.get(publicationTarget);
            if (node.trainableParameter() && gradientBinding != null) {
                trainableParameters.add(new PublicationPlan.TrainableParameterBinding(
                        publicationTarget,
                        node.id(),
                        gradientBinding
                ));
            }
        }

        for (Map.Entry<Tensor, CompiledGradientBinding> entry : gradients.entrySet()) {
            gradientPublications.add(new PublicationPlan.GradientPublicationBinding(entry.getKey(), entry.getValue()));
            gradientClearTargets.put(entry.getKey(), Boolean.TRUE);
        }

        return new PublicationPlan(
                rootTensor,
                graphContract,
                runtimeInputs,
                new PublicationPlan.ForwardPublicationBinding(
                        rootTensor,
                        rootOutputSourceNodeId(actualForwardRoot, compiledNodes, compiledForwardOutput),
                        PublicationPlan.PublicationKind.ROOT_OUTPUT,
                        PublicationPlan.aliasRepairChainFor(rootTensor)
                ),
                forwardPublications,
                gradientPublications,
                new ArrayList<>(gradientClearTargets.keySet()),
                forwardSeedGradient,
                trainableParameters
        );
    }

    private static Map<Tensor, Tensor> composePublicationTensors(
            OptimizerGraphSnapshot snapshot,
            Map<Tensor, Tensor> publicationTensors
    ) {
        Map<Tensor, Tensor> sources = publicationTensors == null ? Map.of() : publicationTensors;
        IdentityHashMap<Tensor, Tensor> composed = new IdentityHashMap<>();
        for (Map.Entry<Tensor, Tensor> entry : snapshot.originalBySnapshot().entrySet()) {
            Tensor original = entry.getValue();
            composed.put(entry.getKey(), sources.getOrDefault(original, original));
        }
        return composed;
    }

    private static Map<String, PartitionPlan> planByPartitionId(List<PlannedPartition> plannedPartitions) {
        HashMap<String, PartitionPlan> out = new HashMap<>();
        for (PlannedPartition plannedPartition : plannedPartitions) {
            if (plannedPartition == null || plannedPartition.partition() == null || plannedPartition.plan() == null) {
                continue;
            }
            out.put(plannedPartition.partition().partitionId(), plannedPartition.plan());
        }
        return out;
    }

    private void mapComputedForwardRootForPublish(
            Tensor actualForwardRoot,
            Map<Tensor, Tensor> publicationTensors
    ) {
        if (actualForwardRoot.getOperation() != null) {
            publicationTensors.put(actualForwardRoot, rootTensor);
        }
    }

    private static PublicationPlan.RuntimeInputBindingKind runtimeInputKind(
            CompiledNode node,
            int forwardBoundaryNodeId
    ) {
        if (node.id() <= forwardBoundaryNodeId) {
            return PublicationPlan.RuntimeInputBindingKind.FORWARD_LEAF_ALIAS;
        }
        return node.backwardNode()
                ? PublicationPlan.RuntimeInputBindingKind.BACKWARD_LEAF_COPY
                : PublicationPlan.RuntimeInputBindingKind.STATIC_LEAF_COPY;
    }

    private static int rootOutputSourceNodeId(
            Tensor actualForwardRoot,
            Map<Tensor, CompiledNode> compiledNodeByTensor,
            CompiledNode compiledForwardOutput
    ) {
        CompiledNode actualRootNode = compiledNodeByTensor.get(actualForwardRoot);
        if (actualRootNode != null) {
            return actualRootNode.id();
        }
        return compiledForwardOutput.id();
    }

    private static Map<Tensor, Tensor> identityTensorCopy(Map<Tensor, Tensor> source) {
        IdentityHashMap<Tensor, Tensor> copy = new IdentityHashMap<>();
        if (source != null) {
            copy.putAll(source);
        }
        return Collections.unmodifiableMap(copy);
    }

    private static Map<Tensor, CompiledNode> identityCompiledNodeCopy(Map<Tensor, CompiledNode> source) {
        IdentityHashMap<Tensor, CompiledNode> copy = new IdentityHashMap<>();
        if (source != null) {
            copy.putAll(source);
        }
        return Collections.unmodifiableMap(copy);
    }

    public int forwardGraphSize() {
        return forwardGraphSize;
    }

    public OptimizerTrace optimizerTrace() {
        return optimizerTrace;
    }

    public PartitionCompileTrace partitionPlanningTrace() {
        return partitionPlanningTrace;
    }
}
