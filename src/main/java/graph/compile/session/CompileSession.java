package graph.compile.session;

import backend.partition.BackendPartitionDescriptorRegistry;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.GradientDTypePolicy;
import graph.SemanticForwardCanonicalizer;
import graph.compile.CompileArtifacts;
import graph.compile.GraphStructureContract;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.planning.BackendPlanningRequest;
import graph.compile.planning.BackendPlanningResult;
import graph.compile.planning.BackendPlanningService;
import graph.optimizer.GraphOptimizer;
import graph.compile.intent.BackendIntentPropagator;
import graph.compile.planning.memory.MemoryPlan;
import graph.compile.planning.memory.MemoryPlanner;
import graph.compile.planning.memory.MemoryPlannerPolicy;
import graph.compile.planning.memory.MemoryPlanningInput;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.PlannedPartition;
import graph.compile.planning.region.DefaultRegionOptimizer;
import graph.compile.planning.region.OptimizedRegion;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.optimizer.state.OptimizerState;
import graph.optimizer.state.OptimizerTrace;
import graph.execution.trace.PartitionCompileTrace;
import tensor.AutogradCompilationScope;
import tensor.CompileMode;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Mutable state for one graph compile.
 */
public final class CompileSession {
    private final Tensor rootTensor;
    private final SemanticForwardCanonicalizer forwardCanonicalizer;
    private final GraphOptimizer optimizer;
    private final CompileConfig compileConfig;
    private final CompileMode compileMode;
    private final BackendPartitionDescriptorRegistry backendPartitionDescriptors;
    private final BackendPlanningService backendPlanningService;

    private final List<Tensor> finalGraph = new ArrayList<>();
    private final List<Tensor> forwardGraph = new ArrayList<>();
    private List<CompiledNode> compiledNodes = List.of();
    private CompiledTensorDescriptorIndex compiledDescriptorIndex = CompiledTensorDescriptorIndex.empty();
    private Map<Tensor, CompiledNode> compiledNodeByTensor = Map.of();
    private Map<Tensor, CompiledGradientBinding> compiledGradients = Map.of();
    private CompiledGradientBinding forwardSeedGradient;
    private CompiledNode compiledForwardOutput;
    private MemoryPlan compiledMemoryPlan;
    private OptimizerState compiledOptimizerState;
    private List<OptimizedRegion> compiledOptimizedRegions = List.of();
    private List<PlannedPartition> compiledPlannedPartitions = List.of();
    private PartitionCompileTrace compiledPartitionPlanningTrace = PartitionCompileTrace.empty();
    private GraphStructureContract graphContract = GraphStructureContract.unchecked();
    private Tensor forwardOutput;
    private int forwardEndIndex = -1;
    private boolean compiledSupportsBackward;

    public CompileSession(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer optimizer,
            CompileConfig compileConfig,
            CompileMode compileMode,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors,
            BackendPlanningService backendPlanningService
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
    }

    public CompileArtifacts compile() {
        Tensor semanticForwardOutput = rootTensor.forwardOutput();
        Map<Tensor, Tensor> sourceTensors = initializeForwardGraph(semanticForwardOutput);
        try (AutogradCompilationScope ignored = AutogradCompilationScope.open()) {
            resetAutogradBuildState();

            boolean trainableLeafInputs = hasTrainableLeafInputs();
            compiledSupportsBackward = shouldCompileBackward(trainableLeafInputs);
            List<Tensor> workingGraph = compiledSupportsBackward
                    ? buildTrainingWorkingGraph()
                    : forwardGraph;
            String graphDescription = compiledSupportsBackward ? "finalGraph" : "inference finalGraph";
            List<Tensor> optimized = optimizeWorkingGraph(workingGraph, sourceTensors);
            finalGraph.clear();
            finalGraph.addAll(optimized);
            return finishCompile(sourceTensors, compiledSupportsBackward, graphDescription);
        }
    }

    public int forwardGraphSize() {
        return forwardGraph.size();
    }

    public OptimizerTrace optimizerTrace() {
        return compiledOptimizerState == null ? OptimizerTrace.empty() : compiledOptimizerState.trace();
    }

    private List<Tensor> buildTrainingWorkingGraph() {
        GradientDTypePolicy.requireGradientSupported(rootTensor.getDataType(), "Backward execution");

        Tensor actualForwardRoot = requireForwardRoot();
        BackwardGraphBuilder.Result backward = BackwardGraphBuilder.build(forwardGraph, actualForwardRoot);

        List<Tensor> targetsToSave = new ArrayList<>();
        targetsToSave.add(forwardOutput);
        targetsToSave.addAll(backward.backwardTargets());
        Tensor superRoot = new Tensor(new int[]{1}, targetsToSave, new operations.layout.noop(), "System_Super_Root");

        List<Tensor> workingGraph = new ArrayList<>(superRoot.topologicalSort());
        workingGraph.remove(superRoot);
        return workingGraph;
    }

    private CompileArtifacts finishCompile(
            Map<Tensor, Tensor> sourceTensors,
            boolean captureGradients,
            String graphDescription
    ) {
        forwardEndIndex = finalGraph.indexOf(forwardOutput);
        if (forwardEndIndex == -1) {
            throw new IllegalStateException("Forward output node not found in " + graphDescription + ".");
        }
        mapComputedForwardRootForPublish(sourceTensors);
        rebuildCompiledNodeSnapshot(sourceTensors);
        if (captureGradients) {
            compiledGradients = GradientBindingCollector.captureCompiledGradients(
                    finalGraph,
                    sourceTensors,
                    compiledNodeByTensor
            );
        }
        forwardSeedGradient = GradientBindingCollector.captureForwardSeedGradient(
                requireForwardRoot(),
                compiledNodeByTensor
        );
        rebuildPartitionPlanningSnapshot();
        finalizeCompilePlanningArtifacts();
        graphContract = GraphStructureContract.capture(rootTensor);
        return artifacts();
    }

    private CompileArtifacts artifacts() {
        return new CompileArtifacts(
                rootTensor,
                graphContract,
                finalGraph,
                compiledNodes,
                compiledDescriptorIndex,
                compiledGradients,
                forwardSeedGradient,
                compiledForwardOutput,
                compiledMemoryPlan,
                compiledOptimizedRegions,
                compiledPlannedPartitions,
                compiledSupportsBackward,
                forwardEndIndex,
                compiledPartitionPlanningTrace
        );
    }

    private boolean hasTrainableLeafInputs() {
        for (Tensor tensor : forwardGraph) {
            if (tensor.getOperation() == null && tensor.getRequiresGrad()) {
                return true;
            }
        }
        return false;
    }

    private boolean shouldCompileBackward(boolean trainableLeafInputs) {
        return switch (compileMode) {
            case INFERENCE_ONLY -> false;
            case TRAINING, AUTO -> trainableLeafInputs;
        };
    }

    private void resetAutogradBuildState() {
        for (Tensor tensor : forwardGraph) {
            TensorInternalAccess.clearGradient(tensor);
            TensorInternalAccess.setBackward(tensor, false);
        }
        TensorInternalAccess.clearGradient(rootTensor);
    }

    private Map<Tensor, Tensor> initializeForwardGraph(Tensor semanticForwardOutput) {
        if (forwardCanonicalizer == null) {
            forwardOutput = semanticForwardOutput;
            forwardGraph.addAll(semanticForwardOutput.topologicalSort());
            return new IdentityHashMap<>();
        }
        SemanticForwardCanonicalizer.Result canonicalized = forwardCanonicalizer.canonicalize(
                semanticForwardOutput.topologicalSort(),
                semanticForwardOutput,
                rootTensor
        );
        forwardOutput = canonicalized.forwardOutput();
        forwardGraph.addAll(canonicalized.graph());
        return new IdentityHashMap<>(canonicalized.sourceTensors());
    }

    private List<Tensor> optimizeWorkingGraph(List<Tensor> workingGraph, Map<Tensor, Tensor> sourceTensors) {
        BackendIntentPropagator.propagateBackwardClosure(workingGraph);
        OptimizerGraphSnapshot snapshot = OptimizerGraphSnapshot.capture(workingGraph, forwardOutput);
        OptimizerState optimizedState = optimizer.optimize(
                OptimizerState.ofGraph(
                        new ArrayList<>(snapshot.graph()),
                        snapshot.forwardOutput()
                ).withExecutionMetadata(
                        compiledSupportsBackward ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD,
                        compiledSupportsBackward,
                        snapshot.graph().indexOf(snapshot.forwardOutput())
                )
        );
        List<Tensor> optimized = optimizedState.graph();
        IdentityHashMap<Tensor, Tensor> composed = new IdentityHashMap<>();
        for (Map.Entry<Tensor, Tensor> entry : snapshot.originalBySnapshot().entrySet()) {
            Tensor original = entry.getValue();
            composed.put(entry.getKey(), sourceTensors.getOrDefault(original, original));
        }
        sourceTensors.clear();
        sourceTensors.putAll(composed);
        forwardOutput = optimizedState.forwardOutput();
        compiledOptimizerState = optimizedState;
        return optimized;
    }

    private void rebuildCompiledNodeSnapshot(Map<Tensor, Tensor> sourceTensors) {
        BackendIntentPropagator.propagateBackwardClosure(finalGraph);
        compiledNodes = CompiledNode.snapshot(finalGraph, sourceTensors);
        compiledDescriptorIndex = CompiledTensorDescriptorBuilder.build(compiledNodes);
        IdentityHashMap<Tensor, CompiledNode> index = new IdentityHashMap<>();
        for (CompiledNode node : compiledNodes) {
            index.put(node.semanticTensor(), node);
        }
        compiledNodeByTensor = Map.copyOf(index);
        compiledForwardOutput = compiledNodeByTensor.get(forwardOutput);
        if (compiledForwardOutput == null) {
            throw new IllegalStateException("Forward output compiled node snapshot is missing.");
        }
    }

    private void rebuildPartitionPlanningSnapshot() {
        BackendPlanningResult planning = backendPlanningService.plan(new BackendPlanningRequest(
                compileConfig.backendPlanning(),
                compiledSupportsBackward,
                compiledNodes,
                compiledDescriptorIndex,
                compiledForwardOutput,
                compiledGradients,
                backendPartitionDescriptors
        ));
        compiledPlannedPartitions = planning.plannedPartitions();
        compiledPartitionPlanningTrace = planning.trace();
    }

    private void finalizeCompilePlanningArtifacts() {
        compiledOptimizedRegions = compileConfig.regionOptimization().enabled()
                ? compiledPlannedPartitions.stream()
                        .map(PlannedPartition::partition)
                        .map(partition -> new DefaultRegionOptimizer().optimize(
                                partition,
                                new RegionOptimizationContext(
                                        compiledNodes,
                                        compileConfig.regionOptimization().fuse(),
                                        compileConfig.regionOptimization().cpuFusion()
                                )
                        ))
                        .toList()
                : List.of();
        OptimizerState base = compiledOptimizerState == null
                ? OptimizerState.ofGraph(finalGraph, compiledForwardOutput.semanticTensor())
                : compiledOptimizerState;
        ExecutionMode executionMode = compiledSupportsBackward ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD;
        compiledOptimizerState = base.withExecutionMetadata(
                executionMode,
                compiledSupportsBackward,
                forwardEndIndex
        );
        boolean memoryRequired = !compiledPlannedPartitions.isEmpty();
        compiledMemoryPlan = (compileConfig.memoryPlanning().enabled() || memoryRequired)
                ? MemoryPlanner.plan(
                        new MemoryPlanningInput(
                                compiledNodes,
                                compiledOptimizedRegions,
                                planByPartitionId(),
                                executionMode,
                                compiledSupportsBackward,
                                forwardEndIndex
                        ),
                        MemoryPlannerPolicy.fromConfig(compileConfig.memoryPlanning().memory())
                )
                : null;
    }

    private Map<String, PartitionPlan> planByPartitionId() {
        java.util.HashMap<String, PartitionPlan> out = new java.util.HashMap<>();
        for (PlannedPartition plannedPartition : compiledPlannedPartitions) {
            if (plannedPartition == null || plannedPartition.partition() == null || plannedPartition.plan() == null) {
                continue;
            }
            out.put(plannedPartition.partition().partitionId(), plannedPartition.plan());
        }
        return Map.copyOf(out);
    }

    private Tensor requireForwardRoot() {
        List<Tensor> inputs = forwardOutput == null ? null : forwardOutput.getPrevTensors();
        if (inputs == null || inputs.size() != 1 || inputs.get(0) == null) {
            throw new IllegalStateException("System forward output must have exactly one input.");
        }
        return inputs.get(0);
    }

    private void mapComputedForwardRootForPublish(Map<Tensor, Tensor> sourceTensors) {
        Tensor actualForwardRoot = requireForwardRoot();
        if (actualForwardRoot.getOperation() != null) {
            sourceTensors.put(actualForwardRoot, rootTensor);
        }
    }
}
