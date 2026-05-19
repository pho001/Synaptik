package graph.compile;

import backend.partition.BackendPartitionDescriptorRegistry;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.GradientDTypePolicy;
import graph.SemanticForwardCanonicalizer;
import graph.execution.trace.CompileTrace;
import graph.execution.trace.PartitionCompileTrace;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.intent.BackendIntentPropagator;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.memory.MemoryPlanner;
import graph.optimizer.memory.MemoryPlannerPolicy;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.partition.PlannedPartition;
import graph.optimizer.region.DefaultRegionOptimizer;
import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.RegionOptimizationContext;
import graph.optimizer.state.OptimizerState;
import graph.optimizer.state.OptimizerTrace;
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
 * Builds compile artifacts for a tensor graph.
 *
 * <p>A compiler session captures the semantic forward graph, optionally canonicalizes it, builds backward targets when
 * the selected {@link CompileMode} requires gradients, applies optimizer rules, snapshots compiled nodes, plans backend
 * partitions, and produces the memory plan used during preparation. Each call to {@link #compile()} creates a fresh
 * session and fresh artifacts; the compiler object itself only stores construction-time configuration.
 *
 * <p>The compiler is not designed for concurrent calls against mutable source tensors. The returned artifacts are
 * immutable views, but compilation reads and updates graph metadata such as gradient bindings and backend intent.
 */
public final class GraphCompiler {
    private final Tensor rootTensor;
    private final SemanticForwardCanonicalizer forwardCanonicalizer;
    private final GraphOptimizer optimizer;
    private final CompileConfig compileConfig;
    private final BackendPartitionDescriptorRegistry backendPartitionDescriptors;
    private final BackendPlanningService backendPlanningService;
    private final CompileMode compileMode;

    /**
     * Creates a compiler using the default graph policy configuration.
     *
     * @param rootTensor output tensor that anchors the graph
     * @param forwardCanonicalizer optional semantic forward canonicalizer; {@code null} disables this stage
     * @param optimizer optimizer pipeline applied after graph construction
     * @param partitionConfig backend partition planning configuration, or {@code null} for defaults
     * @param compileMode requested compile mode, or {@code null} for {@link CompileMode#AUTO}
     */
    public GraphCompiler(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer optimizer,
            CompileConfig compileConfig,
            CompileMode compileMode
    ) {
        this(
                rootTensor,
                forwardCanonicalizer,
                optimizer,
                compileConfig,
                compileMode,
                BackendPartitionDescriptorRegistry.defaults()
        );
    }

    /**
     * Creates a compiler with an explicit backend partition descriptor registry.
     *
     * @param rootTensor output tensor that anchors the graph
     * @param forwardCanonicalizer optional semantic forward canonicalizer; {@code null} disables this stage
     * @param optimizer optimizer pipeline applied after graph construction
     * @param partitionConfig backend partition planning configuration, or {@code null} for defaults
     * @param compileMode requested compile mode, or {@code null} for {@link CompileMode#AUTO}
     * @param backendPartitionDescriptors registry used to resolve backend legality and lowering plans
     * @throws NullPointerException if {@code rootTensor} or {@code optimizer} is {@code null}
     */
    public GraphCompiler(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer optimizer,
            CompileConfig compileConfig,
            CompileMode compileMode,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors
    ) {
        this.rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        this.forwardCanonicalizer = forwardCanonicalizer;
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer cannot be null");
        this.compileConfig = compileConfig == null ? CompileConfig.inference() : compileConfig;
        this.compileMode = compileMode == null ? CompileMode.AUTO : compileMode;
        this.backendPartitionDescriptors = backendPartitionDescriptors == null
                ? BackendPartitionDescriptorRegistry.defaults()
                : backendPartitionDescriptors;
        this.backendPlanningService = new BackendPlanningService(
                new BackendPlanningJobResolver(),
                this.backendPartitionDescriptors
        );
    }

    /**
     * Runs a full compile session.
     *
     * <p>The result contains both the immutable artifact bundle and timing/count metadata. Compile tracing stops at the
     * artifact boundary; kernel selection and execution timing are recorded later during preparation and execution.
     *
     * @return compile result for the latest graph state
     */
    public Result compile() {
        long t0 = System.nanoTime();
        Session session = new Session();
        CompileArtifacts artifacts = session.compile();
        CompileTrace trace = new CompileTrace(
                true,
                System.nanoTime() - t0,
                artifacts.finalGraph().size(),
                session.forwardGraphSize(),
                artifacts.supportsBackward(),
                artifacts.partitionPlanningTrace(),
                session.optimizerTrace()
        );
        return new Result(artifacts, trace);
    }

    /**
     * Pair of compile artifacts and the trace that describes the compile session.
     *
     * @param artifacts immutable artifact bundle produced by compilation
     * @param trace compile timing and node-count metadata; {@link CompileTrace#skipped()} is substituted for
     *              {@code null}
     */
    public record Result(CompileArtifacts artifacts, CompileTrace trace) {
        public Result {
            artifacts = Objects.requireNonNull(artifacts, "artifacts cannot be null");
            trace = trace == null ? CompileTrace.skipped() : trace;
        }
    }

    private final class Session {
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
        private Tensor forwardOutput;
        private int forwardEndIndex = -1;
        private boolean compiledSupportsBackward;

        private CompileArtifacts compile() {
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
            return artifacts();
        }

        private int forwardGraphSize() {
            return forwardGraph.size();
        }

        private CompileArtifacts artifacts() {
            return new CompileArtifacts(
                    rootTensor,
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
            compiledMemoryPlan = optimizedState.memoryPlan();
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

        private OptimizerTrace optimizerTrace() {
            return compiledOptimizerState == null ? OptimizerTrace.empty() : compiledOptimizerState.trace();
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
            OptimizerState planningState = base
                    .withExecutionMetadata(
                            compiledSupportsBackward ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD,
                            compiledSupportsBackward,
                            forwardEndIndex
                    )
                    .withPartitions(
                            compiledPlannedPartitions.stream().map(PlannedPartition::partition).toList(),
                            planByPartitionId()
                    )
                    .withOptimizedRegions(compiledOptimizedRegions);
            boolean memoryRequired = !compiledPlannedPartitions.isEmpty();
            compiledOptimizerState = (compileConfig.memoryPlanning().enabled() || memoryRequired)
                    ? planningState.withMemoryPlan(MemoryPlanner.plan(
                            planningState,
                            MemoryPlannerPolicy.fromConfig(compileConfig.memoryPlanning().memory())
                    ))
                    : planningState;
            compiledMemoryPlan = compiledOptimizerState.memoryPlan();
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
}
