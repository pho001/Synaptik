package graph.compile;

import backend.partition.BackendPartitionDescriptorRegistry;
import backend.runtime.ExecutionMode;
import config.optimizer.CpuFusionConfig;
import config.optimizer.CpuRegionConfig;
import config.optimizer.FuseConfig;
import config.optimizer.MemoryConfig;
import config.optimizer.OffloadConfig;
import config.optimizer.OptimizerConfig;
import config.optimizer.PartitionConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.SemanticForwardCanonicalizer;
import graph.execution.trace.CompileTrace;
import graph.execution.trace.PartitionCompileTrace;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.memory.MemoryPlanner;
import graph.optimizer.memory.MemoryPlannerPolicy;
import graph.optimizer.partition.BackendCandidatePartition;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.region.DefaultRegionOptimizer;
import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.RegionOptimizationContext;
import graph.optimizer.state.OptimizerState;
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
    private final PartitionConfig partitionConfig;
    private final OffloadConfig offloadConfig;
    private final CpuRegionConfig cpuRegionConfig;
    private final FuseConfig fuseConfig;
    private final CpuFusionConfig cpuFusionConfig;
    private final MemoryConfig memoryConfig;
    private final BackendPartitionDescriptorRegistry backendPartitionDescriptors;
    private final CompileMode compileMode;

    /**
     * Creates a compiler using the default backend partition descriptor registry.
     *
     * @param rootTensor output tensor that anchors the graph
     * @param forwardCanonicalizer optional semantic forward canonicalizer; {@code null} disables this stage
     * @param optimizer optimizer pipeline applied after graph construction
     * @param optimizerConfig graph optimizer policy configuration, or {@code null} for defaults
     * @param compileMode requested compile mode, or {@code null} for {@link CompileMode#AUTO}
     */
    public GraphCompiler(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer optimizer,
            OptimizerConfig optimizerConfig,
            CompileMode compileMode
    ) {
        this(
                rootTensor,
                forwardCanonicalizer,
                optimizer,
                optimizerConfig == null ? PartitionConfig.defaults() : optimizerConfig.partition(),
                optimizerConfig == null ? OffloadConfig.defaults() : optimizerConfig.offload(),
                optimizerConfig == null ? CpuRegionConfig.defaults() : optimizerConfig.cpuRegion(),
                optimizerConfig == null ? FuseConfig.inferenceDefaults() : optimizerConfig.fuse(),
                optimizerConfig == null ? CpuFusionConfig.defaults() : optimizerConfig.cpuFusion(),
                optimizerConfig == null ? MemoryConfig.defaults() : optimizerConfig.memory(),
                compileMode,
                BackendPartitionDescriptorRegistry.defaults()
        );
    }

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
            PartitionConfig partitionConfig,
            CompileMode compileMode
    ) {
        this(
                rootTensor,
                forwardCanonicalizer,
                optimizer,
                partitionConfig,
                OffloadConfig.defaults(),
                CpuRegionConfig.defaults(),
                FuseConfig.inferenceDefaults(),
                CpuFusionConfig.defaults(),
                MemoryConfig.defaults(),
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
            PartitionConfig partitionConfig,
            CompileMode compileMode,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors
    ) {
        this(
                rootTensor,
                forwardCanonicalizer,
                optimizer,
                partitionConfig,
                OffloadConfig.defaults(),
                CpuRegionConfig.defaults(),
                FuseConfig.inferenceDefaults(),
                CpuFusionConfig.defaults(),
                MemoryConfig.defaults(),
                compileMode,
                backendPartitionDescriptors
        );
    }

    /**
     * Creates a compiler with explicit graph policy and backend descriptor configuration.
     *
     * @param rootTensor output tensor that anchors the graph
     * @param forwardCanonicalizer optional semantic forward canonicalizer; {@code null} disables this stage
     * @param optimizer optimizer pipeline applied after graph construction
     * @param partitionConfig shared partition planning limits, or {@code null} for defaults
     * @param offloadConfig accelerator/offload policy, or {@code null} for defaults
     * @param cpuRegionConfig CPU execution region policy, or {@code null} for defaults
     * @param fuseConfig region fusion config, or {@code null} for inference defaults
     * @param cpuFusionConfig CPU fused-loop policy, or {@code null} for defaults
     * @param memoryConfig memory planner policy, or {@code null} for defaults
     * @param compileMode requested compile mode, or {@code null} for {@link CompileMode#AUTO}
     * @param backendPartitionDescriptors registry used to resolve backend legality and lowering plans
     * @throws NullPointerException if {@code rootTensor} or {@code optimizer} is {@code null}
     */
    public GraphCompiler(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer optimizer,
            PartitionConfig partitionConfig,
            OffloadConfig offloadConfig,
            CpuRegionConfig cpuRegionConfig,
            FuseConfig fuseConfig,
            CpuFusionConfig cpuFusionConfig,
            MemoryConfig memoryConfig,
            CompileMode compileMode,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors
    ) {
        this.rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        this.forwardCanonicalizer = forwardCanonicalizer;
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer cannot be null");
        this.partitionConfig = partitionConfig == null ? PartitionConfig.defaults() : partitionConfig;
        this.offloadConfig = offloadConfig == null ? OffloadConfig.defaults() : offloadConfig;
        this.cpuRegionConfig = cpuRegionConfig == null ? CpuRegionConfig.defaults() : cpuRegionConfig;
        this.fuseConfig = fuseConfig == null ? FuseConfig.inferenceDefaults() : fuseConfig;
        this.cpuFusionConfig = cpuFusionConfig == null ? CpuFusionConfig.defaults() : cpuFusionConfig;
        this.memoryConfig = memoryConfig == null ? MemoryConfig.defaults() : memoryConfig;
        this.compileMode = compileMode == null ? CompileMode.AUTO : compileMode;
        this.backendPartitionDescriptors = backendPartitionDescriptors == null
                ? BackendPartitionDescriptorRegistry.defaults()
                : backendPartitionDescriptors;
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
                artifacts.partitionPlanningTrace()
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
        private Map<Tensor, CompiledNode> compiledNodeByTensor = Map.of();
        private Map<Tensor, CompiledGradientBinding> compiledGradients = Map.of();
        private CompiledGradientBinding forwardSeedGradient;
        private CompiledNode compiledForwardOutput;
        private MemoryPlan compiledMemoryPlan;
        private OptimizerState compiledOptimizerState;
        private List<Partition> compiledPartitions = List.of();
        private List<PartitionPlan> compiledBackendPlans = List.of();
        private List<BackendCandidatePartition> compiledBackendSelectionCandidates = List.of();
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
                if (!compiledSupportsBackward) {
                    List<Tensor> optimizedForward = optimizeWorkingGraph(forwardGraph, sourceTensors);
                    finalGraph.addAll(optimizedForward);
                    forwardEndIndex = finalGraph.indexOf(forwardOutput);
                    if (forwardEndIndex == -1) {
                        throw new IllegalStateException("Forward output node not found in inference finalGraph.");
                    }
                    mapComputedForwardRootForPublish(sourceTensors);
                    rebuildCompiledNodeSnapshot(sourceTensors);
                    forwardSeedGradient = GradientBindingCollector.captureForwardSeedGradient(
                            requireForwardRoot(),
                            compiledNodeByTensor
                    );
                    rebuildPartitionPlanningSnapshot();
                    completeLoweringReadyOptimizerState();
                    return artifacts();
                }

                if (rootTensor.getDataType() == tensor.DataType.BOOL || rootTensor.getDataType() == tensor.DataType.INT32) {
                    throw new UnsupportedOperationException("BOOL/INT32 root tensors do not support backward execution.");
                }

                Tensor actualForwardRoot = requireForwardRoot();
                BackwardGraphBuilder.Result backward = BackwardGraphBuilder.build(forwardGraph, actualForwardRoot);

                List<Tensor> targetsToSave = new ArrayList<>();
                targetsToSave.add(forwardOutput);
                targetsToSave.addAll(backward.backwardTargets());
                Tensor superRoot = new Tensor(new int[]{1}, targetsToSave, new operations.layout.noop(), "System_Super_Root");

                finalGraph.addAll(superRoot.topologicalSort());
                finalGraph.remove(superRoot);

                List<Tensor> optimized = optimizeWorkingGraph(finalGraph, sourceTensors);
                finalGraph.clear();
                finalGraph.addAll(optimized);
                forwardEndIndex = finalGraph.indexOf(forwardOutput);
                if (forwardEndIndex == -1) {
                    throw new IllegalStateException("Forward output node not found in finalGraph.");
                }
                mapComputedForwardRootForPublish(sourceTensors);
                rebuildCompiledNodeSnapshot(sourceTensors);
                compiledGradients = GradientBindingCollector.captureCompiledGradients(
                        finalGraph,
                        sourceTensors,
                        compiledNodeByTensor
                );
                forwardSeedGradient = GradientBindingCollector.captureForwardSeedGradient(
                        requireForwardRoot(),
                        compiledNodeByTensor
                );
                rebuildPartitionPlanningSnapshot();
                completeLoweringReadyOptimizerState();
                return artifacts();
            }
        }

        private int forwardGraphSize() {
            return forwardGraph.size();
        }

        private CompileArtifacts artifacts() {
            return new CompileArtifacts(
                    rootTensor,
                    finalGraph,
                    compiledNodes,
                    compiledGradients,
                    forwardSeedGradient,
                    compiledForwardOutput,
                    compiledMemoryPlan,
                    compiledOptimizerState,
                    compiledPartitions,
                    compiledBackendPlans,
                    compiledBackendSelectionCandidates,
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
            compiledNodes = CompiledNode.snapshot(finalGraph, sourceTensors);
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
            PartitionPlanningSnapshotBuilder.Snapshot snapshot = PartitionPlanningSnapshotBuilder.build(
                    partitionConfig,
                    offloadConfig,
                    cpuRegionConfig,
                    compiledSupportsBackward,
                    compiledNodes,
                    compiledForwardOutput,
                    compiledGradients,
                    backendPartitionDescriptors
            );
            compiledPartitions = snapshot.partitions();
            compiledBackendPlans = snapshot.backendPlans();
            compiledBackendSelectionCandidates = snapshot.backendSelectionCandidates();
            compiledPartitionPlanningTrace = snapshot.trace();
        }

        private void completeLoweringReadyOptimizerState() {
            if (compiledOptimizerState != null
                    && !compiledOptimizerState.optimizedRegions().isEmpty()
                    && compiledOptimizerState.memoryPlan() != null) {
                return;
            }
            if (compiledBackendSelectionCandidates.isEmpty() || compiledPartitions.isEmpty()) {
                return;
            }
            List<OptimizedRegion> optimizedRegions = compiledPartitions.stream()
                    .map(partition -> new DefaultRegionOptimizer().optimize(
                            partition,
                            new RegionOptimizationContext(
                                    compiledNodes,
                                    fuseConfig,
                                    cpuFusionConfig
                            )
                    ))
                    .toList();
            OptimizerState base = compiledOptimizerState == null
                    ? OptimizerState.ofGraph(finalGraph, compiledForwardOutput.semanticTensor())
                    : compiledOptimizerState;
            OptimizerState loweringReady = base
                    .withExecutionMetadata(
                            compiledSupportsBackward ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD,
                            compiledSupportsBackward,
                            forwardEndIndex
                    )
                    .withPartitions(compiledPartitions, planByPartitionId())
                    .withOptimizedRegions(optimizedRegions);
            compiledOptimizerState = loweringReady.withMemoryPlan(MemoryPlanner.plan(
                    loweringReady,
                    MemoryPlannerPolicy.fromConfig(memoryConfig)
            ));
            compiledMemoryPlan = compiledOptimizerState.memoryPlan();
        }

        private Map<String, PartitionPlan> planByPartitionId() {
            java.util.HashMap<String, PartitionPlan> out = new java.util.HashMap<>();
            for (BackendCandidatePartition candidate : compiledBackendSelectionCandidates) {
                if (candidate == null || candidate.partition() == null || candidate.plan() == null) {
                    continue;
                }
                out.put(candidate.partition().partitionId(), candidate.plan());
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
