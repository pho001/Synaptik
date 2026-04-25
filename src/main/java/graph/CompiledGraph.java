package graph;

import backend.runtime.ExecutionMode;
import graph.execution.PreparedExecution;
import graph.execution.trace.AcceleratorPartitionCompileTrace;
import graph.execution.trace.CompileTrace;
import graph.execution.trace.RunTrace;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.partition.AcceleratorCandidatePartition;
import graph.optimizer.partition.AcceleratorPartitionPlan;
import graph.optimizer.partition.AcceleratorPartitionPlanner;
import graph.optimizer.partition.AcceleratorRegionAdapterRegistry;
import graph.optimizer.partition.AcceleratorTarget;
import graph.optimizer.partition.GreedyMaxRegionPartitionPlanner;
import graph.optimizer.partition.PartitionPlanningRequest;
import graph.optimizer.partition.PartitionPlanningResult;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.ScoredCandidatePartitionPlanner;
import tensor.AutogradCompilationScope;
import tensor.CompileMode;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.IdentityHashMap;
import java.util.Set;

public class CompiledGraph {
    private final Tensor rootTensor;
    private final SemanticForwardCanonicalizer forwardCanonicalizer;
    private final GraphOptimizer optimizer;
    private final config.optimizer.PartitionConfig partitionConfig;
    private final CompileMode compileMode;
    private CompileTrace compileTrace = CompileTrace.skipped();
    private final List<Tensor> finalGraph = new ArrayList<>();
    private final List<Tensor> forwardGraph = new ArrayList<>();
    private List<CompiledNode> compiledNodes = List.of();
    private Map<Tensor, CompiledNode> compiledNodeByTensor = Map.of();
    private Map<Tensor, CompiledGradientBinding> compiledGradients = Map.of();
    private CompiledGradientBinding forwardSeedGradient;
    private CompiledNode compiledForwardOutput;
    private List<AcceleratorPartitionPlan> compiledAcceleratorPlans = List.of();
    private List<AcceleratorCandidatePartition> compiledAcceleratorCandidates = List.of();
    private AcceleratorPartitionCompileTrace compiledAcceleratorPartitionTrace = AcceleratorPartitionCompileTrace.empty();
    private Tensor forwardOutput;
    private int forwardEndIndex = -1;
    private boolean compiledSupportsBackward;

    private CompiledGraph(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer forwardOptimizer,
            config.optimizer.PartitionConfig partitionConfig,
            CompileMode compileMode
    ) {
        this.rootTensor = rootTensor;
        this.forwardCanonicalizer = forwardCanonicalizer;
        this.optimizer = forwardOptimizer;
        this.partitionConfig = partitionConfig == null ? config.optimizer.PartitionConfig.defaults() : partitionConfig;
        this.compileMode = compileMode == null ? CompileMode.AUTO : compileMode;
        long t0 = System.nanoTime();
        compile();
        this.compileTrace = new CompileTrace(
                true,
                System.nanoTime() - t0,
                finalGraph.size(),
                forwardGraph.size(),
                supportsBackward(),
                compiledAcceleratorPartitionTrace
        );
    }

    public static CompiledGraph compile(Tensor rootTensor, config.optimizer.OptimizerConfig optimizerConfig) {
        return compile(rootTensor, optimizerConfig, CompileMode.AUTO);
    }

    public static CompiledGraph compile(Tensor rootTensor, config.optimizer.OptimizerConfig optimizerConfig, CompileMode compileMode) {
        if (rootTensor == null) {
            throw new IllegalArgumentException("rootTensor cannot be null");
        }
        if (optimizerConfig == null) {
            throw new IllegalArgumentException("optimizerConfig cannot be null");
        }
        return new CompiledGraph(
                rootTensor,
                graph.optimizer.OptimizerFactory.createSemanticForwardCanonicalizer(optimizerConfig),
                graph.optimizer.OptimizerFactory.create(optimizerConfig),
                optimizerConfig.partition(),
                compileMode
        );
    }

    public static CompiledGraph compile(Tensor rootTensor, GraphOptimizer optimizer) {
        return compile(rootTensor, optimizer, CompileMode.AUTO);
    }

    public static CompiledGraph compile(Tensor rootTensor, GraphOptimizer optimizer, CompileMode compileMode) {
        if (rootTensor == null) {
            throw new IllegalArgumentException("rootTensor cannot be null");
        }
        if (optimizer == null) {
            throw new IllegalArgumentException("optimizer cannot be null");
        }
        return new CompiledGraph(rootTensor, null, optimizer, config.optimizer.PartitionConfig.defaults(), compileMode);
    }

    public void compile() {
        compiledNodes = List.of();
        compiledNodeByTensor = Map.of();
        compiledForwardOutput = null;
        compiledGradients = Map.of();
        forwardSeedGradient = null;
        compiledAcceleratorPlans = List.of();
        compiledAcceleratorCandidates = List.of();
        compiledAcceleratorPartitionTrace = AcceleratorPartitionCompileTrace.empty();
        compiledSupportsBackward = false;
        finalGraph.clear();
        forwardGraph.clear();

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
                sourceTensors.put(requireForwardRoot(), rootTensor);
                rebuildCompiledNodeSnapshot(sourceTensors);
                captureForwardSeedGradient();
                return;
            }

            if (rootTensor.getDataType() == tensor.DataType.BOOL || rootTensor.getDataType() == tensor.DataType.INT32) {
                throw new UnsupportedOperationException("BOOL/INT32 root tensors do not support backward execution.");
            }

            Tensor actualForwardRoot = requireForwardRoot();
            TensorInternalAccess.setGradient(actualForwardRoot, Tensor.onesLike(actualForwardRoot));
            for (int i = forwardGraph.size() - 1; i >= 0; i--) {
                TensorInternalAccess.buildBackwardGraph(forwardGraph.get(i));
            }

            List<Tensor> backwardTargets = collectBackwardTargets();
            if (backwardTargets.isEmpty()) {
                for (Tensor tensor : forwardGraph) {
                    if (tensor.getGradient() != null) {
                        backwardTargets.add(tensor.getGradient());
                    }
                }
            }

            collectBackwardNodes();

            List<Tensor> targetsToSave = new ArrayList<>();
            targetsToSave.add(forwardOutput);
            targetsToSave.addAll(backwardTargets);
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
            sourceTensors.put(requireForwardRoot(), rootTensor);
            rebuildCompiledNodeSnapshot(sourceTensors);
            compiledGradients = captureCompiledGradients(finalGraph, sourceTensors);
            captureForwardSeedGradient();
        }
    }

    public boolean supportsBackward() {
        return compiledSupportsBackward;
    }

    public CompileMode compileMode() {
        return compileMode;
    }

    public PreparedExecution prepare() {
        return prepare((config.runtime.RuntimeConfig) null);
    }

    public PreparedExecution prepare(config.runtime.RuntimeConfig runtimeConfig) {
        config.runtime.RuntimeConfig effectiveConfig = runtimeConfig == null
                ? (supportsBackward() ? config.runtime.RuntimeConfig.trainingDefaults() : config.runtime.RuntimeConfig.inferenceDefaults())
                : runtimeConfig;
        return PreparedExecutionBuilder.prepare(this, effectiveConfig);
    }

    public PreparedExecution prepare(config.profile.ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        return prepare(profile.runtime());
    }

    public void execute(config.runtime.RuntimeConfig runtimeConfig, ExecutionMode mode) {
        prepare(runtimeConfig).execute(mode);
    }

    public void execute(config.profile.ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        prepare(profile.runtime()).execute(profile.mode());
    }

    public RunTrace executeTraced(config.runtime.RuntimeConfig runtimeConfig, ExecutionMode mode) {
        return prepare(runtimeConfig).executeTraced(mode);
    }

    public RunTrace executeTraced(config.profile.ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        return prepare(profile.runtime()).executeTraced(profile.mode());
    }

    public void executePrepared(PreparedExecution execution, ExecutionMode mode) {
        execution.execute(mode);
    }

    public void zeroGrad() {
        for (CompiledNode node : compiledNodes) {
            if (node.backwardNode()) {
                continue;
            }
            Tensor gradient = node.sourceTensor().getGradient();
            if (gradient == null) {
                continue;
            }
            switch (gradient.getDataType()) {
                case FLOAT64 -> java.util.Arrays.fill(gradient.getFloat64Data(), 0.0d);
                case FLOAT32 -> java.util.Arrays.fill(gradient.getFloat32Data(), 0.0f);
                case BFLOAT16 -> java.util.Arrays.fill(gradient.getBFloat16Data(), (short) 0);
                case INT32 -> java.util.Arrays.fill(gradient.getInt32Data(), 0);
                case BOOL -> java.util.Arrays.fill(gradient.getBoolData(), (byte) 0);
            }
        }
    }

    public Tensor getRootTensor() {
        return rootTensor;
    }

    public List<Tensor> getCompiledGraphAsList() {
        return List.copyOf(finalGraph);
    }

    public CompileTrace compileTrace() {
        return compileTrace;
    }

    Map<Tensor, CompiledGradientBinding> compiledGradients() {
        return compiledGradients;
    }

    CompiledGradientBinding forwardSeedGradient() {
        return forwardSeedGradient;
    }

    List<CompiledNode> compiledNodesView() {
        return compiledNodes;
    }

    int forwardBoundaryNodeId() {
        return forwardEndIndex;
    }

    CompiledNode compiledForwardOutputNode() {
        return compiledForwardOutput;
    }

    Map<Tensor, CompiledGradientBinding> compiledGradientBindings() {
        return compiledGradients;
    }

    List<AcceleratorPartitionPlan> compiledAcceleratorPlansView() {
        return compiledAcceleratorPlans;
    }

    List<AcceleratorCandidatePartition> compiledAcceleratorCandidatesView() {
        return compiledAcceleratorCandidates;
    }

    AcceleratorPartitionCompileTrace compiledAcceleratorPartitionTrace() {
        return compiledAcceleratorPartitionTrace;
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
        List<Tensor> optimized = optimizer.optimize(new ArrayList<>(snapshot.graph()));
        IdentityHashMap<Tensor, Tensor> composed = new IdentityHashMap<>();
        for (Map.Entry<Tensor, Tensor> entry : snapshot.originalBySnapshot().entrySet()) {
            Tensor original = entry.getValue();
            composed.put(entry.getKey(), sourceTensors.getOrDefault(original, original));
        }
        sourceTensors.clear();
        sourceTensors.putAll(composed);
        forwardOutput = snapshot.forwardOutput();
        return optimized;
    }

    private Map<Tensor, CompiledGradientBinding> captureCompiledGradients(List<Tensor> graph, Map<Tensor, Tensor> sourceTensors) {
        IdentityHashMap<Tensor, CompiledGradientBinding> out = new IdentityHashMap<>();
        for (Tensor tensor : graph) {
            Tensor gradient = tensor.getGradient();
            if (gradient == null) {
                continue;
            }
            Tensor publishedTensor = sourceTensors.getOrDefault(tensor, tensor);
            CompiledNode gradientNode = compiledNodeByTensor.get(gradient);
            if (gradientNode != null) {
                out.put(publishedTensor, CompiledGradientBinding.node(gradientNode.id()));
                continue;
            }
            if (gradient.getOperation() == null) {
                out.put(publishedTensor, CompiledGradientBinding.constant(gradient));
                continue;
            }
            throw new IllegalStateException("Gradient binding for tensor '" + publishedTensor.getLabel()
                    + "' does not resolve to a compiled node or constant.");
        }
        if (out.isEmpty()) {
            return Map.of();
        }
        return Map.copyOf(out);
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
        rebuildAcceleratorPartitionSnapshot();
    }

    private void rebuildAcceleratorPartitionSnapshot() {
        if (compiledNodes.isEmpty()) {
            compiledAcceleratorPlans = List.of();
            compiledAcceleratorCandidates = List.of();
            compiledAcceleratorPartitionTrace = AcceleratorPartitionCompileTrace.empty();
            return;
        }
        AcceleratorTarget target = resolveAcceleratorTarget();
        if (target.isNone()) {
            compiledAcceleratorPlans = List.of();
            compiledAcceleratorCandidates = List.of();
            compiledAcceleratorPartitionTrace = AcceleratorPartitionCompileTrace.empty();
            return;
        }
        backend.prepare.BackendPrepareContext prepareContext = new backend.prepare.BackendPrepareContext(
                supportsBackward() ? config.runtime.RuntimeConfig.trainingDefaults() : config.runtime.RuntimeConfig.inferenceDefaults(),
                compiledSupportsBackward,
                compiledNodes,
                buildConsumerMap(compiledNodes)
        );
        PartitionPlanningResult planning = selectPlanner(partitionConfig.plannerStrategy()).plan(
                new PartitionPlanningRequest(
                        partitionConfig.plannerStrategy(),
                        target,
                        prepareContext,
                        graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.PlannerPolicy.fromConfig(partitionConfig),
                        AcceleratorRegionAdapterRegistry.forTarget(target)
                )
        );
        compiledAcceleratorPlans = planning.plans();
        compiledAcceleratorCandidates = planning.plans().stream()
                .map(plan -> new AcceleratorCandidatePartition(
                        plan.anchorNodeId(),
                        plan.nodeIds(),
                        java.util.Set.of(plan.backend()),
                        plan
                ))
                .toList();
        compiledAcceleratorPartitionTrace = planning.trace();
    }

    private AcceleratorTarget resolveAcceleratorTarget() {
        AcceleratorTarget configured = partitionConfig.target();
        if (configured != null && !configured.isAuto()) {
            return configured;
        }
        for (CompiledNode node : compiledNodes) {
            AcceleratorTarget target = AcceleratorTarget.fromBackend(node.backend());
            if (!target.isNone()) {
                return target;
            }
        }
        return AcceleratorTarget.NONE;
    }

    private AcceleratorPartitionPlanner selectPlanner(PartitionPlannerStrategy strategy) {
        PartitionPlannerStrategy resolved = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        return switch (resolved) {
            case GREEDY_MAX_REGION -> new GreedyMaxRegionPartitionPlanner();
            case SCORED_CANDIDATE_SEARCH -> new ScoredCandidatePartitionPlanner();
        };
    }

    private static Map<Integer, List<CompiledNode>> buildConsumerMap(List<CompiledNode> graph) {
        Map<Integer, List<CompiledNode>> consumers = new HashMap<>();
        for (CompiledNode node : graph) {
            consumers.computeIfAbsent(node.id(), ignored -> new ArrayList<>());
        }
        for (CompiledNode node : graph) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new ArrayList<>()).add(node);
            }
        }
        return consumers;
    }

    private void captureForwardSeedGradient() {
        Tensor gradient = requireForwardRoot().getGradient();
        if (gradient == null) {
            forwardSeedGradient = null;
            return;
        }
        CompiledNode gradientNode = compiledNodeByTensor.get(gradient);
        if (gradientNode != null) {
            forwardSeedGradient = CompiledGradientBinding.node(gradientNode.id());
            return;
        }
        if (gradient.getOperation() == null) {
            forwardSeedGradient = CompiledGradientBinding.constant(gradient);
            return;
        }
        throw new IllegalStateException("Forward seed gradient does not resolve to a compiled node or constant.");
    }

    private Tensor requireForwardRoot() {
        List<Tensor> inputs = forwardOutput == null ? null : forwardOutput.getPrevTensors();
        if (inputs == null || inputs.size() != 1 || inputs.get(0) == null) {
            throw new IllegalStateException("System forward output must have exactly one input.");
        }
        return inputs.get(0);
    }

    private List<Tensor> collectBackwardNodes() {
        List<Tensor> backwardNodes = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        Set<Tensor> forwardSet = new HashSet<>(forwardGraph);

        for (int i = forwardGraph.size() - 1; i >= 0; i--) {
            Tensor gradTensor = forwardGraph.get(i).getGradient();
            if (gradTensor != null) {
                collectDFS(gradTensor, visited, backwardNodes, forwardSet);
            }
        }

        return backwardNodes;
    }

    private void collectDFS(Tensor tensor, Set<Tensor> visited, List<Tensor> sortedList, Set<Tensor> forwardSet) {
        if (tensor == null || visited.contains(tensor)) {
            return;
        }

        visited.add(tensor);
        if (tensor.getPrevTensors() != null) {
            for (Tensor parent : tensor.getPrevTensors()) {
                collectDFS(parent, visited, sortedList, forwardSet);
            }
        }

        if (tensor.getOperation() != null && !forwardSet.contains(tensor)) {
            TensorInternalAccess.setBackward(tensor, true);
            sortedList.add(tensor);
        }
    }

    private List<Tensor> collectBackwardTargets() {
        List<Tensor> targets = new ArrayList<>();
        Set<Tensor> unique = new LinkedHashSet<>();
        for (Tensor tensor : forwardGraph) {
            if (tensor.getOperation() == null && tensor.getRequiresGrad() && tensor.getGradient() != null) {
                unique.add(tensor.getGradient());
            }
        }
        targets.addAll(unique);
        return targets;
    }
}
