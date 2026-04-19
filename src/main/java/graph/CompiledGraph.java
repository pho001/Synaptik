package graph;

import backend.runtime.ExecutionMode;
import graph.execution.PreparedExecution;
import graph.execution.trace.CompileTrace;
import graph.execution.trace.RunTrace;
import graph.optimizer.GraphOptimizer;
import tensor.AutogradCompilationScope;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.IdentityHashMap;
import java.util.Set;

public class CompiledGraph {
    private final Tensor rootTensor;
    private final GraphOptimizer optimizer;
    private CompileTrace compileTrace = CompileTrace.skipped();
    private final List<Tensor> finalGraph = new ArrayList<>();
    private final List<Tensor> forwardGraph = new ArrayList<>();
    private List<CompiledNode> compiledNodes = List.of();
    private Map<Tensor, CompiledNode> compiledNodeByTensor = Map.of();
    private Map<Tensor, CompiledGradientBinding> compiledGradients = Map.of();
    private CompiledNode compiledForwardOutput;
    private Tensor forwardOutput;
    private int forwardEndIndex = -1;

    public CompiledGraph(Tensor rootTensor, GraphOptimizer forwardOptimizer) {
        this.rootTensor = rootTensor;
        this.optimizer = forwardOptimizer;
        long t0 = System.nanoTime();
        compile();
        this.compileTrace = new CompileTrace(
                true,
                System.nanoTime() - t0,
                finalGraph.size(),
                forwardGraph.size(),
                supportsBackward()
        );
    }

    public static CompiledGraph compile(Tensor rootTensor, config.optimizer.OptimizerConfig optimizerConfig) {
        if (rootTensor == null) {
            throw new IllegalArgumentException("rootTensor cannot be null");
        }
        if (optimizerConfig == null) {
            throw new IllegalArgumentException("optimizerConfig cannot be null");
        }
        return new CompiledGraph(rootTensor, graph.optimizer.OptimizerFactory.create(optimizerConfig));
    }

    public static CompiledGraph compile(Tensor rootTensor, GraphOptimizer optimizer) {
        if (rootTensor == null) {
            throw new IllegalArgumentException("rootTensor cannot be null");
        }
        if (optimizer == null) {
            throw new IllegalArgumentException("optimizer cannot be null");
        }
        return new CompiledGraph(rootTensor, optimizer);
    }

    public void compile() {
        compiledNodes = List.of();
        compiledNodeByTensor = Map.of();
        compiledForwardOutput = null;
        compiledGradients = Map.of();
        finalGraph.clear();
        forwardGraph.clear();

        forwardOutput = rootTensor.forwardOutput();
        forwardGraph.addAll(forwardOutput.topologicalSort());
        try (AutogradCompilationScope ignored = AutogradCompilationScope.open()) {
            resetAutogradBuildState();

            if (!hasTrainableLeafInputs()) {
                finalGraph.addAll(optimizer.optimize(new ArrayList<>(forwardGraph)));
                forwardEndIndex = finalGraph.indexOf(forwardOutput);
                if (forwardEndIndex == -1) {
                    throw new IllegalStateException("Forward output node not found in inference finalGraph.");
                }
                rebuildCompiledNodeSnapshot();
                return;
            }

            if (rootTensor.getDataType() == tensor.DataType.BOOL || rootTensor.getDataType() == tensor.DataType.INT32) {
                throw new UnsupportedOperationException("BOOL/INT32 root tensors do not support backward execution.");
            }

            TensorInternalAccess.setGradient(rootTensor, Tensor.onesLike(rootTensor));
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

            List<Tensor> optimized = optimizer.optimize(new ArrayList<>(finalGraph));
            finalGraph.clear();
            finalGraph.addAll(optimized);
            forwardEndIndex = finalGraph.indexOf(forwardOutput);
            if (forwardEndIndex == -1) {
                throw new IllegalStateException("Forward output node not found in finalGraph.");
            }
            rebuildCompiledNodeSnapshot();
            compiledGradients = captureCompiledGradients(finalGraph);
        }
    }

    public boolean supportsBackward() {
        return hasTrainableLeafInputs();
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
            Tensor gradient = node.semanticTensor().getGradient();
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

    private boolean hasTrainableLeafInputs() {
        for (Tensor tensor : forwardGraph) {
            if (tensor.getOperation() == null && tensor.getRequiresGrad()) {
                return true;
            }
        }
        return false;
    }

    private void resetAutogradBuildState() {
        for (Tensor tensor : forwardGraph) {
            TensorInternalAccess.clearGradient(tensor);
            TensorInternalAccess.setBackward(tensor, false);
        }
        TensorInternalAccess.clearGradient(rootTensor);
    }

    private Map<Tensor, CompiledGradientBinding> captureCompiledGradients(List<Tensor> graph) {
        IdentityHashMap<Tensor, CompiledGradientBinding> out = new IdentityHashMap<>();
        for (Tensor tensor : graph) {
            Tensor gradient = tensor.getGradient();
            if (gradient == null) {
                continue;
            }
            CompiledNode gradientNode = compiledNodeByTensor.get(gradient);
            if (gradientNode != null) {
                out.put(tensor, CompiledGradientBinding.node(gradientNode.id()));
                continue;
            }
            if (gradient.getOperation() == null) {
                out.put(tensor, CompiledGradientBinding.constant(gradient));
                continue;
            }
            throw new IllegalStateException("Gradient binding for tensor '" + tensor.getLabel()
                    + "' does not resolve to a compiled node or constant.");
        }
        if (out.isEmpty()) {
            return Map.of();
        }
        return Map.copyOf(out);
    }

    private void rebuildCompiledNodeSnapshot() {
        compiledNodes = CompiledNode.snapshot(finalGraph);
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
