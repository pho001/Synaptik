package graph;

import backend.CPUBackend;
import backend.ComputeBackend;
import backend.kernels.cpu.CpuExecutionPlanner;
import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.kernels.cpu.CpuNodeWorkspace;
import backend.kernels.cpu.fused.CompiledFusedKernel;
import backend.registry.CpuKernelRegistry;
import backend.runtime.ExecutionMode;
import graph.codegen.CompiledFusedKernelFactory;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.PreparedExecution;
import graph.execution.PreparedNodeExecution;
import graph.execution.trace.CompileTrace;
import graph.execution.trace.RunTrace;
import graph.optimizer.GraphOptimizer;
import operations.FusedOperation;
import operations.Operation;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;

public class CompiledGraph {
    private static final CompiledFusedKernelFactory FUSED_KERNEL_FACTORY = new CompiledFusedKernelFactory();
    private final Tensor rootTensor;
    private final GraphOptimizer optimizer;
    private CompileTrace compileTrace = CompileTrace.skipped();
    private final List<Tensor> finalGraph = new ArrayList<>();
    private final List<Tensor> forwardGraph = new ArrayList<>();
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
        finalGraph.clear();
        forwardGraph.clear();

        forwardOutput = rootTensor.forwardOutput();
        forwardGraph.addAll(forwardOutput.topologicalSort());

        if (!hasTrainableLeafInputs()) {
            finalGraph.addAll(optimizer.optimize(new ArrayList<>(forwardGraph)));
            forwardEndIndex = finalGraph.indexOf(forwardOutput);
            if (forwardEndIndex == -1) {
                throw new IllegalStateException("Forward output node not found in inference finalGraph.");
            }
            return;
        }

        if (rootTensor.getDataType() == tensor.DataType.BOOL || rootTensor.getDataType() == tensor.DataType.INT32) {
            throw new UnsupportedOperationException("BOOL/INT32 root tensors do not support backward execution.");
        }

        rootTensor.setGradient(Tensor.onesLike(rootTensor));
        for (int i = forwardGraph.size() - 1; i >= 0; i--) {
            forwardGraph.get(i).buildBackwardGraph();
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
        Tensor superRoot = new Tensor(new int[]{1}, targetsToSave, new operations.noop(), "System_Super_Root");

        finalGraph.addAll(superRoot.topologicalSort());
        finalGraph.remove(superRoot);

        List<Tensor> optimized = optimizer.optimize(new ArrayList<>(finalGraph));
        finalGraph.clear();
        finalGraph.addAll(optimized);
        forwardEndIndex = finalGraph.indexOf(forwardOutput);
        if (forwardEndIndex == -1) {
            throw new IllegalStateException("Forward output node not found in finalGraph.");
        }
    }

    public boolean supportsBackward() {
        return hasTrainableLeafInputs();
    }

    public PreparedExecution prepare(config.runtime.RuntimeConfig runtimeConfig) {
        config.runtime.RuntimeConfig effectiveConfig = runtimeConfig == null
                ? (supportsBackward() ? config.runtime.RuntimeConfig.trainingDefaults() : config.runtime.RuntimeConfig.inferenceDefaults())
                : runtimeConfig;
        long t0 = System.nanoTime();
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(effectiveConfig.cpuKernelConfig());
        backend.runtime.RuntimeConfig backendRuntimeConfig = effectiveConfig.toBackendRuntimeConfig();

        List<PreparedNodeExecution> forwardSteps = new ArrayList<>();
        List<PreparedNodeExecution> backwardSteps = new ArrayList<>();
        Map<Tensor, CompiledNodeExecutionMetadata> preparedMetadata = new HashMap<>();
        for (int i = 0; i < finalGraph.size(); i++) {
            Tensor tensor = finalGraph.get(i);
            if (tensor.getOperation() == null || tensor.getPrevTensors() == null) {
                continue;
            }
            PreparedNodeExecution step = new PreparedNodeExecution(
                    tensor,
                    prepareMetadata(tensor, planner, backendRuntimeConfig, preparedMetadata)
            );
            preparedMetadata.put(tensor, step.metadata());
            if (i <= forwardEndIndex) {
                forwardSteps.add(step);
            } else {
                backwardSteps.add(step);
            }
        }

        return new PreparedExecution(
                effectiveConfig,
                supportsBackward(),
                forwardSteps,
                backwardSteps,
                finalGraph,
                rootTensor,
                forwardOutput,
                new graph.execution.trace.PrepareTrace(
                        true,
                        System.nanoTime() - t0,
                        forwardSteps.size(),
                        backwardSteps.size()
                )
        );
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

    public tuning.report.BenchmarkReport benchmark(tuning.session.BenchmarkRequest request) {
        return tuning.session.BenchmarkSession.create(request).run();
    }

    public tuning.report.BenchmarkSuiteReport benchmark(tuning.session.BenchmarkSuiteRequest request) {
        return tuning.session.BenchmarkSuiteSession.create(request).run();
    }

    public tuning.session.TuningResult autotune(tuning.session.AutotuneRequest request) {
        return tuning.session.AutotuneSession.create(request).run();
    }

    public void zeroGrad() {
        for (Tensor tensor : finalGraph) {
            if (tensor.getGradient() == null) {
                continue;
            }
            switch (tensor.getGradient().getDataType()) {
                case FLOAT64 -> java.util.Arrays.fill(tensor.getGradient().getFloat64Data(), 0.0d);
                case FLOAT32 -> java.util.Arrays.fill(tensor.getGradient().getFloat32Data(), 0.0f);
                case FLOAT16 -> java.util.Arrays.fill(tensor.getGradient().getFloat16Data(), (short) 0);
                case INT32 -> java.util.Arrays.fill(tensor.getGradient().getInt32Data(), 0);
                case BOOL -> java.util.Arrays.fill(tensor.getGradient().getBoolData(), (byte) 0);
            }
        }
    }

    public Tensor getRootTensor() {
        return rootTensor;
    }

    public List<Tensor> getCompiledGraphAsList() {
        return finalGraph;
    }

    public CompileTrace compileTrace() {
        return compileTrace;
    }

    private CompiledNodeExecutionMetadata prepareMetadata(
            Tensor tensor,
            CpuExecutionPlanner planner,
            backend.runtime.RuntimeConfig runtimeConfig,
            Map<Tensor, CompiledNodeExecutionMetadata> preparedMetadata
    ) {
        ComputeBackend backend = tensor.resolveBackend();
        if (backend != ComputeBackend.CPU) {
            return new CompiledNodeExecutionMetadata(backend, null, null, null, null);
        }

        Operation operation = tensor.getOperation();
        CpuKernel kernel = CpuKernelRegistry.resolve(operation.opType());
        if (kernel == null) {
            throw new IllegalStateException("Missing CPU kernel for opType=" + operation.opType());
        }
        CpuNodeExecutionPlan cpuPlan = CPUBackend.buildExecutionPlan(
                operation,
                tensor.getPrevTensors(),
                tensor,
                planner,
                runtimeConfig.blasConfig()
        );
        CompiledFusedKernel fusedKernel = null;
        if (operation.opType() == Operation.OpType.FUSED) {
            fusedKernel = FUSED_KERNEL_FACTORY.create((FusedOperation) operation);
        }
        CpuNodeWorkspace cpuWorkspace = resolveCpuWorkspace(tensor, operation, preparedMetadata);
        return new CompiledNodeExecutionMetadata(backend, kernel, cpuPlan, fusedKernel, cpuWorkspace);
    }

    private CpuNodeWorkspace resolveCpuWorkspace(
            Tensor tensor,
            Operation operation,
            Map<Tensor, CompiledNodeExecutionMetadata> preparedMetadata
    ) {
        return switch (operation.opType()) {
            case MAX_POOL2D -> CpuNodeWorkspace.withIntWorkspace(tensor.getFlatDataSize());
            case MAX_POOL2D_BACKWARD_INPUT -> resolveSharedMaxPoolWorkspace(tensor, preparedMetadata);
            default -> null;
        };
    }

    private CpuNodeWorkspace resolveSharedMaxPoolWorkspace(
            Tensor tensor,
            Map<Tensor, CompiledNodeExecutionMetadata> preparedMetadata
    ) {
        for (Tensor input : tensor.getPrevTensors()) {
            if (input == null || input.getOperation() == null) {
                continue;
            }
            if (input.getOperation().opType() != Operation.OpType.MAX_POOL2D) {
                continue;
            }
            CompiledNodeExecutionMetadata metadata = preparedMetadata.get(input);
            if (metadata == null || metadata.cpuWorkspace() == null) {
                throw new IllegalStateException("Missing prepared maxPool2d workspace for backward node " + tensor.getLabel());
            }
            return metadata.cpuWorkspace();
        }
        throw new IllegalStateException("maxPool2d backward node is missing its forward maxPool2d dependency.");
    }

    private boolean hasTrainableLeafInputs() {
        for (Tensor tensor : forwardGraph) {
            if (tensor.getOperation() == null && tensor.getRequiresGrad()) {
                return true;
            }
        }
        return false;
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
            tensor.setBackward(true);
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
