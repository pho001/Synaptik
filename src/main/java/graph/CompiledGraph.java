package graph;

import backend.CPUBackend;
import backend.ComputeBackend;
import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.kernels.cpu.CpuNodeWorkspace;
import backend.kernels.cpu.ResolvedCpuComputeContract;
import backend.kernels.cpu.fused.plan.PreparedFusedDispatch;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.plan.CpuExecutionPlanner;
import backend.registry.CpuKernelResolver;
import backend.runtime.ExecutionMode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.PreparedExecution;
import graph.execution.PreparedNodeExecution;
import graph.execution.trace.CompileTrace;
import graph.fused.FusedExecutionBackendResolver;
import graph.fused.FusedExecutionPlan;
import graph.fused.PreparedFusedExecutable;
import graph.execution.trace.RunTrace;
import graph.optimizer.GraphOptimizer;
import operations.fused.FusedOperation;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;

public class CompiledGraph {
    private static final FusedExecutionBackendResolver FUSED_BACKEND_RESOLVER = new FusedExecutionBackendResolver();
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
        resetAutogradBuildState();

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

        List<PreparedNodeExecution> forwardSteps = new ArrayList<>();
        List<PreparedNodeExecution> backwardSteps = new ArrayList<>();
        Map<Tensor, CompiledNodeExecutionMetadata> preparedMetadata = new HashMap<>();
        Map<Tensor, List<Tensor>> consumers = buildConsumerMap(finalGraph);
        for (int i = 0; i < finalGraph.size(); i++) {
            Tensor tensor = finalGraph.get(i);
            if (tensor.getOperation() == null || tensor.getPrevTensors() == null) {
                continue;
            }
            PreparedNodeExecution step = new PreparedNodeExecution(
                    tensor,
                    prepareMetadata(tensor, planner, effectiveConfig, preparedMetadata, consumers)
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

    public void zeroGrad() {
        for (Tensor tensor : finalGraph) {
            if (tensor.getGradient() == null) {
                continue;
            }
            switch (tensor.getGradient().getDataType()) {
                case FLOAT64 -> java.util.Arrays.fill(tensor.getGradient().getFloat64Data(), 0.0d);
                case FLOAT32 -> java.util.Arrays.fill(tensor.getGradient().getFloat32Data(), 0.0f);
                case BFLOAT16 -> java.util.Arrays.fill(tensor.getGradient().getBFloat16Data(), (short) 0);
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
            config.runtime.RuntimeConfig runtimeConfig,
            Map<Tensor, CompiledNodeExecutionMetadata> preparedMetadata,
            Map<Tensor, List<Tensor>> consumers
    ) {
        ComputeBackend backend = tensor.resolveBackend();
        if (backend != ComputeBackend.CPU) {
            return new CompiledNodeExecutionMetadata(backend, null, null, null, null);
        }

        Operation operation = tensor.getOperation();
        CpuKernel kernel = CpuKernelResolver.resolve(operation.opType());
        if (kernel == null) {
            throw new IllegalStateException("Missing CPU kernel for opType=" + operation.opType());
        }
        ResolvedDispatchHints dispatchHintsOverride = null;
        PreparedFusedDispatch preparedFusedDispatch = null;
        if (operation.opType().category() == Operation.OpArityClass.ELEMENT_WISE) {
            ResolvedCpuComputeContract elementwiseContract = planner.resolveComputeContract(
                    operation,
                    tensor.getPrevTensors(),
                    tensor,
                    runtimeConfig.blas(),
                    null,
                    null
            );
            dispatchHintsOverride = planner.resolveDispatchHints(operation, tensor, elementwiseContract);
        }
        if (operation.opType() == Operation.OpType.FUSED) {
            ResolvedCpuComputeContract fusedContract = planner.resolveComputeContract(
                    operation,
                    tensor.getPrevTensors(),
                    tensor,
                    runtimeConfig.blas(),
                    null,
                    null
            );
            preparedFusedDispatch = planner.resolveFusedDispatch((FusedOperation) operation, tensor, fusedContract);
            dispatchHintsOverride = preparedFusedDispatch.dispatchHints();
        }
        CpuNodeExecutionPlan cpuPlan = CPUBackend.buildExecutionPlan(
                operation,
                tensor.getPrevTensors(),
                tensor,
                planner,
                runtimeConfig.blas(),
                runtimeConfig.conv2d(),
                shouldPublishFloatContinuation(tensor, operation, consumers),
                dispatchHintsOverride
        );
        PreparedFusedExecutable fusedExecutable = null;
        if (operation.opType() == Operation.OpType.FUSED && cpuPlan != null) {
            fusedExecutable = FUSED_BACKEND_RESOLVER.resolve(
                    new FusedExecutionPlan(
                            (FusedOperation) operation,
                            cpuPlan.computeContract(),
                            tensor.getFlatDataSize(),
                            preparedFusedDispatch == null ? 1 : preparedFusedDispatch.cpuVectorMinSize(),
                            preparedFusedDispatch == null ? 1 : preparedFusedDispatch.asmVectorWidth()
                    ),
                    runtimeConfig.fused()
            );
        }
        CpuNodeWorkspace cpuWorkspace = resolveCpuWorkspace(tensor, operation, cpuPlan, preparedMetadata);
        return new CompiledNodeExecutionMetadata(backend, kernel, cpuPlan, fusedExecutable, cpuWorkspace);
    }

    private CpuNodeWorkspace resolveCpuWorkspace(
            Tensor tensor,
            Operation operation,
            CpuNodeExecutionPlan cpuPlan,
            Map<Tensor, CompiledNodeExecutionMetadata> preparedMetadata
    ) {
        return switch (operation.opType()) {
            case MAX_POOL2D -> CpuNodeWorkspace.withIntWorkspace(tensor.getFlatDataSize());
            case MAX_POOL2D_BACKWARD_INPUT -> resolveSharedMaxPoolWorkspace(tensor, preparedMetadata);
            case MATMUL -> needsBFloat16BlasWorkspace(tensor, cpuPlan)
                    ? CpuNodeWorkspace.withFloatWorkspace(tensor.getFlatDataSize())
                    : null;
            case LINEAR -> needsBFloat16BlasWorkspace(tensor, cpuPlan)
                    ? CpuNodeWorkspace.withFloatWorkspaceAndPackedLinearWeights(tensor.getFlatDataSize())
                    : CpuNodeWorkspace.withPackedLinearWeights();
            case LOG_SOFTMAX -> shouldPublishFloatContinuation(tensor, operation, buildConsumerMap(finalGraph))
                    ? CpuNodeWorkspace.withFloatWorkspace(tensor.getFlatDataSize())
                    : null;
            case CONV2D_GEMM -> tensor.getDataType() == DataType.BFLOAT16
                    ? CpuNodeWorkspace.withFloatWorkspace(tensor.getFlatDataSize())
                    : null;
            default -> null;
        };
    }

    private boolean needsBFloat16BlasWorkspace(Tensor tensor, CpuNodeExecutionPlan cpuPlan) {
        return tensor.getDataType() == DataType.BFLOAT16
                && cpuPlan != null
                && cpuPlan.matMulHints() != null
                && (cpuPlan.matMulHints().useBlas() || cpuPlan.matMulHints().useBatchedBlas());
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

    private Map<Tensor, List<Tensor>> buildConsumerMap(List<Tensor> graph) {
        Map<Tensor, List<Tensor>> consumers = new HashMap<>();
        for (Tensor tensor : graph) {
            consumers.computeIfAbsent(tensor, ignored -> new ArrayList<>());
        }
        for (Tensor tensor : graph) {
            if (tensor.getPrevTensors() == null) {
                continue;
            }
            for (Tensor input : tensor.getPrevTensors()) {
                if (input == null) {
                    continue;
                }
                consumers.computeIfAbsent(input, ignored -> new ArrayList<>()).add(tensor);
            }
        }
        return consumers;
    }

    private boolean shouldPublishFloatContinuation(
            Tensor tensor,
            Operation operation,
            Map<Tensor, List<Tensor>> consumers
    ) {
        if (supportsBackward()) {
            return false;
        }
        if (tensor.getDataType() != DataType.BFLOAT16 || operation == null) {
            return false;
        }
        if (operation.opType() != Operation.OpType.MATMUL && operation.opType() != Operation.OpType.LINEAR
                && operation.opType() != Operation.OpType.SUM && operation.opType() != Operation.OpType.MEAN
                && operation.opType() != Operation.OpType.LOG_SOFTMAX
                && operation.opType() != Operation.OpType.EXPAND && operation.opType() != Operation.OpType.PERMUTE
                && operation.opType() != Operation.OpType.RESHAPE && operation.opType() != Operation.OpType.CONTIGUOUS) {
            return false;
        }
        List<Tensor> next = consumers.getOrDefault(tensor, List.of());
        if (next.size() != 1) {
            return false;
        }
        Tensor consumer = next.getFirst();
        if (consumer == null || consumer.getOperation() == null || consumer.getDataType() != DataType.BFLOAT16) {
            return false;
        }
        if (operation.opType() == Operation.OpType.SUM || operation.opType() == Operation.OpType.MEAN
                || operation.opType() == Operation.OpType.EXPAND || operation.opType() == Operation.OpType.PERMUTE
                || operation.opType() == Operation.OpType.RESHAPE || operation.opType() == Operation.OpType.CONTIGUOUS) {
            return isSingleInputAliasOrReductionChain(tensor, consumer, consumers);
        }
        return switch (consumer.getOperation().opType()) {
            case RELU, ABS, CLAMP_MIN, CLAMP_MAX, SQRT, EXP, FAST_EXP, LOG, TANH, FAST_TANH, SIGMOID, INV ->
                    isSupportedUnaryContinuationConsumer(consumer, tensor);
            case ADD, SUB, MUL, DIV, MIN, MAX ->
                    isSupportedBinaryContinuationConsumer(consumer, tensor);
            case SUM, MEAN, SOFTMAX, LOG_SOFTMAX -> isSupportedReductionContinuationConsumer(consumer, tensor);
            case NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES -> isSupportedDenseLossContinuationConsumer(consumer, tensor);
            case FUSED -> isSupportedFusedContinuationConsumer((FusedOperation) consumer.getOperation(), consumer, tensor);
            default -> false;
        };
    }

    private boolean isSupportedUnaryContinuationConsumer(Tensor consumer, Tensor producer) {
        return consumer.getPrevTensors() != null
                && consumer.getPrevTensors().size() == 1
                && consumer.getPrevTensors().getFirst() == producer;
    }

    private boolean isSupportedBinaryContinuationConsumer(Tensor consumer, Tensor producer) {
        if (consumer.getPrevTensors() == null || consumer.getPrevTensors().size() != 2) {
            return false;
        }
        Tensor left = consumer.getPrevTensors().get(0);
        Tensor right = consumer.getPrevTensors().get(1);
        if (left != producer && right != producer) {
            return false;
        }
        Tensor other = left == producer ? right : left;
        if (other == null || other.getDataType() != DataType.BFLOAT16) {
            return false;
        }
        if (!java.util.Arrays.equals(producer.getShapeUnsafe(), consumer.getShapeUnsafe())
                || !java.util.Arrays.equals(other.getShapeUnsafe(), consumer.getShapeUnsafe())) {
            return false;
        }
        return producer.isContiguous() && !producer.hasStorageOffset()
                && other.isContiguous() && !other.hasStorageOffset()
                && consumer.isContiguous() && !consumer.hasStorageOffset();
    }

    private boolean isSupportedReductionContinuationConsumer(Tensor consumer, Tensor producer) {
        return consumer.getPrevTensors() != null
                && consumer.getPrevTensors().size() == 1
                && consumer.getPrevTensors().getFirst() == producer
                && producer.isContiguous()
                && !producer.hasStorageOffset();
    }

    private boolean isSupportedDenseLossContinuationConsumer(Tensor consumer, Tensor producer) {
        if (consumer.getPrevTensors() == null || consumer.getPrevTensors().size() != 2) {
            return false;
        }
        if (consumer.getPrevTensors().getFirst() != producer) {
            return false;
        }
        Tensor targets = consumer.getPrevTensors().get(1);
        return targets != null
                && targets.getDataType() == DataType.BFLOAT16
                && java.util.Arrays.equals(targets.getShapeUnsafe(), producer.getShapeUnsafe())
                && producer.isContiguous()
                && !producer.hasStorageOffset();
    }

    private boolean isSupportedFusedContinuationConsumer(FusedOperation fused, Tensor consumer, Tensor producer) {
        if (fused == null || consumer.getPrevTensors() == null || !consumer.getPrevTensors().contains(producer)) {
            return false;
        }
        var plan = fused.getPlan();
        if (plan.outputNode().outputType() != DataType.BFLOAT16 || !consumer.isContiguous() || consumer.hasStorageOffset()) {
            return false;
        }
        for (int i = 0; i < plan.inputCount(); i++) {
            var inputPlan = plan.inputs().get(i);
            Tensor input = consumer.getPrevTensors().get(i);
            if (inputPlan.dataType() != DataType.BFLOAT16 || !inputPlan.isLinearAccess()) {
                return false;
            }
            if (input == null || input.getDataType() != DataType.BFLOAT16 || !input.isContiguous() || input.hasStorageOffset()) {
                return false;
            }
        }
        for (var node : plan.nodes()) {
            if (node.outputType() == DataType.BOOL) {
                return false;
            }
            switch (node.opType()) {
                case ADD, SUB, MUL, DIV, MIN, MAX, NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH,
                        POW, SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID, NOOP -> {
                }
                default -> {
                    return false;
                }
            }
        }
        return true;
    }

    private boolean isSingleInputAliasOrReductionChain(Tensor producer, Tensor consumer, Map<Tensor, List<Tensor>> consumers) {
        if (consumer == null || consumer.getOperation() == null || consumer.getPrevTensors() == null
                || consumer.getPrevTensors().size() != 1 || consumer.getPrevTensors().getFirst() != producer) {
            return false;
        }
        Operation.OpType opType = consumer.getOperation().opType();
        if (opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN) {
            return true;
        }
        if (opType != Operation.OpType.EXPAND && opType != Operation.OpType.PERMUTE
                && opType != Operation.OpType.RESHAPE && opType != Operation.OpType.CONTIGUOUS) {
            return false;
        }
        List<Tensor> next = consumers.getOrDefault(consumer, List.of());
        if (next.size() != 1) {
            return false;
        }
        Tensor nextConsumer = next.getFirst();
        return nextConsumer != null
                && nextConsumer.getDataType() == DataType.BFLOAT16
                && isSingleInputAliasOrReductionChain(consumer, nextConsumer, consumers);
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
            tensor.setGradient(null);
            tensor.setBackward(false);
        }
        rootTensor.setGradient(null);
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
