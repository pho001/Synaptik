package backend.prepare;

import backend.CPUBackend;
import backend.ComputeBackend;
import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.kernels.cpu.CpuNodeWorkspace;
import backend.kernels.cpu.ResolvedCpuComputeContract;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.fused.plan.PreparedFusedDispatch;
import backend.kernels.cpu.plan.CpuExecutionPlanner;
import backend.registry.CpuKernelResolver;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.fused.FusedExecutionBackendResolver;
import graph.fused.FusedExecutionPlan;
import graph.fused.PreparedFusedExecutable;
import operations.Operation;
import operations.fused.FusedOperation;
import tensor.DataType;

final class CpuNodePreparer {
    private static final FusedExecutionBackendResolver FUSED_BACKEND_RESOLVER = new FusedExecutionBackendResolver();

    private final config.runtime.RuntimeConfig runtimeConfig;
    private final CpuExecutionPlanner planner;

    CpuNodePreparer(config.runtime.RuntimeConfig runtimeConfig) {
        this.runtimeConfig = runtimeConfig;
        this.planner = CpuExecutionPlanner.from(runtimeConfig.cpuKernelConfig());
    }

    CompiledNodeExecutionMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        if (node.backend() != ComputeBackend.CPU) {
            return new CompiledNodeExecutionMetadata(node.backend(), null, null, null, null);
        }

        Operation operation = node.operation();
        CpuKernel kernel = CpuKernelResolver.resolve(operation.opType());
        if (kernel == null) {
            throw new IllegalStateException("Missing CPU kernel for opType=" + operation.opType());
        }

        ResolvedDispatchHints dispatchHintsOverride = null;
        PreparedFusedDispatch preparedFusedDispatch = null;
        if (operation.opType().category() == Operation.OpArityClass.ELEMENT_WISE) {
            ResolvedCpuComputeContract elementwiseContract = planner.resolveComputeContract(
                    operation,
                    node.inputTensors(),
                    node.semanticTensor(),
                    runtimeConfig.blas(),
                    null,
                    null
            );
            dispatchHintsOverride = planner.resolveDispatchHints(operation, node.semanticTensor(), elementwiseContract);
        }
        if (operation.opType() == Operation.OpType.FUSED) {
            ResolvedCpuComputeContract fusedContract = planner.resolveComputeContract(
                    operation,
                    node.inputTensors(),
                    node.semanticTensor(),
                    runtimeConfig.blas(),
                    null,
                    null
            );
            preparedFusedDispatch = planner.resolveFusedDispatch((FusedOperation) operation, node.semanticTensor(), fusedContract);
            dispatchHintsOverride = preparedFusedDispatch.dispatchHints();
        }

        CpuNodeExecutionPlan cpuPlan = CPUBackend.buildExecutionPlan(
                operation,
                node.inputTensors(),
                node.semanticTensor(),
                planner,
                runtimeConfig.blas(),
                runtimeConfig.conv2d(),
                shouldPublishFloatContinuation(node, operation, context),
                dispatchHintsOverride
        );

        PreparedFusedExecutable fusedExecutable = null;
        if (operation.opType() == Operation.OpType.FUSED && cpuPlan != null) {
            fusedExecutable = FUSED_BACKEND_RESOLVER.resolve(
                    new FusedExecutionPlan(
                            (FusedOperation) operation,
                            cpuPlan.computeContract(),
                            node.flatDataSize(),
                            preparedFusedDispatch == null ? 1 : preparedFusedDispatch.cpuVectorMinSize(),
                            preparedFusedDispatch == null ? 1 : preparedFusedDispatch.asmVectorWidth()
                    ),
                    runtimeConfig.fused()
            );
        }

        CpuNodeWorkspace cpuWorkspace = resolveCpuWorkspace(node, operation, cpuPlan, context);
        return new CompiledNodeExecutionMetadata(ComputeBackend.CPU, kernel, cpuPlan, fusedExecutable, cpuWorkspace);
    }

    private CpuNodeWorkspace resolveCpuWorkspace(
            CompiledNode node,
            Operation operation,
            CpuNodeExecutionPlan cpuPlan,
            BackendPrepareContext context
    ) {
        return switch (operation.opType()) {
            case MAX_POOL2D -> CpuNodeWorkspace.withIntWorkspace(node.flatDataSize());
            case MAX_POOL2D_BACKWARD_INPUT -> resolveSharedMaxPoolWorkspace(node, context);
            case MATMUL -> needsBFloat16BlasWorkspace(node, cpuPlan)
                    ? CpuNodeWorkspace.withFloatWorkspace(node.flatDataSize())
                    : null;
            case LINEAR -> needsBFloat16BlasWorkspace(node, cpuPlan)
                    ? CpuNodeWorkspace.withFloatWorkspaceAndPackedLinearWeights(node.flatDataSize())
                    : CpuNodeWorkspace.withPackedLinearWeights();
            case LOG_SOFTMAX -> shouldPublishFloatContinuation(node, operation, context)
                    ? CpuNodeWorkspace.withFloatWorkspace(node.flatDataSize())
                    : null;
            case CONV2D_GEMM -> node.dataType() == DataType.BFLOAT16
                    ? CpuNodeWorkspace.withFloatWorkspace(node.flatDataSize())
                    : null;
            default -> null;
        };
    }

    private boolean needsBFloat16BlasWorkspace(CompiledNode node, CpuNodeExecutionPlan cpuPlan) {
        return node.dataType() == DataType.BFLOAT16
                && cpuPlan != null
                && cpuPlan.matMulHints() != null
                && (cpuPlan.matMulHints().useBlas() || cpuPlan.matMulHints().useBatchedBlas());
    }

    private CpuNodeWorkspace resolveSharedMaxPoolWorkspace(CompiledNode node, BackendPrepareContext context) {
        for (int inputId : node.inputIds()) {
            CompiledNode input = context.compiledNode(inputId);
            if (input == null || input.operation() == null) {
                continue;
            }
            if (input.operation().opType() != Operation.OpType.MAX_POOL2D) {
                continue;
            }
            CompiledNodeExecutionMetadata metadata = context.preparedMetadataFor(input.id());
            if (metadata == null || metadata.cpuWorkspace() == null) {
                throw new IllegalStateException("Missing prepared maxPool2d workspace for backward node " + node.label());
            }
            return metadata.cpuWorkspace();
        }
        throw new IllegalStateException("maxPool2d backward node is missing its forward maxPool2d dependency.");
    }

    private boolean shouldPublishFloatContinuation(
            CompiledNode node,
            Operation operation,
            BackendPrepareContext context
    ) {
        if (context.supportsBackward()) {
            return false;
        }
        if (node.dataType() != DataType.BFLOAT16 || operation == null) {
            return false;
        }
        if (operation.opType() != Operation.OpType.MATMUL && operation.opType() != Operation.OpType.LINEAR
                && operation.opType() != Operation.OpType.SUM && operation.opType() != Operation.OpType.MEAN
                && operation.opType() != Operation.OpType.LOG_SOFTMAX
                && operation.opType() != Operation.OpType.EXPAND && operation.opType() != Operation.OpType.PERMUTE
                && operation.opType() != Operation.OpType.RESHAPE && operation.opType() != Operation.OpType.CONTIGUOUS) {
            return false;
        }
        var next = context.consumersFor(node.id());
        if (next.size() != 1) {
            return false;
        }
        CompiledNode consumer = next.getFirst();
        if (consumer == null || consumer.operation() == null || consumer.dataType() != DataType.BFLOAT16) {
            return false;
        }
        if (operation.opType() == Operation.OpType.SUM || operation.opType() == Operation.OpType.MEAN
                || operation.opType() == Operation.OpType.EXPAND || operation.opType() == Operation.OpType.PERMUTE
                || operation.opType() == Operation.OpType.RESHAPE || operation.opType() == Operation.OpType.CONTIGUOUS) {
            return isSingleInputAliasOrReductionChain(node, consumer, context);
        }
        return switch (consumer.operation().opType()) {
            case RELU, ABS, CLAMP_MIN, CLAMP_MAX, SQRT, EXP, FAST_EXP, LOG, TANH, FAST_TANH, SIGMOID, INV ->
                    isSupportedUnaryContinuationConsumer(consumer, node);
            case ADD, SUB, MUL, DIV, MIN, MAX ->
                    isSupportedBinaryContinuationConsumer(consumer, node, context);
            case SUM, MEAN, SOFTMAX, LOG_SOFTMAX -> isSupportedReductionContinuationConsumer(consumer, node);
            case NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES -> isSupportedDenseLossContinuationConsumer(consumer, node, context);
            case FUSED -> isSupportedFusedContinuationConsumer((FusedOperation) consumer.operation(), consumer, node, context);
            default -> false;
        };
    }

    private boolean isSupportedUnaryContinuationConsumer(CompiledNode consumer, CompiledNode producer) {
        return consumer.inputIds().size() == 1
                && consumer.inputIds().getFirst() == producer.id();
    }

    private boolean isSupportedBinaryContinuationConsumer(CompiledNode consumer, CompiledNode producer, BackendPrepareContext context) {
        if (consumer.inputIds().size() != 2) {
            return false;
        }
        int leftId = consumer.inputIds().get(0);
        int rightId = consumer.inputIds().get(1);
        if (leftId != producer.id() && rightId != producer.id()) {
            return false;
        }
        CompiledNode other = context.compiledNode(leftId == producer.id() ? rightId : leftId);
        if (other == null || other.dataType() != DataType.BFLOAT16) {
            return false;
        }
        if (!java.util.Arrays.equals(producer.shape(), consumer.shape())
                || !java.util.Arrays.equals(other.shape(), consumer.shape())) {
            return false;
        }
        return producer.contiguous() && !producer.hasStorageOffset()
                && other.contiguous() && !other.hasStorageOffset()
                && consumer.contiguous() && !consumer.hasStorageOffset();
    }

    private boolean isSupportedReductionContinuationConsumer(CompiledNode consumer, CompiledNode producer) {
        return consumer.inputIds().size() == 1
                && consumer.inputIds().getFirst() == producer.id()
                && producer.contiguous()
                && !producer.hasStorageOffset();
    }

    private boolean isSupportedDenseLossContinuationConsumer(CompiledNode consumer, CompiledNode producer, BackendPrepareContext context) {
        if (consumer.inputIds().size() != 2) {
            return false;
        }
        if (consumer.inputIds().getFirst() != producer.id()) {
            return false;
        }
        CompiledNode targets = context.compiledNode(consumer.inputIds().get(1));
        return targets != null
                && targets.dataType() == DataType.BFLOAT16
                && java.util.Arrays.equals(targets.shape(), producer.shape())
                && producer.contiguous()
                && !producer.hasStorageOffset();
    }

    private boolean isSupportedFusedContinuationConsumer(
            FusedOperation fused,
            CompiledNode consumer,
            CompiledNode producer,
            BackendPrepareContext context
    ) {
        if (fused == null || !consumer.inputIds().contains(producer.id())) {
            return false;
        }
        var plan = fused.getPlan();
        if (plan.outputNode().outputType() != DataType.BFLOAT16 || !consumer.contiguous() || consumer.hasStorageOffset()) {
            return false;
        }
        for (int i = 0; i < plan.inputCount(); i++) {
            var inputPlan = plan.inputs().get(i);
            CompiledNode input = context.compiledNode(consumer.inputIds().get(i));
            if (inputPlan.dataType() != DataType.BFLOAT16 || !inputPlan.isLinearAccess()) {
                return false;
            }
            if (input == null || input.dataType() != DataType.BFLOAT16 || !input.contiguous() || input.hasStorageOffset()) {
                return false;
            }
        }
        for (var fusedNode : plan.nodes()) {
            if (fusedNode.outputType() == DataType.BOOL) {
                return false;
            }
            switch (fusedNode.opType()) {
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

    private boolean isSingleInputAliasOrReductionChain(CompiledNode producer, CompiledNode consumer, BackendPrepareContext context) {
        if (consumer == null || consumer.operation() == null
                || consumer.inputIds().size() != 1 || consumer.inputIds().getFirst() != producer.id()) {
            return false;
        }
        Operation.OpType opType = consumer.operation().opType();
        if (opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN) {
            return true;
        }
        if (opType != Operation.OpType.EXPAND && opType != Operation.OpType.PERMUTE
                && opType != Operation.OpType.RESHAPE && opType != Operation.OpType.CONTIGUOUS) {
            return false;
        }
        var next = context.consumersFor(consumer.id());
        if (next.size() != 1) {
            return false;
        }
        CompiledNode nextConsumer = next.getFirst();
        return nextConsumer != null
                && nextConsumer.dataType() == DataType.BFLOAT16
                && isSingleInputAliasOrReductionChain(consumer, nextConsumer, context);
    }
}
