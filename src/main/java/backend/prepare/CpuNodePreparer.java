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
    private record ContinuationConsumerTarget(CompiledNode consumer, int producerInputIndex) {}

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

        boolean publishFloatContinuation = shouldPublishFloatContinuation(node, operation, context);

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
                publishFloatContinuation,
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

        CpuNodeWorkspace cpuWorkspace = resolveCpuWorkspace(node, operation, cpuPlan, publishFloatContinuation, context);
        return new CompiledNodeExecutionMetadata(ComputeBackend.CPU, kernel, cpuPlan, fusedExecutable, cpuWorkspace);
    }

    private CpuNodeWorkspace resolveCpuWorkspace(
            CompiledNode node,
            Operation operation,
            CpuNodeExecutionPlan cpuPlan,
            boolean publishFloatContinuation,
            BackendPrepareContext context
    ) {
        return switch (operation.opType()) {
            case MAX_POOL2D -> CpuNodeWorkspace.withIntWorkspace(node.flatDataSize());
            case MAX_POOL2D_BACKWARD_INPUT -> resolveSharedMaxPoolWorkspace(node, context);
            case MATMUL -> resolveMatMulWorkspace(node, cpuPlan, publishFloatContinuation);
            case LINEAR -> (publishFloatContinuation || needsBFloat16BlasWorkspace(node, cpuPlan))
                    ? CpuNodeWorkspace.withFloatWorkspaceAndPackedLinearWeights(node.flatDataSize())
                    : CpuNodeWorkspace.withPackedLinearWeights();
            case LOG_SOFTMAX -> publishFloatContinuation
                    ? CpuNodeWorkspace.withFloatWorkspace(node.flatDataSize())
                    : null;
            case CONV2D_GEMM -> node.dataType() == DataType.BFLOAT16
                    ? CpuNodeWorkspace.withFloatWorkspace(node.flatDataSize())
                    : null;
            default -> publishFloatContinuation
                    ? CpuNodeWorkspace.withFloatWorkspace(node.flatDataSize())
                    : null;
        };
    }

    private CpuNodeWorkspace resolveMatMulWorkspace(
            CompiledNode node,
            CpuNodeExecutionPlan cpuPlan,
            boolean publishFloatContinuation
    ) {
        boolean needsFloatWorkspace = publishFloatContinuation || needsBFloat16BlasWorkspace(node, cpuPlan);
        if (node.dataType() != DataType.BFLOAT16) {
            return needsFloatWorkspace ? CpuNodeWorkspace.withFloatWorkspace(node.flatDataSize()) : null;
        }
        return needsFloatWorkspace
                ? CpuNodeWorkspace.withFloatWorkspaceAndPackedLinearWeights(node.flatDataSize())
                : CpuNodeWorkspace.withPackedLinearWeights();
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
        if (node.dataType() != DataType.BFLOAT16 || operation == null) {
            return false;
        }
        if (!supportsFloatContinuationProducer(operation.opType())) {
            return false;
        }
        ContinuationConsumerTarget target = resolveContinuationConsumerTarget(node, context);
        if (target == null) {
            return false;
        }
        CompiledNode consumer = target.consumer();
        if (consumer == null || consumer.operation() == null || consumer.dataType() != DataType.BFLOAT16) {
            return false;
        }
        return switch (consumer.operation().opType()) {
            case RELU, ABS, CLAMP_MIN, CLAMP_MAX, SQRT, EXP, FAST_EXP, LOG, TANH, FAST_TANH, SIGMOID, INV,
                    NEG, MUL_SCALAR, POW ->
                    isSupportedUnaryContinuationConsumer(target);
            case ADD, SUB, MUL, DIV, MIN, MAX ->
                    isSupportedBinaryContinuationConsumer(target, context);
            case SUM, MEAN, SOFTMAX, LOG_SOFTMAX -> isSupportedReductionContinuationConsumer(target);
            case LAYER_NORM -> isSupportedLayerNormContinuationConsumer(target, context);
            case RMS_NORM -> isSupportedRmsNormContinuationConsumer(target, context);
            case SOFTMAX_GRAD, LOG_SOFTMAX_GRAD -> isSupportedSoftmaxGradContinuationConsumer(target, context);
            case NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES -> isSupportedDenseLossContinuationConsumer(target, context);
            case FUSED -> isSupportedFusedContinuationConsumer((FusedOperation) consumer.operation(), target, context);
            default -> false;
        };
    }

    private ContinuationConsumerTarget resolveContinuationConsumerTarget(
            CompiledNode producer,
            BackendPrepareContext context
    ) {
        CompiledNode current = producer;
        while (true) {
            var next = context.consumersFor(current.id());
            if (next.size() != 1) {
                return null;
            }
            CompiledNode consumer = next.getFirst();
            if (consumer == null || consumer.operation() == null || consumer.dataType() != DataType.BFLOAT16) {
                return null;
            }
            int producerInputIndex = consumer.inputIds().indexOf(current.id());
            if (producerInputIndex < 0) {
                return null;
            }
            if (!isSafeContinuationPassthrough(current, consumer, producerInputIndex)) {
                return new ContinuationConsumerTarget(consumer, producerInputIndex);
            }
            current = consumer;
        }
    }

    private boolean isSafeContinuationPassthrough(
            CompiledNode producer,
            CompiledNode consumer,
            int producerInputIndex
    ) {
        if (consumer.operation() == null || consumer.operation().opType() == null) {
            return false;
        }
        if (producerInputIndex != 0 || consumer.inputIds().size() != 1 || consumer.dataType() != DataType.BFLOAT16) {
            return false;
        }
        if (producer.flatDataSize() != consumer.flatDataSize()) {
            return false;
        }
        return switch (consumer.operation().opType()) {
            case RESHAPE, CONTIGUOUS -> true;
            default -> false;
        };
    }

    private boolean isSupportedUnaryContinuationConsumer(ContinuationConsumerTarget target) {
        return target.producerInputIndex() == 0
                && target.consumer().inputIds().size() == 1;
    }

    private boolean isSupportedBinaryContinuationConsumer(ContinuationConsumerTarget target, BackendPrepareContext context) {
        CompiledNode consumer = target.consumer();
        if (consumer.inputIds().size() != 2) {
            return false;
        }
        int otherIndex = target.producerInputIndex() == 0 ? 1 : 0;
        CompiledNode other = context.compiledNode(consumer.inputIds().get(otherIndex));
        if (other == null || other.dataType() != DataType.BFLOAT16) {
            return false;
        }
        if (!java.util.Arrays.equals(other.shape(), consumer.shape())) {
            return false;
        }
        return other.contiguous() && !other.hasStorageOffset()
                && consumer.contiguous() && !consumer.hasStorageOffset();
    }

    private boolean isSupportedReductionContinuationConsumer(ContinuationConsumerTarget target) {
        return target.producerInputIndex() == 0
                && target.consumer().inputIds().size() == 1;
    }

    private boolean isSupportedLayerNormContinuationConsumer(ContinuationConsumerTarget target, BackendPrepareContext context) {
        CompiledNode consumer = target.consumer();
        if (target.producerInputIndex() != 0 || consumer.inputIds().size() != 3) {
            return false;
        }
        CompiledNode gamma = context.compiledNode(consumer.inputIds().get(1));
        CompiledNode beta = context.compiledNode(consumer.inputIds().get(2));
        return consumer.contiguous()
                && !consumer.hasStorageOffset()
                && isContiguousBFloat16Parameter(gamma)
                && isContiguousBFloat16Parameter(beta);
    }

    private boolean isSupportedRmsNormContinuationConsumer(ContinuationConsumerTarget target, BackendPrepareContext context) {
        CompiledNode consumer = target.consumer();
        if (target.producerInputIndex() != 0 || consumer.inputIds().size() != 2) {
            return false;
        }
        CompiledNode gamma = context.compiledNode(consumer.inputIds().get(1));
        return consumer.contiguous() && !consumer.hasStorageOffset() && isContiguousBFloat16Parameter(gamma);
    }

    private boolean isSupportedSoftmaxGradContinuationConsumer(ContinuationConsumerTarget target, BackendPrepareContext context) {
        CompiledNode consumer = target.consumer();
        if (consumer.inputIds().size() != 2) {
            return false;
        }
        int otherIndex = target.producerInputIndex() == 0 ? 1 : 0;
        CompiledNode other = context.compiledNode(consumer.inputIds().get(otherIndex));
        if (other == null || other.dataType() != DataType.BFLOAT16) {
            return false;
        }
        return java.util.Arrays.equals(other.shape(), consumer.shape())
                && other.contiguous()
                && !other.hasStorageOffset()
                && consumer.contiguous()
                && !consumer.hasStorageOffset();
    }

    private boolean isSupportedDenseLossContinuationConsumer(ContinuationConsumerTarget target, BackendPrepareContext context) {
        CompiledNode consumer = target.consumer();
        if (consumer.inputIds().size() != 2) {
            return false;
        }
        if (target.producerInputIndex() != 0) {
            return false;
        }
        CompiledNode logits = context.compiledNode(consumer.inputIds().get(0));
        CompiledNode targets = context.compiledNode(consumer.inputIds().get(1));
        return logits != null
                && targets != null
                && logits.dataType() == DataType.BFLOAT16
                && targets.dataType() == DataType.BFLOAT16
                && java.util.Arrays.equals(targets.shape(), logits.shape())
                && logits.contiguous()
                && !logits.hasStorageOffset()
                && targets.contiguous()
                && !targets.hasStorageOffset();
    }

    private boolean isSupportedFusedContinuationConsumer(
            FusedOperation fused,
            ContinuationConsumerTarget target,
            BackendPrepareContext context
    ) {
        CompiledNode consumer = target.consumer();
        if (fused == null || target.producerInputIndex() < 0) {
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

    private boolean supportsFloatContinuationProducer(Operation.OpType opType) {
        return switch (opType) {
            case MATMUL, LINEAR, SOFTMAX, LOG_SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX_GRAD, LAYER_NORM, RMS_NORM,
                    SCALED_DOT_PRODUCT_ATTENTION, SCALED_DOT_PRODUCT_ATTENTION_BACKWARD,
                    ADD, SUB, MUL, DIV, MIN, MAX, NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH,
                    POW, SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID -> true;
            default -> false;
        };
    }

    private boolean isContiguousBFloat16Parameter(CompiledNode node) {
        return node != null
                && node.dataType() == DataType.BFLOAT16
                && node.contiguous()
                && !node.hasStorageOffset();
    }
}
