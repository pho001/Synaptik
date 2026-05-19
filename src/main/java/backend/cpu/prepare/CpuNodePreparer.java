package backend.cpu.prepare;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.cpu.CpuBackend;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.CpuNodeWorkspace;
import backend.cpu.kernels.ResolvedCpuComputeContract;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.fused.plan.PreparedFusedDispatch;
import backend.cpu.kernels.plan.CpuExecutionPlanner;
import backend.cpu.nativecpu.PreparedNativeCpuRegionExecutable;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweringFamily;
import backend.lowering.region.CpuFusedRegionPayload;
import backend.lowering.region.RegionExecutionPlan;
import backend.prepare.BackendPrepareContext;
import backend.prepare.RegionPlanValidator;
import backend.cpu.registry.CpuKernelResolver;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.PreparedNodeExecution;
import backend.cpu.fused.exec.FusedExecutionBackendResolver;
import backend.cpu.fused.plan.FusedExecutionPlan;
import backend.cpu.fused.plan.FusedOperationPreparation;
import backend.cpu.fused.exec.PreparedFusedExecutable;
import operations.Operation;
import backend.cpu.fused.plan.FusedOperation;
import tensor.layout.BroadcastPlanner;
import tensor.DataType;
import tensor.Tensor;

import config.runtime.CpuStorageProfile;
import java.util.List;

public final class CpuNodePreparer {
    private static final FusedExecutionBackendResolver FUSED_BACKEND_RESOLVER = new FusedExecutionBackendResolver();
    private record ContinuationConsumerTarget(CompiledNode consumer, int producerInputIndex) {}

    private final config.runtime.RuntimeConfig runtimeConfig;
    private final CpuExecutionPlanner planner;

    public CpuNodePreparer(config.runtime.RuntimeConfig runtimeConfig) {
        this.runtimeConfig = runtimeConfig;
        this.planner = CpuExecutionPlanner.from(runtimeConfig.cpuKernelConfig());
    }

    public CompiledNodeExecutionMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        if (node.backend() != ComputeBackend.CPU) {
            return new CompiledNodeExecutionMetadata(node.backend(), null, null, null, null, null, PartitionExecutionRole.NONE);
        }
        PartitionExecutionRole role = context.partitionRoleFor(node.id());
        if (role == PartitionExecutionRole.INTERIOR) {
            return new CompiledNodeExecutionMetadata(ComputeBackend.CPU, null, null, null, null, null, role);
        }
        if (role == PartitionExecutionRole.ANCHOR) {
            LoweredExecutionUnit loweredUnit = context.cpuLoweredUnitForAnchor(node.id());
            if (loweredUnit != null && loweredUnit.loweringFamily() == LoweringFamily.FUSED_NATIVE) {
                return prepareLoweredFusedAnchor(node, loweredUnit, context);
            }
            if (loweredUnit != null && loweredUnit.loweringFamily() == LoweringFamily.CPU_NATIVE_REGION) {
                return prepareNativeCpuRegionAnchor(loweredUnit, context);
            }
        }
        return prepareAsCpu(node, context);
    }

    private CompiledNodeExecutionMetadata prepareNativeCpuRegionAnchor(
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        RegionExecutionPlan regionPlan = loweredUnit.requireRegionPlan();
        RegionPlanValidator.requireBoundaryCoverage(regionPlan, context);
        List<PreparedNodeExecution> nativeSteps = regionPlan.orderedNodeIds().stream()
                .map(context::compiledNode)
                .map(node -> {
                    if (node == null) {
                        throw new IllegalStateException("Missing compiled node for CPU native region " + regionPlan.regionId());
                    }
                    return new PreparedNodeExecution(node, prepareAsCpu(node, context));
                })
                .toList();

        CpuNodePreparer fallbackPreparer = new CpuNodePreparer(
                runtimeConfig.withCpuStorageProfile(CpuStorageProfile.CPU_ARRAY)
        );
        List<PreparedNodeExecution> fallbackSteps = regionPlan.orderedNodeIds().stream()
                .map(context::compiledNode)
                .map(node -> {
                    if (node == null) {
                        throw new IllegalStateException("Missing compiled node for CPU native region fallback " + regionPlan.regionId());
                    }
                    return new PreparedNodeExecution(node, fallbackPreparer.prepareAsCpu(node, context));
                })
                .toList();

        PreparedNativeCpuRegionExecutable executable = new PreparedNativeCpuRegionExecutable(
                regionPlan,
                nativeSteps,
                fallbackSteps
        );
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                executable,
                PartitionExecutionRole.ANCHOR
        );
    }

    private CompiledNodeExecutionMetadata prepareLoweredFusedAnchor(
            CompiledNode anchorNode,
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        FusedOperationPreparation fusedPreparation = fusedPreparation(loweredUnit);
        Operation operation = fusedPreparation.operation();
        CpuKernel kernel = CpuKernelResolver.resolve(operation.opType());
        boolean publishFloatContinuation = shouldPublishFloatContinuation(anchorNode, operation, context);

        ResolvedCpuComputeContract fusedContract = planner.resolveComputeContract(
                operation,
                fusedPreparation.runtimeInputs(),
                anchorNode.semanticTensor(),
                runtimeConfig.blas(),
                null,
                null
        );
        PreparedFusedDispatch preparedFusedDispatch = planner.resolveFusedDispatch(
                (FusedOperation) operation,
                anchorNode.semanticTensor(),
                fusedContract
        );
        ResolvedDispatchHints dispatchHintsOverride = preparedFusedDispatch.dispatchHints();
        CpuNodeExecutionPlan cpuPlan = CpuBackend.buildExecutionPlan(
                operation,
                fusedPreparation.runtimeInputs(),
                anchorNode.semanticTensor(),
                planner,
                runtimeConfig.blas(),
                runtimeConfig.conv2d(),
                runtimeConfig.cpuStorageProfile(),
                publishFloatContinuation,
                dispatchHintsOverride
        );
        PreparedFusedExecutable fusedExecutable = FUSED_BACKEND_RESOLVER.resolve(
                new FusedExecutionPlan(
                        (FusedOperation) operation,
                        cpuPlan.computeContract(),
                        anchorNode.flatDataSize(),
                        preparedFusedDispatch.cpuVectorMinSize(),
                        preparedFusedDispatch.asmVectorWidth()
                ),
                runtimeConfig.fused()
        );
        CpuNodeWorkspace cpuWorkspace = resolveCpuWorkspace(anchorNode, operation, cpuPlan, publishFloatContinuation, context);
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                kernel,
                cpuPlan,
                fusedExecutable,
                cpuWorkspace,
                null,
                operation,
                loweredUnit.inputNodeIds(),
                PartitionExecutionRole.ANCHOR
        );
    }

    private FusedOperationPreparation fusedPreparation(LoweredExecutionUnit loweredUnit) {
        if (loweredUnit.artifact() instanceof RegionExecutionPlan plan
                && plan.backendPayload() instanceof CpuFusedRegionPayload payload) {
            return payload.requirePreparation(FusedOperationPreparation.class);
        }
        return loweredUnit.requireArtifact(FusedOperationPreparation.class);
    }

    public CompiledNodeExecutionMetadata prepareAsCpu(CompiledNode node, BackendPrepareContext context) {
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

        CpuNodeExecutionPlan cpuPlan = CpuBackend.buildExecutionPlan(
                operation,
                node.inputTensors(),
                node.semanticTensor(),
                planner,
                runtimeConfig.blas(),
                runtimeConfig.conv2d(),
                runtimeConfig.cpuStorageProfile(),
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
        return new CompiledNodeExecutionMetadata(ComputeBackend.CPU, kernel, cpuPlan, fusedExecutable, cpuWorkspace, null, PartitionExecutionRole.NONE);
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
            case WHERE -> isSupportedWhereContinuationConsumer(target, context);
            case MATMUL -> isSupportedMatMulContinuationConsumer(target, context);
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
        CompiledNode producer = context.compiledNode(consumer.inputIds().get(target.producerInputIndex()));
        int otherIndex = target.producerInputIndex() == 0 ? 1 : 0;
        CompiledNode other = context.compiledNode(consumer.inputIds().get(otherIndex));
        if (producer == null || producer.dataType() != DataType.BFLOAT16 || other == null || other.dataType() != DataType.BFLOAT16) {
            return false;
        }
        return producer.contiguous()
                && !producer.hasStorageOffset()
                && other.contiguous()
                && !other.hasStorageOffset()
                && consumer.contiguous()
                && !consumer.hasStorageOffset()
                && isBroadcastCompatible(producer, other, consumer.shape());
    }

    private boolean isSupportedWhereContinuationConsumer(ContinuationConsumerTarget target, BackendPrepareContext context) {
        CompiledNode consumer = target.consumer();
        if (consumer.inputIds().size() != 3 || target.producerInputIndex() == 0) {
            return false;
        }
        CompiledNode producer = context.compiledNode(consumer.inputIds().get(target.producerInputIndex()));
        CompiledNode condition = context.compiledNode(consumer.inputIds().get(0));
        int otherBranchIndex = target.producerInputIndex() == 1 ? 2 : 1;
        CompiledNode otherBranch = context.compiledNode(consumer.inputIds().get(otherBranchIndex));
        if (producer == null
                || condition == null
                || otherBranch == null
                || producer.dataType() != DataType.BFLOAT16
                || condition.dataType() != DataType.BOOL
                || otherBranch.dataType() != DataType.BFLOAT16) {
            return false;
        }
        return producer.contiguous()
                && !producer.hasStorageOffset()
                && condition.contiguous()
                && !condition.hasStorageOffset()
                && otherBranch.contiguous()
                && !otherBranch.hasStorageOffset()
                && consumer.contiguous()
                && !consumer.hasStorageOffset()
                && isBroadcastCompatible(producer, otherBranch, consumer.shape())
                && isBroadcastCompatible(condition, producer, consumer.shape());
    }

    private boolean isSupportedReductionContinuationConsumer(ContinuationConsumerTarget target) {
        return target.producerInputIndex() == 0
                && target.consumer().inputIds().size() == 1;
    }

    private boolean isSupportedMatMulContinuationConsumer(ContinuationConsumerTarget target, BackendPrepareContext context) {
        CompiledNode consumer = target.consumer();
        if (consumer.inputIds().size() != 2) {
            return false;
        }
        int otherIndex = target.producerInputIndex() == 0 ? 1 : 0;
        CompiledNode other = context.compiledNode(consumer.inputIds().get(otherIndex));
        return other != null
                && other.dataType() == DataType.BFLOAT16
                && other.contiguous()
                && !other.hasStorageOffset()
                && consumer.contiguous()
                && !consumer.hasStorageOffset();
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
                    POW, SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID, WHERE -> true;
            default -> false;
        };
    }

    private boolean isBroadcastCompatible(CompiledNode left, CompiledNode right, int[] expectedOutputShape) {
        try {
            return java.util.Arrays.equals(
                    BroadcastPlanner.plan(left.shape(), left.strides(), right.shape(), right.strides()).outShape(),
                    expectedOutputShape
            );
        } catch (IllegalArgumentException ignored) {
            return false;
        }
    }

    private boolean isContiguousBFloat16Parameter(CompiledNode node) {
        return node != null
                && node.dataType() == DataType.BFLOAT16
                && node.contiguous()
                && !node.hasStorageOffset();
    }
}
