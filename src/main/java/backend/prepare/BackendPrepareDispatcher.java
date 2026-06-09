package backend.prepare;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.cpu.prepare.CpuNodePreparer;
import backend.cpu1.prepare.Cpu1FusedElementwisePreparer;
import backend.cpu1.prepare.Cpu1MatmulPostOp;
import backend.cpu1.prepare.Cpu1MatmulPreparer;
import backend.cpu1.prepare.Cpu1MseLossPreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cuda.prepare.CudaGpuNodePreparer;
import backend.metal.prepare.MetalNodePreparer;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.region.CpuSpecializedPrimitivePayload;
import backend.lowering.region.RegionExecutionPlan;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.InputResidencyRequirement;
import graph.execution.plan.OutputResidencyEffect;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.region.specialization.RegionSpecializationCandidate;
import graph.compile.planning.region.specialization.RegionSpecializationKind;
import graph.compile.planning.value.GraphValueRef;
import operations.Operation;

import java.util.List;
import java.util.Objects;

public final class BackendPrepareDispatcher {
    private final RuntimeConfig runtimeConfig;
    private final CpuNodePreparer cpuPreparer;
    private final Cpu1FusedElementwisePreparer cpu1FusedElementwisePreparer;
    private final Cpu1MseLossPreparer cpu1MseLossPreparer;
    private final Cpu1MatmulPreparer cpu1MatmulPreparer;
    private MetalNodePreparer metalPreparer;
    private CudaGpuNodePreparer cudaGpuPreparer;

    private BackendPrepareDispatcher(RuntimeConfig runtimeConfig) {
        this.runtimeConfig = runtimeConfig;
        this.cpuPreparer = new CpuNodePreparer(runtimeConfig);
        this.cpu1FusedElementwisePreparer = new Cpu1FusedElementwisePreparer(runtimeConfig);
        this.cpu1MseLossPreparer = new Cpu1MseLossPreparer(runtimeConfig);
        this.cpu1MatmulPreparer = new Cpu1MatmulPreparer();
    }

    public static BackendPrepareDispatcher from(RuntimeConfig runtimeConfig) {
        return new BackendPrepareDispatcher(Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null"));
    }

    public CompiledNodeExecutionMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return switch (executionBackendFor(node, context)) {
            case CPU -> cpuPreparer.prepare(node, context);
            case GPU_METAL -> metalPreparer().prepare(node, context);
            case GPU_CUDA -> cudaGpuPreparer().prepare(node, context);
            case GPU_OPENCL ->
                    new CompiledNodeExecutionMetadata(node.backend(), null, java.util.List.of(), null);
        };
    }

    public CompiledNodeExecutionMetadata prepareCpuFusedStep(
            CompiledNode outputNode,
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        Objects.requireNonNull(outputNode, "outputNode cannot be null");
        Objects.requireNonNull(loweredUnit, "loweredUnit cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        if (runtimeConfig.fused().useCpu1Elementwise()) {
            return cpu1FusedElementwisePreparer.prepare(outputNode, loweredUnit, context);
        }
        return cpuPreparer.prepareLoweredFusedStep(outputNode, loweredUnit, context);
    }

    public CompiledNodeExecutionMetadata prepareCpuSpecializedStep(
            CompiledNode outputNode,
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        Objects.requireNonNull(outputNode, "outputNode cannot be null");
        Objects.requireNonNull(loweredUnit, "loweredUnit cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        RegionSpecializationCandidate candidate = requireSpecializationCandidate(loweredUnit);
        return switch (candidate.kind()) {
            case MSE_LOSS -> cpu1MseLossPreparer.prepare(outputNode, loweredUnit, context);
            case MATMUL_RELU -> prepareCpu1MatmulRelu(outputNode, candidate, context);
            case MATMUL_ADD_BIAS_RELU -> prepareCpu1MatmulBiasRelu(outputNode, candidate, context);
        };
    }

    private CompiledNodeExecutionMetadata prepareCpu1MatmulRelu(
            CompiledNode outputNode,
            RegionSpecializationCandidate candidate,
            BackendPrepareContext context
    ) {
        validateMatmulReluCandidate(outputNode, candidate, context);
        CompiledNode matmulNode = context.compiledNode(candidate.orderedNodeIds().getFirst());
        List<Integer> inputNodeIds = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .toList();
        Cpu1PreparedArtifact artifact = cpu1MatmulPreparer.prepare(
                matmulNode,
                outputNode,
                context.descriptorIndex(),
                Cpu1PrepareConfig.automatic(runtimeConfig, Runtime.getRuntime().availableProcessors()),
                Cpu1MatmulPostOp.RELU
        );
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                inputNodeIds,
                artifact,
                InputResidencyRequirement.cpuReadableAll(),
                OutputResidencyEffect.cpuCurrentPreserveNative()
        );
    }

    private CompiledNodeExecutionMetadata prepareCpu1MatmulBiasRelu(
            CompiledNode outputNode,
            RegionSpecializationCandidate candidate,
            BackendPrepareContext context
    ) {
        validateMatmulBiasReluCandidate(outputNode, candidate, context);
        CompiledNode matmulNode = context.compiledNode(candidate.orderedNodeIds().get(0));
        List<Integer> inputNodeIds = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .toList();
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(runtimeConfig, Runtime.getRuntime().availableProcessors());
        Cpu1PreparedArtifact artifact;
        if (opType(matmulNode) == Operation.OpType.LINEAR) {
            artifact = cpu1MatmulPreparer.prepareLinearBiasRelu(
                    matmulNode,
                    outputNode,
                    context.descriptorIndex(),
                    config
            );
        } else {
            CompiledNode addNode = context.compiledNode(candidate.orderedNodeIds().get(1));
            artifact = cpu1MatmulPreparer.prepareMatmulBiasRelu(
                    matmulNode,
                    addNode,
                    outputNode,
                    context.descriptorIndex(),
                    config
            );
        }
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                inputNodeIds,
                artifact,
                InputResidencyRequirement.cpuReadableAll(),
                OutputResidencyEffect.cpuCurrentPreserveNative()
        );
    }

    private static RegionSpecializationCandidate requireSpecializationCandidate(LoweredExecutionUnit loweredUnit) {
        RegionExecutionPlan plan = loweredUnit.requireRegionPlan();
        if (!(plan.backendPayload() instanceof CpuSpecializedPrimitivePayload payload)) {
            throw new IllegalStateException("CPU specialized prepare requires CpuSpecializedPrimitivePayload.");
        }
        return payload.candidate();
    }

    private static void validateMatmulReluCandidate(
            CompiledNode outputNode,
            RegionSpecializationCandidate candidate,
            BackendPrepareContext context
    ) {
        if (candidate.kind() != RegionSpecializationKind.MATMUL_RELU) {
            throw new UnsupportedOperationException("cpu1 MATMUL_RELU preparer does not support " + candidate.kind());
        }
        if (candidate.outputValueRef().nodeId() != outputNode.id()) {
            throw new IllegalStateException("MATMUL_RELU specialization output node mismatch. candidate="
                    + candidate.outputValueRef().nodeId() + ", outputNode=" + outputNode.id());
        }
        if (candidate.orderedNodeIds().size() != 2 || candidate.inputValueRefs().size() != 2) {
            throw new UnsupportedOperationException("cpu1 MATMUL_RELU expects two nodes and two external inputs, got nodes="
                    + candidate.orderedNodeIds() + ", inputs=" + candidate.inputValueRefs());
        }
        CompiledNode matmul = context.compiledNode(candidate.orderedNodeIds().get(0));
        CompiledNode relu = context.compiledNode(candidate.orderedNodeIds().get(1));
        if (opType(matmul) != Operation.OpType.MATMUL || opType(relu) != Operation.OpType.RELU) {
            throw new UnsupportedOperationException("cpu1 MATMUL_RELU requires MATMUL -> RELU nodes.");
        }
        if (relu.id() != outputNode.id() || relu.inputIds().size() != 1 || relu.inputIds().getFirst() != matmul.id()) {
            throw new UnsupportedOperationException("cpu1 MATMUL_RELU RELU node must consume the MATMUL output.");
        }
        List<Integer> expectedInputs = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .toList();
        if (!matmul.inputIds().equals(expectedInputs)) {
            throw new UnsupportedOperationException("cpu1 MATMUL_RELU MATMUL inputs do not match candidate inputs.");
        }
    }

    private static void validateMatmulBiasReluCandidate(
            CompiledNode outputNode,
            RegionSpecializationCandidate candidate,
            BackendPrepareContext context
    ) {
        if (candidate.kind() != RegionSpecializationKind.MATMUL_ADD_BIAS_RELU) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU preparer does not support "
                    + candidate.kind());
        }
        if (candidate.outputValueRef().nodeId() != outputNode.id()) {
            throw new IllegalStateException("MATMUL_ADD_BIAS_RELU specialization output node mismatch. candidate="
                    + candidate.outputValueRef().nodeId() + ", outputNode=" + outputNode.id());
        }
        if (candidate.orderedNodeIds().size() == 2) {
            validateLinearBiasReluCandidate(outputNode, candidate, context);
            return;
        }
        if (candidate.orderedNodeIds().size() != 3 || candidate.inputValueRefs().size() != 3) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU expects three nodes and three external inputs, got nodes="
                    + candidate.orderedNodeIds() + ", inputs=" + candidate.inputValueRefs());
        }
        CompiledNode matmul = context.compiledNode(candidate.orderedNodeIds().get(0));
        CompiledNode add = context.compiledNode(candidate.orderedNodeIds().get(1));
        CompiledNode relu = context.compiledNode(candidate.orderedNodeIds().get(2));
        if (opType(matmul) != Operation.OpType.MATMUL
                || opType(add) != Operation.OpType.ADD
                || opType(relu) != Operation.OpType.RELU) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU requires MATMUL -> ADD -> RELU nodes.");
        }
        if (relu.id() != outputNode.id() || relu.inputIds().size() != 1 || relu.inputIds().getFirst() != add.id()) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU RELU node must consume the ADD output.");
        }
        if (add.inputIds().size() != 2 || !add.inputIds().contains(matmul.id())) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU ADD node must consume the MATMUL output.");
        }
        int biasNodeId = add.inputIds().get(0) == matmul.id()
                ? add.inputIds().get(1)
                : add.inputIds().get(0);
        List<Integer> expectedInputs = List.of(
                matmul.inputIds().get(0),
                matmul.inputIds().get(1),
                biasNodeId
        );
        List<Integer> actualInputs = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .toList();
        if (!expectedInputs.equals(actualInputs)) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU inputs do not match candidate inputs.");
        }
    }

    private static void validateLinearBiasReluCandidate(
            CompiledNode outputNode,
            RegionSpecializationCandidate candidate,
            BackendPrepareContext context
    ) {
        if (candidate.inputValueRefs().size() != 3) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU LINEAR expects three external inputs, got nodes="
                    + candidate.orderedNodeIds() + ", inputs=" + candidate.inputValueRefs());
        }
        CompiledNode linear = context.compiledNode(candidate.orderedNodeIds().get(0));
        CompiledNode relu = context.compiledNode(candidate.orderedNodeIds().get(1));
        if (opType(linear) != Operation.OpType.LINEAR || opType(relu) != Operation.OpType.RELU) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU requires MATMUL -> ADD -> RELU or LINEAR -> RELU nodes.");
        }
        if (relu.id() != outputNode.id() || relu.inputIds().size() != 1 || relu.inputIds().getFirst() != linear.id()) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU RELU node must consume the LINEAR output.");
        }
        List<Integer> expectedInputs = linear.inputIds();
        List<Integer> actualInputs = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .toList();
        if (!expectedInputs.equals(actualInputs)) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU LINEAR inputs do not match candidate inputs.");
        }
    }

    private static Operation.OpType opType(CompiledNode node) {
        return node == null || node.operation() == null ? Operation.OpType.UNKNOWN : node.operation().opType();
    }

    public CompiledNodeExecutionMetadata prepareMetalRegionStep(
            backend.lowering.LoweredRegion loweredRegion,
            BackendPrepareContext context
    ) {
        Objects.requireNonNull(loweredRegion, "loweredRegion cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return metalPreparer().prepareRegionStep(loweredRegion, context);
    }

    public CompiledNodeExecutionMetadata prepareCudaRegionStep(
            backend.lowering.LoweredRegion loweredRegion,
            BackendPrepareContext context
    ) {
        Objects.requireNonNull(loweredRegion, "loweredRegion cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return cudaGpuPreparer().prepareRegionStep(loweredRegion, context);
    }

    private MetalNodePreparer metalPreparer() {
        if (metalPreparer == null) {
            metalPreparer = new MetalNodePreparer(cpuPreparer);
        }
        return metalPreparer;
    }

    private CudaGpuNodePreparer cudaGpuPreparer() {
        if (cudaGpuPreparer == null) {
            cudaGpuPreparer = new CudaGpuNodePreparer(cpuPreparer);
        }
        return cudaGpuPreparer;
    }

    private ComputeBackend executionBackendFor(CompiledNode node, BackendPrepareContext context) {
        if (context.partitionRoleFor(node.id()) == PartitionExecutionRole.ANCHOR) {
            PartitionPlan selectedPlan = context.backendPlanForAnchor(node.id());
            if (selectedPlan != null && selectedPlan.backend() != null) {
                return selectedPlan.backend();
            }
        }
        return node.backend();
    }
}
