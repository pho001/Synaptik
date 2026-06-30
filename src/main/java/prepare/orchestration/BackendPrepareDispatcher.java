package prepare.orchestration;

import backend.contract.ComputeBackend;
import backend.cpu.prepare.CpuNodePreparer;
import backend.cpu1.prepare.Cpu1AttentionBackwardPreparer;
import backend.cpu1.prepare.Cpu1FusedElementwisePreparer;
import backend.cpu1.prepare.Cpu1MatmulPostOp;
import backend.cpu1.prepare.Cpu1MatmulPreparer;
import backend.cpu1.prepare.Cpu1MseLossPreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.cuda.prepare.CudaGpuNodePreparer;
import backend.opencl.exec.OpenClDirectPreparedExecutable;
import backend.metal.prepare.MetalNodePreparer;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.region.CpuSpecializedPrimitivePayload;
import backend.lowering.region.RegionExecutionPlan;
import config.runtime.RuntimeConfig;
import graph.model.CompiledNode;
import runtime.execution.PreparedStepMetadata;
import runtime.execution.InputResidencyRequirement;
import runtime.execution.OutputResidencyEffect;
import planning.partition.PartitionPlan;
import planning.region.specialization.RegionSpecializationCandidate;
import planning.region.specialization.RegionSpecializationKind;
import planning.value.GraphValueRef;
import operations.Operation;
import prepare.context.BackendPrepareContext;
import prepare.context.PartitionExecutionRole;

import java.util.List;
import java.util.Objects;

public final class BackendPrepareDispatcher {
    private final RuntimeConfig runtimeConfig;
    private final CpuNodePreparer cpuPreparer;
    private final Cpu1FusedElementwisePreparer cpu1FusedElementwisePreparer;
    private final Cpu1MseLossPreparer cpu1MseLossPreparer;
    private final Cpu1MatmulPreparer cpu1MatmulPreparer;
    private final Cpu1AttentionBackwardPreparer cpu1AttentionBackwardPreparer;
    private MetalNodePreparer metalPreparer;
    private CudaGpuNodePreparer cudaGpuPreparer;

    private BackendPrepareDispatcher(RuntimeConfig runtimeConfig) {
        this.runtimeConfig = runtimeConfig;
        this.cpuPreparer = new CpuNodePreparer(runtimeConfig);
        this.cpu1FusedElementwisePreparer = new Cpu1FusedElementwisePreparer(runtimeConfig);
        this.cpu1MseLossPreparer = new Cpu1MseLossPreparer(runtimeConfig);
        this.cpu1MatmulPreparer = new Cpu1MatmulPreparer();
        this.cpu1AttentionBackwardPreparer = new Cpu1AttentionBackwardPreparer();
    }

    public static BackendPrepareDispatcher from(RuntimeConfig runtimeConfig) {
        return new BackendPrepareDispatcher(Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null"));
    }

    public PreparedStepMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return switch (executionBackendFor(node, context)) {
            case CPU -> cpuPreparer.prepare(node, context);
            case GPU_METAL -> metalPreparer().prepare(node, context);
            case GPU_CUDA -> cudaGpuPreparer().prepare(node, context);
            case GPU_OPENCL -> new PreparedStepMetadata(
                    ComputeBackend.GPU_OPENCL,
                    null,
                    node.inputIds(),
                    OpenClDirectPreparedExecutable.prepare(node),
                    InputResidencyRequirement.cpuReadableAll(),
                    OutputResidencyEffect.cpuCurrentPreserveNative()
            );
        };
    }

    public PreparedStepMetadata prepareCpuFusedStep(
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

    public PreparedStepMetadata prepareCpuSpecializedStep(
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
            case SDPA_BACKWARD -> prepareCpu1SdpaBackward(outputNode, candidate, context);
            case MATMUL_RELU -> prepareCpu1MatmulRelu(outputNode, candidate, context);
            case MATMUL_ADD_BIAS -> prepareCpu1MatmulBiasEpilogue(
                    outputNode,
                    candidate,
                    context,
                    Cpu1MatmulPostOp.ADD_BIAS
            );
            case MATMUL_ADD_BIAS_RELU -> prepareCpu1MatmulBiasEpilogue(
                    outputNode,
                    candidate,
                    context,
                    Cpu1MatmulPostOp.ADD_BIAS_RELU
            );
        };
    }

    private PreparedStepMetadata prepareCpu1SdpaBackward(
            CompiledNode outputNode,
            RegionSpecializationCandidate candidate,
            BackendPrepareContext context
    ) {
        Cpu1PrepareConfig config = automaticCpu1PrepareConfig();
        Cpu1PreparedArtifact artifact = cpu1AttentionBackwardPreparer.prepare(
                outputNode,
                candidate,
                context.descriptorIndex(),
                config
        );
        List<Integer> inputNodeIds = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .toList();
        return new PreparedStepMetadata(
                ComputeBackend.CPU,
                null,
                inputNodeIds,
                artifact,
                inputResidencyRequirement(artifact.preparedAttentionBackwardUnit().storageKind()),
                outputResidencyEffect(artifact.preparedAttentionBackwardUnit().storageKind())
        );
    }

    private PreparedStepMetadata prepareCpu1MatmulRelu(
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
                automaticCpu1MatmulPrepareConfig(),
                Cpu1MatmulPostOp.RELU
        );
        return new PreparedStepMetadata(
                ComputeBackend.CPU,
                null,
                inputNodeIds,
                artifact,
                inputResidencyRequirement(artifact.preparedMatmulUnit().storageKind()),
                outputResidencyEffect(artifact.preparedMatmulUnit().storageKind())
        );
    }

    private PreparedStepMetadata prepareCpu1MatmulBiasEpilogue(
            CompiledNode outputNode,
            RegionSpecializationCandidate candidate,
            BackendPrepareContext context,
            Cpu1MatmulPostOp postOp
    ) {
        validateMatmulBiasEpilogueCandidate(outputNode, candidate, context, postOp);
        CompiledNode computeNode = context.compiledNode(candidate.orderedNodeIds().getFirst());
        List<Integer> inputNodeIds = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .toList();
        Cpu1PrepareConfig config = automaticCpu1MatmulPrepareConfig();
        Cpu1PreparedArtifact artifact = opType(computeNode) == Operation.OpType.LINEAR
                ? cpu1MatmulPreparer.prepareLinearEpilogue(
                    computeNode,
                    outputNode,
                    context.descriptorIndex(),
                    config,
                    postOp
                )
                : cpu1MatmulPreparer.prepareMatmulBiasEpilogue(
                    computeNode,
                    context.compiledNode(candidate.orderedNodeIds().get(1)),
                    outputNode,
                    context.descriptorIndex(),
                    config,
                    postOp
                );
        return new PreparedStepMetadata(
                ComputeBackend.CPU,
                null,
                inputNodeIds,
                artifact,
                inputResidencyRequirement(artifact.preparedMatmulUnit().storageKind()),
                outputResidencyEffect(artifact.preparedMatmulUnit().storageKind())
        );
    }

    private Cpu1PrepareConfig automaticCpu1MatmulPrepareConfig() {
        return Cpu1PrepareConfig.automatic(
                runtimeConfig,
                Runtime.getRuntime().availableProcessors()
        );
    }

    private Cpu1PrepareConfig automaticCpu1PrepareConfig() {
        return Cpu1PrepareConfig.automaticForRuntimeStorage(
                runtimeConfig,
                Runtime.getRuntime().availableProcessors()
        );
    }

    private static InputResidencyRequirement inputResidencyRequirement(Cpu1StorageKind storageKind) {
        return storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                ? InputResidencyRequirement.none()
                : InputResidencyRequirement.cpuReadableAll();
    }

    private static OutputResidencyEffect outputResidencyEffect(Cpu1StorageKind storageKind) {
        return storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                ? OutputResidencyEffect.none()
                : OutputResidencyEffect.cpuCurrentPreserveNative();
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

    private static void validateMatmulBiasEpilogueCandidate(
            CompiledNode outputNode,
            RegionSpecializationCandidate candidate,
            BackendPrepareContext context,
            Cpu1MatmulPostOp postOp
    ) {
        RegionSpecializationKind expectedKind = switch (postOp) {
            case ADD_BIAS -> RegionSpecializationKind.MATMUL_ADD_BIAS;
            case ADD_BIAS_RELU -> RegionSpecializationKind.MATMUL_ADD_BIAS_RELU;
            default -> throw new UnsupportedOperationException("cpu1 matmul bias epilogue does not support " + postOp);
        };
        if (candidate.kind() != expectedKind) {
            throw new UnsupportedOperationException("cpu1 " + expectedKind + " preparer does not support "
                    + candidate.kind());
        }
        if (candidate.outputValueRef().nodeId() != outputNode.id()) {
            throw new IllegalStateException(expectedKind + " specialization output node mismatch. candidate="
                    + candidate.outputValueRef().nodeId() + ", outputNode=" + outputNode.id());
        }
        if (opType(context.compiledNode(candidate.orderedNodeIds().getFirst())) == Operation.OpType.LINEAR) {
            validateLinearBiasEpilogueCandidate(outputNode, candidate, context, expectedKind, postOp);
            return;
        }
        int expectedNodes = postOp == Cpu1MatmulPostOp.ADD_BIAS ? 2 : 3;
        if (candidate.orderedNodeIds().size() != expectedNodes || candidate.inputValueRefs().size() != 3) {
            throw new UnsupportedOperationException("cpu1 " + expectedKind
                    + " expects " + expectedNodes + " nodes and three external inputs, got nodes="
                    + candidate.orderedNodeIds() + ", inputs=" + candidate.inputValueRefs());
        }
        CompiledNode matmul = context.compiledNode(candidate.orderedNodeIds().get(0));
        CompiledNode add = context.compiledNode(candidate.orderedNodeIds().get(1));
        CompiledNode relu = postOp == Cpu1MatmulPostOp.ADD_BIAS_RELU
                ? context.compiledNode(candidate.orderedNodeIds().get(2))
                : null;
        if (opType(matmul) != Operation.OpType.MATMUL
                || opType(add) != Operation.OpType.ADD
                || (postOp == Cpu1MatmulPostOp.ADD_BIAS_RELU && opType(relu) != Operation.OpType.RELU)) {
            throw new UnsupportedOperationException("cpu1 " + expectedKind
                    + " requires MATMUL -> ADD or MATMUL -> ADD -> RELU nodes.");
        }
        if (postOp == Cpu1MatmulPostOp.ADD_BIAS && add.id() != outputNode.id()) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS ADD node must be the output node.");
        }
        if (postOp == Cpu1MatmulPostOp.ADD_BIAS_RELU
                && (relu.id() != outputNode.id()
                || relu.inputIds().size() != 1
                || relu.inputIds().getFirst() != add.id())) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU RELU node must consume the ADD output.");
        }
        if (add.inputIds().size() != 2 || !add.inputIds().contains(matmul.id())) {
            throw new UnsupportedOperationException("cpu1 " + expectedKind + " ADD node must consume the MATMUL output.");
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
            throw new UnsupportedOperationException("cpu1 " + expectedKind + " inputs do not match candidate inputs.");
        }
    }

    private static void validateLinearBiasEpilogueCandidate(
            CompiledNode outputNode,
            RegionSpecializationCandidate candidate,
            BackendPrepareContext context,
            RegionSpecializationKind expectedKind,
            Cpu1MatmulPostOp postOp
    ) {
        if (candidate.inputValueRefs().size() != 3) {
            throw new UnsupportedOperationException("cpu1 " + expectedKind
                    + " LINEAR expects three external inputs, got nodes="
                    + candidate.orderedNodeIds() + ", inputs=" + candidate.inputValueRefs());
        }
        int expectedNodes = postOp == Cpu1MatmulPostOp.ADD_BIAS ? 1 : 2;
        if (candidate.orderedNodeIds().size() != expectedNodes) {
            throw new UnsupportedOperationException("cpu1 " + expectedKind
                    + " LINEAR expects " + expectedNodes + " nodes, got " + candidate.orderedNodeIds());
        }
        CompiledNode linear = context.compiledNode(candidate.orderedNodeIds().get(0));
        CompiledNode relu = postOp == Cpu1MatmulPostOp.ADD_BIAS_RELU
                ? context.compiledNode(candidate.orderedNodeIds().get(1))
                : null;
        if (opType(linear) != Operation.OpType.LINEAR
                || (postOp == Cpu1MatmulPostOp.ADD_BIAS_RELU && opType(relu) != Operation.OpType.RELU)) {
            throw new UnsupportedOperationException("cpu1 " + expectedKind
                    + " requires LINEAR or LINEAR -> RELU nodes.");
        }
        if (postOp == Cpu1MatmulPostOp.ADD_BIAS && linear.id() != outputNode.id()) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS LINEAR node must be the output node.");
        }
        if (postOp == Cpu1MatmulPostOp.ADD_BIAS_RELU
                && (relu.id() != outputNode.id()
                || relu.inputIds().size() != 1
                || relu.inputIds().getFirst() != linear.id())) {
            throw new UnsupportedOperationException("cpu1 MATMUL_ADD_BIAS_RELU RELU node must consume the LINEAR output.");
        }
        List<Integer> expectedInputs = linear.inputIds();
        List<Integer> actualInputs = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .toList();
        if (!expectedInputs.equals(actualInputs)) {
            throw new UnsupportedOperationException("cpu1 " + expectedKind + " LINEAR inputs do not match candidate inputs.");
        }
    }

    private static Operation.OpType opType(CompiledNode node) {
        return node == null || node.operation() == null ? Operation.OpType.UNKNOWN : node.operation().opType();
    }

    public PreparedStepMetadata prepareMetalRegionStep(
            backend.lowering.LoweredRegion loweredRegion,
            BackendPrepareContext context
    ) {
        Objects.requireNonNull(loweredRegion, "loweredRegion cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return metalPreparer().prepareRegionStep(loweredRegion, context);
    }

    public PreparedStepMetadata prepareCudaRegionStep(
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
