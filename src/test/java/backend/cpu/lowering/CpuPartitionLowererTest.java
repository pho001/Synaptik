package backend.cpu.lowering;

import planning.descriptor.CompiledTensorDescriptorBuilder;
import planning.intent.BackendIntentPlan;

import backend.contract.ComputeBackend;
import config.runtime.BlasProvider;
import backend.cpu.fused.plan.FusedOperationPreparation;
import backend.lowering.partition.CpuFusedPartitionPayload;
import backend.lowering.partition.CpuSpecializedPrimitivePayload;
import backend.lowering.partition.BackendPartitionExecutionPlan;
import backend.lowering.BackendCapabilities;
import backend.lowering.LoweredPartition;
import backend.lowering.LoweringContext;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import config.backend.KernelTuningConfig;
import config.optimizer.FuseConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import trace.compile.PartitionDecisionTrace;
import planning.memory.MemoryPlanner;
import planning.partition.Partition;
import planning.partition.ExecutablePartitionPlan;
import planning.partition.PlannedPartition;
import planning.partition.PartitionBoundaryReason;
import planning.partition.PartitionEdge;
import planning.partition.PartitionPlannerStrategy;
import planning.partition.PartitionTarget;
import planning.partition.PartitionValue;
import planning.value.GraphValueRef;
import planning.partition.execution.PartitionExecutionPlanner;
import planning.partition.execution.PartitionExecutionPlan;
import planning.partition.execution.PartitionExecutionPlanningContext;
import planning.partition.specialization.PartitionSpecializationKind;
import org.junit.jupiter.api.Test;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class CpuPartitionLowererTest {
    @Test
    void lowersFusedElementwiseUnitToFusedNativeFamily() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-fused",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of(GraphValueRef.node(3))
        );
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(partition, new PartitionExecutionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuPartitionLowerer lowerer = new CpuPartitionLowerer();
        LoweringResult result = lowerer.lowerPartition(new LoweringRequest(
                new ExecutablePartitionPlan(new PlannedPartition(partition, null, Set.of(ComputeBackend.CPU)), executionPlan),
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes))
        ));

        LoweredPartition lowered = result.loweredPartition();
        assertNotNull(lowered);
        assertEquals(backend.lowering.LoweringFamily.FUSED_NATIVE, lowered.units().getFirst().loweringFamily());
        BackendPartitionExecutionPlan partitionPlan = assertInstanceOf(BackendPartitionExecutionPlan.class, lowered.units().getFirst().artifact());
        assertInstanceOf(CpuFusedPartitionPayload.class, partitionPlan.backendPayload());
    }

    @Test
    void lowersMatmulReluSpecializationToDirectKernelWhenBlasIsEnabledAndWorkIsLargeEnough() {
        Tensor a = new Tensor(new float[128 * 256], new int[]{128, 256}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[256 * 512], new int[]{256, 512}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-matmul",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of(GraphValueRef.node(3))
        );
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(partition, new PartitionExecutionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults()));

        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults()
                .withAccelerator(config.runtime.AcceleratorConfig.defaultsInference());
        RuntimeConfig blasRuntime = new RuntimeConfig(
                runtime.kernel(),
                runtime.approximation(),
                new BlasConfig(BlasProvider.OPENBLAS_FFM, 1L, true, 3.0d, false),
                runtime.conv2d(),
                runtime.fused(),
                runtime.accelerator()
        );

        CpuPartitionLowerer lowerer = new CpuPartitionLowerer();
        LoweringResult result = lowerer.lowerPartition(new LoweringRequest(
                new ExecutablePartitionPlan(new PlannedPartition(partition, null, Set.of(ComputeBackend.CPU)), executionPlan),
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(blasRuntime, compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes))
        ));

        assertEquals(backend.lowering.LoweringFamily.DIRECT_KERNEL, result.loweredPartition().units().getFirst().loweringFamily());
        BackendPartitionExecutionPlan partitionPlan = assertInstanceOf(
                BackendPartitionExecutionPlan.class,
                result.loweredPartition().units().getFirst().artifact()
        );
        CpuSpecializedPrimitivePayload payload = assertInstanceOf(
                CpuSpecializedPrimitivePayload.class,
                partitionPlan.backendPayload()
        );
        assertEquals(PartitionSpecializationKind.MATMUL_RELU, payload.kind());
        assertEquals("CPU1_MATMUL_RELU", partitionPlan.executionGroups().getFirst().physicalKernel());
    }

    @Test
    void lowersMatmulBiasReluSpecializationToCpu1DirectKernel() {
        Tensor a = new Tensor(new float[128 * 256], new int[]{128, 256}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[256 * 512], new int[]{256, 512}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[512], new int[]{512}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        int matmulNodeId = nodeId(compiledNodes, Operation.OpType.MATMUL);
        int addNodeId = nodeId(compiledNodes, Operation.OpType.ADD);
        int reluNodeId = nodeId(compiledNodes, Operation.OpType.RELU);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId);
        Partition partition = partition(
                "cpu-matmul-bias-relu",
                PartitionTarget.CPU,
                selectedNodeIds,
                externalInputNodeIds(compiledNodes, selectedNodeIds),
                List.of(GraphValueRef.node(reluNodeId)),
                List.of(GraphValueRef.node(reluNodeId))
        );
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(partition, new PartitionExecutionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuPartitionLowerer lowerer = new CpuPartitionLowerer();
        LoweringResult result = lowerer.lowerPartition(new LoweringRequest(
                new ExecutablePartitionPlan(new PlannedPartition(partition, null, Set.of(ComputeBackend.CPU)), executionPlan),
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(openBlasRuntime(CpuStorageProfile.AUTO), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes))
        ));

        assertEquals(backend.lowering.LoweringFamily.DIRECT_KERNEL, result.loweredPartition().units().getFirst().loweringFamily());
        BackendPartitionExecutionPlan partitionPlan = assertInstanceOf(
                BackendPartitionExecutionPlan.class,
                result.loweredPartition().units().getFirst().artifact()
        );
        CpuSpecializedPrimitivePayload payload = assertInstanceOf(
                CpuSpecializedPrimitivePayload.class,
                partitionPlan.backendPayload()
        );
        assertEquals(PartitionSpecializationKind.MATMUL_ADD_BIAS_RELU, payload.kind());
        assertEquals("CPU1_MATMUL_ADD_BIAS_RELU", partitionPlan.executionGroups().getFirst().physicalKernel());
    }

    @Test
    void cpuNativeProfileUsesCpu1MatmulReluSpecializationForExactPostOpPartition() {
        Tensor a = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-native-matmul",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of(GraphValueRef.node(3))
        );
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(partition, new PartitionExecutionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuPartitionLowerer lowerer = new CpuPartitionLowerer();
        LoweringResult result = lowerer.lowerPartition(new LoweringRequest(
                new ExecutablePartitionPlan(new PlannedPartition(partition, null, Set.of(ComputeBackend.CPU)), executionPlan),
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes))
        ));

        LoweredPartition lowered = result.loweredPartition();
        assertEquals(1, lowered.units().size());
        assertEquals(backend.lowering.LoweringFamily.DIRECT_KERNEL, lowered.units().getFirst().loweringFamily());
        BackendPartitionExecutionPlan partitionPlan = assertInstanceOf(BackendPartitionExecutionPlan.class, lowered.units().getFirst().artifact());
        assertEquals("CPU1_MATMUL_RELU", partitionPlan.executionGroups().getFirst().physicalKernel());
    }

    @Test
    void autoProfileUsesCpu1MatmulReluSpecializationForExactPostOpPartition() {
        Tensor a = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-native-auto-reject",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of(GraphValueRef.node(3))
        );
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(partition, new PartitionExecutionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuPartitionLowerer lowerer = new CpuPartitionLowerer();
        LoweringResult result = lowerer.lowerPartition(new LoweringRequest(
                new ExecutablePartitionPlan(new PlannedPartition(partition, null, Set.of(ComputeBackend.CPU)), executionPlan),
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(openBlasRuntime(CpuStorageProfile.AUTO), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes))
        ));

        assertEquals(backend.lowering.LoweringFamily.DIRECT_KERNEL, result.loweredPartition().units().getFirst().loweringFamily());
        BackendPartitionExecutionPlan partitionPlan = assertInstanceOf(
                BackendPartitionExecutionPlan.class,
                result.loweredPartition().units().getFirst().artifact()
        );
        assertEquals("CPU1_MATMUL_RELU", partitionPlan.executionGroups().getFirst().physicalKernel());
    }

    @Test
    void fusedUnitUsesExternalValueNodeExecutionInputForViewChain() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor out = base.select(0, 1).relu().exp();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-view-chain",
                PartitionTarget.CPU,
                List.of(1, 2, 3),
                List.of(0),
                List.of(GraphValueRef.node(3)),
                List.of(GraphValueRef.node(3))
        );
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(partition, new PartitionExecutionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuPartitionLowerer lowerer = new CpuPartitionLowerer();
        LoweringResult result = lowerer.lowerPartition(new LoweringRequest(
                new ExecutablePartitionPlan(new PlannedPartition(partition, null, Set.of(ComputeBackend.CPU)), executionPlan),
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes))
        ));

        LoweredPartition lowered = result.loweredPartition();
        assertNotNull(lowered);
        assertEquals(backend.lowering.LoweringFamily.FUSED_NATIVE, lowered.units().getLast().loweringFamily());
        assertEquals(List.of(0), lowered.units().getLast().inputNodeIds());
        BackendPartitionExecutionPlan partitionPlan = assertInstanceOf(BackendPartitionExecutionPlan.class, lowered.units().getLast().artifact());
        FusedOperationPreparation preparation = assertInstanceOf(CpuFusedPartitionPayload.class, partitionPlan.backendPayload())
                .requirePreparation(FusedOperationPreparation.class);
        assertEquals(List.of(1), preparation.runtimeInputNodeIds());
    }

    private static Partition partition(
            String id,
            PartitionTarget target,
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            List<GraphValueRef> outputValueRefs,
            List<GraphValueRef> requiredMaterialized
    ) {
        List<PartitionValue> values = orderedNodeIds.stream()
                .map(nodeId -> new PartitionValue(GraphValueRef.node(nodeId), nodeId))
                .toList();
        List<PartitionEdge> internalEdges = orderedNodeIds.size() < 2
                ? List.of()
                : java.util.stream.IntStream.range(0, orderedNodeIds.size() - 1)
                        .mapToObj(i -> new PartitionEdge(orderedNodeIds.get(i), orderedNodeIds.get(i + 1)))
                        .toList();
        return new Partition(
                id,
                target,
                orderedNodeIds,
                values,
                internalEdges,
                externalInputNodeIds,
                outputValueRefs,
                orderedNodeIds.getFirst(),
                requiredMaterialized,
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                orderedNodeIds.size(),
                new planning.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(orderedNodeIds.size(), internalEdges.size(), externalInputNodeIds.size(), 0, Math.max(0, orderedNodeIds.size() - 1)),
                PartitionPlannerStrategy.GREEDY_MAX_PARTITION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_PARTITION.name(),
                        target.name(),
                        orderedNodeIds.getFirst(),
                        true,
                        "test",
                        orderedNodeIds,
                        orderedNodeIds,
                        List.of(),
                        orderedNodeIds.size(),
                        0.0d,
                        0.0d,
                        0,
                        false,
                        -1
                )
        );
    }

    private static RuntimeConfig openBlasRuntime(CpuStorageProfile profile) {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                new BlasConfig(
                        BlasProvider.OPENBLAS_FFM,
                        1L,
                        false,
                        100.0d,
                        false,
                        100.0d,
                        BlasStorageMode.AUTO,
                        false
                )
        ).withCpuStorageProfile(profile);
    }

    private static int nodeId(List<CompiledNode> nodes, Operation.OpType opType) {
        return nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static List<Integer> externalInputNodeIds(List<CompiledNode> nodes, List<Integer> selectedNodeIds) {
        java.util.Set<Integer> selected = java.util.Set.copyOf(selectedNodeIds);
        java.util.LinkedHashSet<Integer> out = new java.util.LinkedHashSet<>();
        for (int nodeId : selectedNodeIds) {
            CompiledNode node = nodes.stream().filter(candidate -> candidate.id() == nodeId).findFirst().orElseThrow();
            node.inputIds().stream().filter(inputId -> !selected.contains(inputId)).forEach(out::add);
        }
        return List.copyOf(out);
    }
}
