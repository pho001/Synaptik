package backend.cpu.lowering;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.intent.BackendIntentPlan;

import backend.contract.ComputeBackend;
import backend.blas.BlasProvider;
import backend.cpu.fused.plan.FusedOperationPreparation;
import backend.lowering.region.CpuFusedRegionPayload;
import backend.lowering.region.CpuSpecializedPrimitivePayload;
import backend.lowering.region.RegionExecutionPlan;
import backend.lowering.BackendCapabilities;
import backend.lowering.LoweredRegion;
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
import graph.compile.planning.memory.MemoryPlanner;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionBoundaryReason;
import graph.compile.planning.partition.PartitionEdge;
import graph.compile.planning.partition.PartitionPlannerStrategy;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.PartitionValue;
import graph.compile.planning.value.GraphValueRef;
import graph.compile.planning.region.DefaultRegionOptimizer;
import graph.compile.planning.region.OptimizedRegion;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.compile.planning.region.specialization.RegionSpecializationKind;
import org.junit.jupiter.api.Test;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class CpuRegionLowererTest {
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
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of())
        ));

        LoweredRegion lowered = result.loweredRegion();
        assertNotNull(lowered);
        assertEquals(backend.lowering.LoweringFamily.FUSED_NATIVE, lowered.units().getFirst().loweringFamily());
        RegionExecutionPlan regionPlan = assertInstanceOf(RegionExecutionPlan.class, lowered.units().getFirst().artifact());
        assertInstanceOf(CpuFusedRegionPayload.class, regionPlan.backendPayload());
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
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

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

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(blasRuntime, compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of())
        ));

        assertEquals(backend.lowering.LoweringFamily.DIRECT_KERNEL, result.loweredRegion().units().getFirst().loweringFamily());
        RegionExecutionPlan regionPlan = assertInstanceOf(
                RegionExecutionPlan.class,
                result.loweredRegion().units().getFirst().artifact()
        );
        CpuSpecializedPrimitivePayload payload = assertInstanceOf(
                CpuSpecializedPrimitivePayload.class,
                regionPlan.backendPayload()
        );
        assertEquals(RegionSpecializationKind.MATMUL_RELU, payload.kind());
        assertEquals("CPU1_MATMUL_RELU", regionPlan.executionGroups().getFirst().physicalKernel());
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
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(openBlasRuntime(CpuStorageProfile.AUTO), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of())
        ));

        assertEquals(backend.lowering.LoweringFamily.DIRECT_KERNEL, result.loweredRegion().units().getFirst().loweringFamily());
        RegionExecutionPlan regionPlan = assertInstanceOf(
                RegionExecutionPlan.class,
                result.loweredRegion().units().getFirst().artifact()
        );
        CpuSpecializedPrimitivePayload payload = assertInstanceOf(
                CpuSpecializedPrimitivePayload.class,
                regionPlan.backendPayload()
        );
        assertEquals(RegionSpecializationKind.MATMUL_ADD_BIAS_RELU, payload.kind());
        assertEquals("CPU1_MATMUL_ADD_BIAS_RELU", regionPlan.executionGroups().getFirst().physicalKernel());
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
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of())
        ));

        LoweredRegion lowered = result.loweredRegion();
        assertEquals(1, lowered.units().size());
        assertEquals(backend.lowering.LoweringFamily.DIRECT_KERNEL, lowered.units().getFirst().loweringFamily());
        RegionExecutionPlan regionPlan = assertInstanceOf(RegionExecutionPlan.class, lowered.units().getFirst().artifact());
        assertEquals("CPU1_MATMUL_RELU", regionPlan.executionGroups().getFirst().physicalKernel());
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
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(openBlasRuntime(CpuStorageProfile.AUTO), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of())
        ));

        assertEquals(backend.lowering.LoweringFamily.DIRECT_KERNEL, result.loweredRegion().units().getFirst().loweringFamily());
        RegionExecutionPlan regionPlan = assertInstanceOf(
                RegionExecutionPlan.class,
                result.loweredRegion().units().getFirst().artifact()
        );
        assertEquals("CPU1_MATMUL_RELU", regionPlan.executionGroups().getFirst().physicalKernel());
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
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of())
        ));

        LoweredRegion lowered = result.loweredRegion();
        assertNotNull(lowered);
        assertEquals(backend.lowering.LoweringFamily.FUSED_NATIVE, lowered.units().getLast().loweringFamily());
        assertEquals(List.of(0), lowered.units().getLast().inputNodeIds());
        RegionExecutionPlan regionPlan = assertInstanceOf(RegionExecutionPlan.class, lowered.units().getLast().artifact());
        FusedOperationPreparation preparation = assertInstanceOf(CpuFusedRegionPayload.class, regionPlan.backendPayload())
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
                new graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(orderedNodeIds.size(), internalEdges.size(), externalInputNodeIds.size(), 0, Math.max(0, orderedNodeIds.size() - 1)),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION.name(),
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
