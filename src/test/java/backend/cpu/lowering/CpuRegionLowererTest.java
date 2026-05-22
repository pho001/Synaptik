package backend.cpu.lowering;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;

import backend.ComputeBackend;
import backend.blas.BlasProvider;
import backend.blas.OpenBlasRuntime;
import backend.cpu.fused.plan.FusedOperationPreparation;
import backend.lowering.region.CpuFusedRegionPayload;
import backend.lowering.region.CpuNativeRegionPayload;
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
import graph.CompiledNode;
import graph.execution.trace.PartitionDecisionTrace;
import graph.compile.planning.memory.MemoryPlanner;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionBoundaryReason;
import graph.compile.planning.partition.PartitionEdge;
import graph.compile.planning.partition.PartitionPlannerStrategy;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.PartitionValue;
import graph.compile.planning.value.GraphValueRef;
import graph.compile.planning.region.DefaultRegionOptimizer;
import graph.compile.planning.region.ExecutionUnit;
import graph.compile.planning.region.ExecutionUnitKind;
import graph.compile.planning.region.OptimizedRegion;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.compile.planning.region.RegionOptimizationTrace;
import org.junit.jupiter.api.Assumptions;
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
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph, BackendIntentPlan.empty());
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
    void lowersSingleMatmulUnitToBlasWhenBlasIsEnabledAndWorkIsLargeEnough() {
        Tensor a = new Tensor(new float[128 * 256], new int[]{128, 256}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[256 * 512], new int[]{256, 512}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph, BackendIntentPlan.empty());
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

        assertEquals(backend.lowering.LoweringFamily.BLAS, result.loweredRegion().units().getFirst().loweringFamily());
    }

    @Test
    void cpuNativeProfileLowersProviderBackedPartitionToNativeRegion() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());

        Tensor a = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph, BackendIntentPlan.empty());
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
        assertEquals(backend.lowering.LoweringFamily.CPU_NATIVE_REGION, lowered.units().getFirst().loweringFamily());
        RegionExecutionPlan regionPlan = lowered.units().getFirst().requireRegionPlan();
        CpuNativeRegionPayload payload = assertInstanceOf(CpuNativeRegionPayload.class, regionPlan.backendPayload());
        assertEquals(List.of(2), payload.providerNodeIds());
        assertEquals(List.of(3), payload.localKernelNodeIds());
        assertEquals(List.of(3), regionPlan.boundaryOutputNodeIds());
        assertEquals(2, regionPlan.executionGroups().size());
        assertEquals(backend.lowering.region.RegionExecutionKind.PROVIDER_CALL, regionPlan.executionGroups().get(0).executionKind());
        assertEquals(backend.lowering.region.RegionExecutionKind.DIRECT_KERNEL, regionPlan.executionGroups().get(1).executionKind());
    }

    @Test
    void cpuNativeProfileSchedulesViewAsRegionLocalAliasGroup() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());

        Tensor a = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).reshape(32, 128).relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-native-view",
                PartitionTarget.CPU,
                List.of(2, 3, 4),
                List.of(0, 1),
                List.of(GraphValueRef.node(4)),
                List.of(GraphValueRef.node(4))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of())
        ));

        RegionExecutionPlan regionPlan = result.loweredRegion().units().getFirst().requireRegionPlan();
        assertEquals(backend.lowering.LoweringFamily.CPU_NATIVE_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        assertEquals(3, regionPlan.executionGroups().size());
        assertEquals(backend.lowering.region.RegionExecutionKind.PROVIDER_CALL, regionPlan.executionGroups().get(0).executionKind());
        assertEquals(backend.lowering.region.RegionExecutionKind.VIEW, regionPlan.executionGroups().get(1).executionKind());
        assertEquals(backend.lowering.region.RegionExecutionKind.DIRECT_KERNEL, regionPlan.executionGroups().get(2).executionKind());
        assertEquals(List.of(3), regionPlan.executionGroups().get(1).orderedNodeIds());
    }

    @Test
    void cpuNativeRegionPlanPreservesMultipleBoundaryOutputs() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());

        Tensor a = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();

        List<Tensor> graph = relu.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-native-multi-boundary",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(2), GraphValueRef.node(3)),
                List.of(GraphValueRef.node(2), GraphValueRef.node(3))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of())
        ));

        RegionExecutionPlan regionPlan = result.loweredRegion().units().getFirst().requireRegionPlan();
        assertEquals(backend.lowering.LoweringFamily.CPU_NATIVE_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        assertEquals(List.of(2, 3), regionPlan.boundaryOutputNodeIds());
        assertEquals(3, regionPlan.anchorNodeId());
        assertEquals(backend.lowering.region.RegionRole.BOUNDARY_OUTPUT, regionPlan.nodePlans().get(0).regionRole());
        assertEquals(backend.lowering.region.RegionRole.BOUNDARY_OUTPUT, regionPlan.nodePlans().get(1).regionRole());
    }

    @Test
    void cpuNativeCompareBoundaryUsesCpuArrayStorageContract() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());

        Tensor a = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "b", DataType.FLOAT32);
        Tensor threshold = new Tensor(new float[64], new int[]{64, 1}, null, "threshold", DataType.FLOAT32);
        Tensor out = a.matmul(b).greaterThan(threshold);

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph, BackendIntentPlan.empty());
        int matmulNodeId = nodeId(compiledNodes, Operation.OpType.MATMUL);
        int compareNodeId = nodeId(compiledNodes, Operation.OpType.GT);
        List<Integer> externalInputNodeIds = externalInputNodeIds(compiledNodes, matmulNodeId, compareNodeId);
        Partition partition = partition(
                "cpu-native-compare-boundary",
                PartitionTarget.CPU,
                List.of(matmulNodeId, compareNodeId),
                externalInputNodeIds,
                List.of(GraphValueRef.node(compareNodeId)),
                List.of(GraphValueRef.node(compareNodeId))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of())
        ));

        RegionExecutionPlan regionPlan = result.loweredRegion().units().getFirst().requireRegionPlan();
        assertEquals(backend.lowering.LoweringFamily.CPU_NATIVE_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        assertEquals(backend.lowering.region.RegionStorageContract.CPU_ARRAY, regionPlan.nodePlans().get(1).storageContract());
        assertEquals(backend.lowering.region.RegionStorageContract.MIXED_BOUNDARY, regionPlan.executionGroups().get(1).storageContract());
        assertEquals(backend.lowering.region.RegionRole.BOUNDARY_OUTPUT, regionPlan.nodePlans().get(1).regionRole());
    }

    @Test
    void cpuNativeSubregionCanStartAfterUnsupportedPrefixInsideCpuRegion() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());

        Tensor prefixInput = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "prefix", DataType.FLOAT32);
        Tensor a = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "b", DataType.FLOAT32);
        Tensor nativeRelu = a.matmul(b).relu();
        Tensor out = prefixInput.erf().add(nativeRelu.erf());

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph, BackendIntentPlan.empty());
        int matmulNodeId = nodeId(compiledNodes, Operation.OpType.MATMUL);
        int reluNodeId = nodeId(compiledNodes, Operation.OpType.RELU);
        int addNodeId = nodeId(compiledNodes, Operation.OpType.ADD);
        int suffixErfNodeId = compiledNodes.stream()
                .filter(node -> node.operation() != null
                        && node.operation().opType() == Operation.OpType.ERF
                        && node.inputIds().contains(reluNodeId))
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
        int prefixErfNodeId = compiledNodes.stream()
                .filter(node -> node.operation() != null
                        && node.operation().opType() == Operation.OpType.ERF
                        && !node.inputIds().contains(reluNodeId))
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
        List<Integer> orderedNodeIds = List.of(prefixErfNodeId, matmulNodeId, reluNodeId, suffixErfNodeId, addNodeId);
        Set<Integer> selectedNodeIds = Set.copyOf(orderedNodeIds);
        Partition partition = partition(
                "cpu-native-prefix-split",
                PartitionTarget.CPU,
                orderedNodeIds,
                externalInputNodeIds(compiledNodes, orderedNodeIds),
                List.of(GraphValueRef.node(addNodeId)),
                List.of(GraphValueRef.node(addNodeId))
        );
        OptimizedRegion region = new OptimizedRegion(
                partition.partitionId(),
                partition,
                partition.target(),
                orderedNodeIds.stream()
                        .map(nodeId -> singleOpUnit(partition, compiledNodes, nodeId, selectedNodeIds))
                        .toList(),
                List.of(),
                List.of(GraphValueRef.node(addNodeId)),
                new RegionOptimizationTrace(List.of("test-native-prefix-split"))
        );

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of())
        ));

        LoweredRegion lowered = result.loweredRegion();
        assertEquals(4, lowered.units().size());
        assertEquals(List.of(prefixErfNodeId), lowered.units().get(0).orderedNodeIds());
        assertEquals(backend.lowering.LoweringFamily.CPU_NATIVE_REGION, lowered.units().get(1).loweringFamily());
        assertEquals(List.of(matmulNodeId, reluNodeId), lowered.units().get(1).orderedNodeIds());
        assertEquals(List.of(suffixErfNodeId), lowered.units().get(2).orderedNodeIds());
        assertEquals(List.of(addNodeId), lowered.units().get(3).orderedNodeIds());

        RegionExecutionPlan nativePlan = lowered.units().get(1).requireRegionPlan();
        CpuNativeRegionPayload payload = assertInstanceOf(CpuNativeRegionPayload.class, nativePlan.backendPayload());
        assertEquals(List.of(matmulNodeId), payload.providerNodeIds());
        assertEquals(List.of(reluNodeId), payload.localKernelNodeIds());
        assertEquals(List.of(reluNodeId), nativePlan.boundaryOutputNodeIds());
    }

    @Test
    void autoProfileRejectsNativeRegionWhenLocalKernelIsOnlySlowSegmentKernel() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());

        Tensor a = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph, BackendIntentPlan.empty());
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

        assertEquals(backend.lowering.LoweringFamily.BLAS, result.loweredRegion().units().getFirst().loweringFamily());
    }

    @Test
    void fusedUnitUsesExternalValueNodeExecutionInputForViewChain() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor out = base.select(0, 1).relu().exp();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph, BackendIntentPlan.empty());
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
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        target,
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

    private static int nodeId(List<CompiledNode> compiledNodes, Operation.OpType opType) {
        return compiledNodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static List<Integer> externalInputNodeIds(List<CompiledNode> compiledNodes, int matmulNodeId, int compareNodeId) {
        CompiledNode matmul = compiledNodes.stream()
                .filter(node -> node.id() == matmulNodeId)
                .findFirst()
                .orElseThrow();
        CompiledNode compare = compiledNodes.stream()
                .filter(node -> node.id() == compareNodeId)
                .findFirst()
                .orElseThrow();
        java.util.LinkedHashSet<Integer> inputs = new java.util.LinkedHashSet<>(matmul.inputIds());
        compare.inputIds().stream()
                .filter(id -> id != matmulNodeId)
                .forEach(inputs::add);
        return List.copyOf(inputs);
    }

    private static List<Integer> externalInputNodeIds(List<CompiledNode> compiledNodes, List<Integer> selectedNodeIds) {
        Set<Integer> selected = Set.copyOf(selectedNodeIds);
        java.util.LinkedHashSet<Integer> inputs = new java.util.LinkedHashSet<>();
        for (int nodeId : selectedNodeIds) {
            compiledNodes.stream()
                    .filter(node -> node.id() == nodeId)
                    .findFirst()
                    .orElseThrow()
                    .inputIds()
                    .stream()
                    .filter(inputId -> !selected.contains(inputId))
                    .forEach(inputs::add);
        }
        return List.copyOf(inputs);
    }

    private static ExecutionUnit singleOpUnit(
            Partition partition,
            List<CompiledNode> compiledNodes,
            int nodeId,
            Set<Integer> selectedNodeIds
    ) {
        CompiledNode node = compiledNodes.stream()
                .filter(candidate -> candidate.id() == nodeId)
                .findFirst()
                .orElseThrow();
        return new ExecutionUnit(
                partition.partitionId() + "-unit-" + nodeId,
                ExecutionUnitKind.UNIT_KERNEL,
                partition.target(),
                node.inputIds().stream()
                        .filter(selectedNodeIds::contains)
                        .map(GraphValueRef::node)
                        .toList(),
                List.of(GraphValueRef.node(nodeId)),
                partition.requiredMaterializedValueRefs().contains(GraphValueRef.node(nodeId))
                        ? List.of(GraphValueRef.node(nodeId))
                        : List.of(),
                partition.outputValueRefs().contains(GraphValueRef.node(nodeId))
                        || partition.requiredMaterializedValueRefs().contains(GraphValueRef.node(nodeId))
                        ? List.of()
                        : List.of(GraphValueRef.node(nodeId)),
                List.of(nodeId),
                Math.max(1L, node.flatDataSize()),
                node.inputIds().stream()
                        .filter(inputId -> !selectedNodeIds.contains(inputId))
                        .toList(),
                new RegionOptimizationTrace(List.of("test-single-op:" + nodeId))
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
}
