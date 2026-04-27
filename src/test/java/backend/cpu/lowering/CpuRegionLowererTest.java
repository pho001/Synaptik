package backend.cpu.lowering;

import backend.ComputeBackend;
import backend.blas.BlasProvider;
import backend.cpu.fused.plan.FusedOperationPreparation;
import backend.lowering.BackendCapabilities;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringContext;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import config.optimizer.FuseConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.memory.MemoryPlanner;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionBoundaryReason;
import graph.optimizer.partition.PartitionEdge;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.partition.PartitionValueRef;
import graph.optimizer.region.DefaultRegionOptimizer;
import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.RegionOptimizationContext;
import org.junit.jupiter.api.Test;
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
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        Partition partition = partition(
                "cpu-fused",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(PartitionValueRef.ofNode(3)),
                List.of(PartitionValueRef.ofNode(3))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes)
        ));

        LoweredRegion lowered = result.loweredRegion();
        assertNotNull(lowered);
        assertEquals(backend.lowering.LoweringFamily.FUSED_NATIVE, lowered.units().getFirst().loweringFamily());
        assertInstanceOf(FusedOperationPreparation.class, lowered.units().getFirst().artifact());
    }

    @Test
    void lowersSingleMatmulUnitToBlasWhenBlasIsEnabledAndWorkIsLargeEnough() {
        Tensor a = new Tensor(new float[128 * 256], new int[]{128, 256}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[256 * 512], new int[]{256, 512}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        Partition partition = partition(
                "cpu-matmul",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(PartitionValueRef.ofNode(3)),
                List.of(PartitionValueRef.ofNode(3))
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
                new LoweringContext(blasRuntime, compiledNodes)
        ));

        assertEquals(backend.lowering.LoweringFamily.BLAS, result.loweredRegion().units().getFirst().loweringFamily());
    }

    @Test
    void fusedUnitUsesBackingTensorExecutionInputForViewChain() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor out = base.select(0, 1).relu().exp();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        Partition partition = partition(
                "cpu-view-chain",
                PartitionTarget.CPU,
                List.of(1, 2, 3),
                List.of(0),
                List.of(PartitionValueRef.ofNode(3)),
                List.of(PartitionValueRef.ofNode(3))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        CpuRegionLowerer lowerer = new CpuRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes)
        ));

        LoweredRegion lowered = result.loweredRegion();
        assertNotNull(lowered);
        assertEquals(backend.lowering.LoweringFamily.FUSED_NATIVE, lowered.units().getLast().loweringFamily());
        assertEquals(List.of(0), lowered.units().getLast().inputNodeIds());
        FusedOperationPreparation preparation = assertInstanceOf(FusedOperationPreparation.class, lowered.units().getLast().artifact());
        assertEquals(1, preparation.runtimeInputs().size());
        assertEquals(base, preparation.runtimeInputs().getFirst());
    }

    private static Partition partition(
            String id,
            PartitionTarget target,
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            List<PartitionValueRef> outputValueRefs,
            List<PartitionValueRef> requiredMaterialized
    ) {
        List<PartitionValue> values = orderedNodeIds.stream()
                .map(nodeId -> new PartitionValue(PartitionValueRef.ofNode(nodeId), nodeId))
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
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(orderedNodeIds.size(), internalEdges.size(), externalInputNodeIds.size(), 0, Math.max(0, orderedNodeIds.size() - 1)),
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
}
