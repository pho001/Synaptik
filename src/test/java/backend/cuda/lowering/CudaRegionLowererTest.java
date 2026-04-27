package backend.cuda.lowering;

import backend.ComputeBackend;
import backend.lowering.BackendCapabilities;
import backend.lowering.LoweringContext;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import config.optimizer.FuseConfig;
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
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import graph.optimizer.region.DefaultRegionOptimizer;
import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.RegionOptimizationContext;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class CudaRegionLowererTest {
    @Test
    void lowersPureElementwiseGpuCudaRegionToCudaFusedElementwiseGraph() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).relu().exp();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        Partition partition = partition(
                "cuda-fused",
                PartitionTarget.GPU_CUDA,
                List.of(2, 3, 4),
                List.of(0, 1),
                List.of(PartitionValueRef.ofNode(4)),
                List.of(PartitionValueRef.ofNode(4))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        CudaGpuPartitionPlan plan = new CudaGpuPartitionPlan(
                4,
                new AcceleratorSubgraphSpec(
                        2,
                        List.of(2, 3, 4),
                        List.of(
                                new AcceleratorSubgraphOp(2, operations.Operation.OpType.ADD),
                                new AcceleratorSubgraphOp(3, operations.Operation.OpType.RELU),
                                new AcceleratorSubgraphOp(4, operations.Operation.OpType.EXP)
                        ),
                        List.of(0, 1),
                        List.of(4)
                ),
                new AcceleratorDagSpec(
                        List.of(
                                new AcceleratorDagInput(0, List.of(4), DataType.FLOAT32),
                                new AcceleratorDagInput(1, List.of(4), DataType.FLOAT32)
                        ),
                        List.of(
                                new AcceleratorDagNode(2, AcceleratorDagNodeType.ADD, AcceleratorDagValueRef.externalInput(0), AcceleratorDagValueRef.externalInput(1), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, 4, 1, 1, 1),
                                new AcceleratorDagNode(3, AcceleratorDagNodeType.RELU, AcceleratorDagValueRef.nodeOutput(0), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, 4, 1, 1, 1),
                                new AcceleratorDagNode(4, AcceleratorDagNodeType.EXP, AcceleratorDagValueRef.nodeOutput(1), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, 4, 1, 1, 1)
                        ),
                        List.of(2),
                        List.of(4)
                ),
                12L
        );

        CudaRegionLowerer lowerer = new CudaRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_CUDA)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), plan))
        ));

        assertNotNull(result);
        assertNotNull(result.loweredRegion());
        assertEquals(backend.lowering.LoweringFamily.CUDA_FUSED_ELEMENTWISE_GRAPH, result.loweredRegion().units().getFirst().loweringFamily());
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
                orderedNodeIds.getLast(),
                requiredMaterialized,
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                orderedNodeIds.size(),
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(orderedNodeIds.size(), internalEdges.size(), externalInputNodeIds.size(), 0, Math.max(0, orderedNodeIds.size() - 1)),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        target,
                        orderedNodeIds.getLast(),
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
