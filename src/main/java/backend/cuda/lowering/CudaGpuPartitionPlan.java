package backend.cuda.lowering;

import backend.ComputeBackend;
import graph.optimizer.partition.PartitionPlan;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorSubgraphSpec;

import java.util.List;
import java.util.Objects;

public record CudaGpuPartitionPlan(
        int anchorNodeId,
        AcceleratorSubgraphSpec subgraph,
        AcceleratorDagSpec dagSpec,
        long estimatedWork
) implements PartitionPlan {
    public CudaGpuPartitionPlan {
        Objects.requireNonNull(subgraph, "subgraph cannot be null");
        Objects.requireNonNull(dagSpec, "dagSpec cannot be null");
        estimatedWork = Math.max(0L, estimatedWork);
        if (!subgraph.orderedNodeIds().contains(anchorNodeId)) {
            throw new IllegalArgumentException("anchorNodeId must be part of nodeIds");
        }
    }

    @Override
    public ComputeBackend backend() {
        return ComputeBackend.GPU_CUDA;
    }

    @Override
    public List<Integer> nodeIds() {
        return subgraph.orderedNodeIds();
    }

    @Override
    public List<Integer> externalInputNodeIds() {
        return subgraph.externalInputNodeIds();
    }

    @Override
    public List<Integer> producedOutputNodeIds() {
        return subgraph.outputNodeIds();
    }
}
