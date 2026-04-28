package backend.cuda.lowering;

import backend.ComputeBackend;
import graph.optimizer.partition.PartitionPlan;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorSubgraphSpec;

import java.util.List;
import java.util.Objects;

/**
 * CUDA partition plan carrying the lowered accelerator DAG selected for execution.
 *
 * @param anchorNodeId node that triggers execution of this partition
 * @param subgraph original accelerator subgraph candidate
 * @param dagSpec lowered DAG consumed by the CUDA bridge
 * @param estimatedWork planner work estimate used by backend selection
 */
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

    /**
     * Returns {@link ComputeBackend#GPU_CUDA}.
     */
    @Override
    public ComputeBackend backend() {
        return ComputeBackend.GPU_CUDA;
    }

    /**
     * Returns the node ids covered by this partition in execution order.
     */
    @Override
    public List<Integer> nodeIds() {
        return subgraph.orderedNodeIds();
    }

    /**
     * Returns node ids read from outside the partition.
     */
    @Override
    public List<Integer> externalInputNodeIds() {
        return subgraph.externalInputNodeIds();
    }

    /**
     * Returns node ids produced by this partition.
     */
    @Override
    public List<Integer> producedOutputNodeIds() {
        return subgraph.outputNodeIds();
    }
}
