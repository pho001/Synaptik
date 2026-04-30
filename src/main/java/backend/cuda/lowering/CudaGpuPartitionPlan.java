package backend.cuda.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
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
 * @param compoundSummary stable compound GPU pattern summary for trace and preparation metadata
 */
public record CudaGpuPartitionPlan(
        int anchorNodeId,
        AcceleratorSubgraphSpec subgraph,
        AcceleratorDagSpec dagSpec,
        long estimatedWork,
        GpuCompoundRegionSummary compoundSummary
) implements PartitionPlan {
    public CudaGpuPartitionPlan {
        Objects.requireNonNull(subgraph, "subgraph cannot be null");
        Objects.requireNonNull(dagSpec, "dagSpec cannot be null");
        estimatedWork = Math.max(0L, estimatedWork);
        compoundSummary = compoundSummary == null
                ? GpuCompoundRegionSummary.none(ComputeBackend.GPU_CUDA, subgraph.orderedNodeIds())
                : compoundSummary;
        if (!subgraph.orderedNodeIds().contains(anchorNodeId)) {
            throw new IllegalArgumentException("anchorNodeId must be part of nodeIds");
        }
    }

    public CudaGpuPartitionPlan(
            int anchorNodeId,
            AcceleratorSubgraphSpec subgraph,
            AcceleratorDagSpec dagSpec,
            long estimatedWork
    ) {
        this(anchorNodeId, subgraph, dagSpec, estimatedWork, null);
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
