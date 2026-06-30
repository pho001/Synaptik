package backend.metal.lowering;

import backend.contract.ComputeBackend;
import backend.accelerator.lowering.AcceleratorMatMulSpec;
import backend.accelerator.lowering.AcceleratorSubgraphLoweringResult;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import backend.accelerator.lowering.AcceleratorPartitionPlan;
import backend.accelerator.dag.AcceleratorSubgraphSpec;

import java.util.List;
import java.util.Objects;

/**
 * Metal partition plan carrying the lowered accelerator result selected for execution.
 *
 * @param anchorNodeId node that triggers execution of this partition
 * @param subgraph original accelerator subgraph candidate
 * @param lowering lowered DAG and metadata consumed by the Metal bridge
 */
public record MetalPartitionPlan(
        int anchorNodeId,
        AcceleratorSubgraphSpec subgraph,
        AcceleratorSubgraphLoweringResult lowering
) implements AcceleratorPartitionPlan {
    public MetalPartitionPlan {
        Objects.requireNonNull(subgraph, "subgraph cannot be null");
        Objects.requireNonNull(lowering, "lowering cannot be null");
        if (!subgraph.orderedNodeIds().contains(anchorNodeId)) {
            throw new IllegalArgumentException("anchorNodeId must be part of nodeIds");
        }
    }

    /**
     * Returns the node ids covered by this partition in execution order.
     */
    public List<Integer> nodeIds() {
        return subgraph.orderedNodeIds();
    }

    /**
     * Returns node ids read from outside the partition.
     */
    public List<Integer> externalInputNodeIds() {
        return subgraph.externalInputNodeIds();
    }

    /**
     * Returns node ids produced by this partition.
     */
    public List<Integer> producedOutputNodeIds() {
        return subgraph.outputNodeIds();
    }

    /**
     * Returns {@link ComputeBackend#GPU_METAL}.
     */
    @Override
    public ComputeBackend backend() {
        return ComputeBackend.GPU_METAL;
    }

    /**
     * Returns the optional legacy matmul descriptor associated with the lowering.
     */
    public AcceleratorMatMulSpec matMulSpec() {
        return lowering.matMulSpec();
    }

    /**
     * Returns the compute node id selected by accelerator lowering.
     */
    public int computeNodeId() {
        return lowering.computeNodeId();
    }

    /**
     * Returns the planner work estimate for this partition.
     */
    public long estimatedWork() {
        return lowering.estimatedWork();
    }

    /**
     * Returns the Java-side lowered-region manifest selected for this Metal plan.
     */
    public GpuLoweredRegionManifest manifest() {
        return lowering.manifest();
    }

    @Override
    public GpuLoweredRegionManifest gpuLoweredRegionManifest() {
        return manifest();
    }
}
