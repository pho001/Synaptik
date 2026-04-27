package backend.metal.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.AcceleratorMatMulSpec;
import backend.accelerator.lowering.AcceleratorSubgraphLoweringResult;
import graph.optimizer.partition.PartitionPlan;
import backend.accelerator.dag.AcceleratorSubgraphSpec;

import java.util.List;
import java.util.Objects;

public record MetalPartitionPlan(
        int anchorNodeId,
        AcceleratorSubgraphSpec subgraph,
        AcceleratorSubgraphLoweringResult lowering
) implements PartitionPlan {
    public MetalPartitionPlan {
        Objects.requireNonNull(subgraph, "subgraph cannot be null");
        Objects.requireNonNull(lowering, "lowering cannot be null");
        if (!subgraph.orderedNodeIds().contains(anchorNodeId)) {
            throw new IllegalArgumentException("anchorNodeId must be part of nodeIds");
        }
    }

    public List<Integer> nodeIds() {
        return subgraph.orderedNodeIds();
    }

    public List<Integer> externalInputNodeIds() {
        return subgraph.externalInputNodeIds();
    }

    public List<Integer> producedOutputNodeIds() {
        return subgraph.outputNodeIds();
    }

    @Override
    public ComputeBackend backend() {
        return ComputeBackend.GPU_METAL;
    }

    public AcceleratorMatMulSpec matMulSpec() {
        return lowering.matMulSpec();
    }

    public int computeNodeId() {
        return lowering.computeNodeId();
    }

    public long estimatedWork() {
        return lowering.estimatedWork();
    }
}
