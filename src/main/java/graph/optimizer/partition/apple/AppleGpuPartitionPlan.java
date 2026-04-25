package graph.optimizer.partition.apple;

import backend.ComputeBackend;
import graph.optimizer.partition.AcceleratorPartitionPlan;
import graph.optimizer.partition.model.AcceleratorSubgraphSpec;

import java.util.List;
import java.util.Objects;

public record AppleGpuPartitionPlan(
        int anchorNodeId,
        AcceleratorSubgraphSpec subgraph,
        AppleGpuSubgraphLoweringResult lowering
) implements AcceleratorPartitionPlan {
    public AppleGpuPartitionPlan {
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

    public AppleGpuMatMulSpec matMulSpec() {
        return lowering.matMulSpec();
    }

    public int computeNodeId() {
        return lowering.computeNodeId();
    }

    public long estimatedWork() {
        return lowering.estimatedWork();
    }
}
